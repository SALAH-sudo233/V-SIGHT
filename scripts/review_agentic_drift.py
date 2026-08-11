#!/usr/bin/env python3
"""仅限本机访问的盲化视觉漂移审核工具。"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import mimetypes
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "data/e3/agentic_drift_review/agentic_drift_review_queue.jsonl.gz"
DEFAULT_IMAGES = Path("/home/u2025141034/benchmark/benchmark_images")
DEFAULT_OUTPUT = ROOT / "data/e3/agentic_drift_review/agentic_drift_reviews.jsonl"
DEFAULT_REVIEWER = "项目负责人"
ANNOTATION_PROTOCOL = "single_project_owner"
SUPPORTED_QUEUE_ROLES = frozenset({
    "development_agentic_audit_not_training",
    "development_agentic_natural_audit_not_training",
})
REVIEWER_PATTERN = re.compile(r"^[\w.-]{1,64}$")
EVIDENCE_STATES = (
    "SUPPORTED_CORRECT", "WRONG_INSTANCE", "ABSENT_UNSUPPORTED",
    "UNOBSERVABLE_AMBIGUOUS",
)
PARSER_STATES = ("correct", "incorrect", "uncertain")
ATOM_STATES = ("supported", "contradicted", "unobservable")
ATOMS = ("identity", "attribute", "action", "relation")
REFERENCE_VISIBILITY = ("visible", "partially_visible", "not_visible")
WRONG_INSTANCE_SUBTYPES = (
    "target_attribute", "target_action", "target_reference_relation",
    "reference_identity", "role_reversal", "correct_target_wrong_witness", "other",
)
FORBIDDEN_QUEUE_FIELDS = frozenset({
    "model", "task", "sample_id", "source_record_id", "agent_evidence_state",
    "agent_action", "drift_risk_raw", "review_priority", "reason_codes",
    "alternative_bbox", "claim_support", "alternative_support", "binding_margin",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_queue(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    seen = set()
    for row in rows:
        annotation_id = str(row.get("annotation_id") or "")
        if not annotation_id or annotation_id in seen:
            raise ValueError(f"标注编号缺失或重复：{annotation_id}")
        seen.add(annotation_id)
        if row.get("data_role") not in SUPPORTED_QUEUE_ROLES:
            raise ValueError("审核队列的数据角色不符合预期")
        leaked = FORBIDDEN_QUEUE_FIELDS & set(row)
        if leaked:
            raise ValueError(f"审核队列泄漏了 agent 字段：{sorted(leaked)}")
    return rows


def read_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_reviews(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {}
    for row in read_reviews(path):
        key = (str(row.get("annotation_id") or ""), str(row.get("reviewer_id") or ""))
        if all(key):
            latest[key] = row
    return latest


def valid_box(value: Any, width: int, height: int) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(item) for item in (x1, y1, x2, y2))
        and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    )


def validate_submission(
    payload: dict[str, Any],
    known: dict[str, dict[str, Any]],
    image_sizes: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    annotation_id = str(payload.get("annotation_id") or "")
    if annotation_id not in known:
        raise ValueError("未知的标注编号")
    reviewer_id = str(payload.get("reviewer_id") or "").strip()
    if not REVIEWER_PATTERN.fullmatch(reviewer_id):
        raise ValueError("审核人编号格式无效")
    if reviewer_id != DEFAULT_REVIEWER:
        raise ValueError("single_project_owner 仅接受项目负责人审核")
    status = str(payload.get("status") or "draft")
    if status not in {"draft", "completed"}:
        raise ValueError("审核状态无效")
    state = payload.get("evidence_state") or None
    parser_state = payload.get("parser_status") or None
    subtype = payload.get("wrong_instance_subtype") or None
    if state not in (*EVIDENCE_STATES, None):
        raise ValueError("证据状态无效")
    if parser_state not in (*PARSER_STATES, None):
        raise ValueError("查询解析状态无效")
    if subtype not in (*WRONG_INSTANCE_SUBTYPES, None):
        raise ValueError("错误实例子类型无效")

    original = known[annotation_id]
    parsed_applicable = tuple(
        str(value) for value in original.get("stage_b_applicable_atoms") or ATOMS
    )
    # A reviewer who marks parsing incorrect/uncertain must be able to recover
    # atoms omitted by the inference-time parser instead of being locked to its
    # original applicability mask.
    applicable = ATOMS if parser_state in {"incorrect", "uncertain"} else parsed_applicable
    atoms = payload.get("atom_states") or {}
    if not isinstance(atoms, dict) or any(atom not in ATOMS for atom in atoms):
        raise ValueError("原子状态必须使用已定义的三态映射")
    if any(value not in (*ATOM_STATES, None, "") for value in atoms.values()):
        raise ValueError("原子证据状态无效")
    image_name = Path(str(original["image_filename"])).name
    width, height = image_sizes[image_name]
    corrected = payload.get("corrected_target_bbox_xyxy")
    reference = payload.get("reference_bbox_xyxy")
    if corrected not in (None, "") and not valid_box(corrected, width, height):
        raise ValueError("目标校正框无效或超出图像范围")
    if reference not in (None, "") and not valid_box(reference, width, height):
        raise ValueError("关系参照框无效或超出图像范围")
    reference_visibility = payload.get("reference_visibility") or None
    if reference_visibility not in (*REFERENCE_VISIBILITY, None):
        raise ValueError("参照对象可见性无效")
    confidence = payload.get("confidence")
    confidence_value = None if confidence in (None, "") else float(confidence)
    if confidence_value is not None and not 0.0 <= confidence_value <= 1.0:
        raise ValueError("置信度必须位于 0 到 1 之间")

    normalized_atoms = {atom: atoms.get(atom) or None for atom in applicable}
    if status == "completed":
        if state not in EVIDENCE_STATES or parser_state not in PARSER_STATES or confidence_value is None:
            raise ValueError("完成审核前必须填写证据状态、查询解析状态和置信度")
        if any(normalized_atoms.get(atom) not in ATOM_STATES for atom in applicable):
            raise ValueError("完成审核前必须填写全部适用的原子证据")
        values = list(normalized_atoms.values())
        if state == "SUPPORTED_CORRECT" and any(value != "supported" for value in values):
            raise ValueError("支持且定位正确要求全部适用原子均为支持")
        if state == "WRONG_INSTANCE":
            if subtype not in WRONG_INSTANCE_SUBTYPES:
                raise ValueError("错误实例绑定必须选择子类型")
            if normalized_atoms.get("identity") != "supported":
                raise ValueError("错误实例绑定要求目标类别身份得到支持")
            observable_conflict = any(
                normalized_atoms.get(atom) == "contradicted"
                for atom in applicable if atom != "identity"
            )
            if not observable_conflict and parser_state == "correct":
                raise ValueError("错误实例绑定需要存在矛盾的绑定原子或查询解析错误")
        elif subtype is not None:
            raise ValueError("只有错误实例绑定状态可以填写错误实例子类型")
        if state == "ABSENT_UNSUPPORTED" and "contradicted" not in values:
            raise ValueError("目标缺失或证据不支持要求至少存在一项矛盾证据")
        if state == "UNOBSERVABLE_AMBIGUOUS" and "unobservable" not in values and parser_state != "uncertain":
            raise ValueError("不可观测或歧义要求存在不可观测证据或不确定的查询解析")
        relation_state = normalized_atoms.get("relation")
        if relation_state in {"supported", "contradicted"}:
            if not valid_box(reference, width, height):
                raise ValueError("可观测关系必须绘制独立的关系参照框")
            if reference_visibility not in {"visible", "partially_visible"}:
                raise ValueError("可观测关系要求参照对象可见或部分可见")
        if state != "WRONG_INSTANCE" and corrected not in (None, ""):
            raise ValueError("只有错误实例绑定状态可以填写目标校正框")

    return {
        "schema_version": "vsight_agentic_drift_review_v1",
        "annotation_protocol": ANNOTATION_PROTOCOL,
        "review_authority": "project_owner",
        "annotation_id": annotation_id,
        "reviewer_id": reviewer_id,
        "status": status,
        "parser_status": parser_state,
        "evidence_state": state,
        "wrong_instance_subtype": subtype,
        "atom_states": normalized_atoms,
        "corrected_target_bbox_xyxy": corrected if corrected not in (None, "") else None,
        "reference_bbox_xyxy": reference if reference not in (None, "") else None,
        "reference_visibility": reference_visibility,
        "confidence": confidence_value,
        "evidence_note": str(payload.get("evidence_note") or "")[:4000],
        "source_queue_sha256": str(payload.get("source_queue_sha256") or ""),
        "reviewed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy_action_assigned": False,
        "training_eligible": False,
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-SIGHT Drift Review</title><style>
:root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#18222c;background:#edf1f4}*{box-sizing:border-box}body{margin:0}button,input,select,textarea{font:inherit}header{height:58px;display:flex;align-items:center;gap:12px;padding:0 16px;background:#fff;border-bottom:1px solid #c8d0d7;position:sticky;top:0;z-index:4}h1{font-size:17px;margin:0}.grow{flex:1}.toolbar{display:flex;gap:8px;align-items:center;padding:9px 16px;background:#f8fafb;border-bottom:1px solid #c8d0d7;position:sticky;top:58px;z-index:3}button,select,input,textarea{border:1px solid #adb8c2;border-radius:5px;background:#fff}button{height:34px;padding:0 11px;font-weight:650;cursor:pointer}.primary{background:#176b52;color:#fff;border-color:#176b52}select,input{height:34px;padding:0 8px}.picker{min-width:300px}.status{font-size:13px}.ok{color:#176b52}.error{color:#b42318}main{display:grid;grid-template-columns:minmax(520px,1.2fr) minmax(390px,.8fr);min-height:calc(100vh - 111px)}.visual{padding:15px;border-right:1px solid #c8d0d7}.form{padding:15px 20px;background:#fff}.query{font-size:19px;margin:0 0 5px}.meta{font-size:12px;color:#5c6974;margin-bottom:10px}.stage{position:relative;background:#222931;border:1px solid #939fa9;border-radius:6px;overflow:auto;max-height:calc(100vh - 235px);line-height:0}.stage img{display:block;width:100%;height:auto}.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.field{margin:9px 0}.field label{display:block;font-size:12px;font-weight:700;margin-bottom:4px}.field select,.field textarea{width:100%}.field textarea{min-height:58px;padding:8px;resize:vertical}.section{border-bottom:1px solid #d9dfe4;padding-bottom:13px;margin-bottom:13px}.section h2{font-size:14px;margin:0 0 8px}.atoms{display:grid;grid-template-columns:1fr 1fr;gap:7px}.atoms label{display:flex;align-items:center;justify-content:space-between;gap:7px;padding:7px;border:1px solid #d0d7dd;border-radius:4px;font-size:13px}.atoms select{max-width:145px}.drawbar{display:flex;gap:8px;align-items:end}.drawbar .field{flex:1}.actions{display:flex;gap:8px;align-items:center}.badge{font-size:12px}@media(max-width:920px){main{grid-template-columns:1fr}.visual{border-right:0;border-bottom:1px solid #c8d0d7}.stage{max-height:65vh}.picker{min-width:180px;max-width:50vw}}
</style></head><body><header><h1>V-SIGHT Drift Review</h1><span id="progress" class="status">加载中</span><span class="grow"></span><label>审核人 <input id="reviewer" value="project_owner"></label></header><div class="toolbar"><button id="prev" title="上一条">←</button><button id="next" title="下一条">→</button><select id="filter"><option value="pending">待审核</option><option value="all">全部</option><option value="completed">已完成</option></select><select id="picker" class="picker"></select><span class="grow"></span><span id="saveStatus" class="status"></span></div><main><section class="visual"><h2 id="query" class="query">加载中</h2><div id="meta" class="meta"></div><div id="stage" class="stage"><img id="image" alt="review image"><canvas id="canvas"></canvas></div><div class="drawbar"><div class="field"><label>框类型</label><select id="drawMode"><option value="corrected">目标校正框（蓝）</option><option value="reference">关系参照框（黄）</option></select></div><button id="clearBox" title="清除当前类型框">×</button></div></section><section class="form"><div class="section"><h2>Evidence State</h2><div class="field"><label>Parser</label><select id="parserStatus"><option value="">未标注</option><option value="correct">correct</option><option value="incorrect">incorrect</option><option value="uncertain">uncertain</option></select></div><div class="field"><label>证据状态</label><select id="evidenceState"><option value="">未标注</option><option value="SUPPORTED_CORRECT">SUPPORTED_CORRECT</option><option value="WRONG_INSTANCE">WRONG_INSTANCE</option><option value="ABSENT_UNSUPPORTED">ABSENT_UNSUPPORTED</option><option value="UNOBSERVABLE_AMBIGUOUS">UNOBSERVABLE_AMBIGUOUS</option></select></div><div class="field"><label>Wrong-instance subtype</label><select id="subtype"><option value="">不适用</option><option value="target_attribute">target_attribute</option><option value="target_action">target_action</option><option value="target_reference_relation">target_reference_relation</option><option value="reference_identity">reference_identity</option><option value="role_reversal">role_reversal</option><option value="correct_target_wrong_witness">correct_target_wrong_witness</option><option value="other">other</option></select></div></div><div class="section"><h2>Atom Evidence</h2><div class="atoms" id="atoms"><label>身份/类别<select data-atom="identity"><option value="">未标注</option><option value="supported">supported</option><option value="contradicted">contradicted</option><option value="unobservable">unobservable</option></select></label><label>属性<select data-atom="attribute"><option value="">未标注</option><option value="supported">supported</option><option value="contradicted">contradicted</option><option value="unobservable">unobservable</option></select></label><label>动作/状态<select data-atom="action"><option value="">未标注</option><option value="supported">supported</option><option value="contradicted">contradicted</option><option value="unobservable">unobservable</option></select></label><label>目标-参照关系<select data-atom="relation"><option value="">未标注</option><option value="supported">supported</option><option value="contradicted">contradicted</option><option value="unobservable">unobservable</option></select></label></div><div class="field"><label>参照可见性</label><select id="referenceVisibility"><option value="">未标注</option><option value="visible">visible</option><option value="partially_visible">partially_visible</option><option value="not_visible">not_visible</option></select></div></div><div class="section"><div class="field"><label>置信度 <output id="confidenceValue">0.90</output></label><input id="confidence" type="range" min="0" max="1" step="0.01" value="0.90" style="width:100%"></div><div class="field"><label>证据备注</label><textarea id="note"></textarea></div></div><div class="actions"><span id="badge" class="badge"></span><span class="grow"></span><button id="draft">保存草稿</button><button id="complete" class="primary">完成并下一条</button></div></section></main><script>
const $=id=>document.getElementById(id);let rows=[],visible=[],current=null,index=0,draft={},drag=null,queueHash='';function reviewer(){return $('reviewer').value.trim()||'project_owner'}function msg(v,e=false){$('saveStatus').textContent=v;$('saveStatus').className='status '+(e?'error':'ok')}function reset(){draft={parser_status:null,evidence_state:null,wrong_instance_subtype:null,atom_states:{},corrected_target_bbox_xyxy:null,reference_bbox_xyxy:null,reference_visibility:null,confidence:.9,evidence_note:''};$('parserStatus').value='';$('evidenceState').value='';$('subtype').value='';$('atoms').querySelectorAll('select').forEach(x=>{x.value='';x.disabled=false});$('referenceVisibility').value='';$('confidence').value=.9;$('confidenceValue').value='0.90';$('note').value=''}function capture(){draft.parser_status=$('parserStatus').value||null;draft.evidence_state=$('evidenceState').value||null;draft.wrong_instance_subtype=$('subtype').value||null;draft.atom_states=Object.fromEntries([...$('atoms').querySelectorAll('select:not(:disabled)')].map(x=>[x.dataset.atom,x.value||null]));draft.reference_visibility=$('referenceVisibility').value||null;draft.confidence=Number($('confidence').value);draft.evidence_note=$('note').value}function draw(){const c=$('canvas'),img=$('image');if(!img.naturalWidth||!current)return;c.width=img.naturalWidth;c.height=img.naturalHeight;const x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);const width=Math.max(3,c.width/400);for(const [box,color] of [[current.original_bbox_xyxy,'#ef4444'],[draft.corrected_target_bbox_xyxy,'#38bdf8'],[draft.reference_bbox_xyxy,'#facc15']]){if(!box)continue;x.strokeStyle=color;x.lineWidth=width;x.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1])}}function render(){reset();$('query').textContent=current.query;$('meta').textContent=`${current.annotation_id} · ${current.query_stratum} · family=${current.relation_family||'none'} · reference=${current.reference_phrase||'none'}`;$('image').src='/image/'+current.index;const applicable=current.stage_b_applicable_atoms||[];$('atoms').querySelectorAll('select').forEach(x=>x.disabled=!applicable.includes(x.dataset.atom));const r=current.saved_review;if(r){draft={...r};$('parserStatus').value=r.parser_status||'';$('evidenceState').value=r.evidence_state||'';$('subtype').value=r.wrong_instance_subtype||'';$('atoms').querySelectorAll('select').forEach(x=>x.value=(r.atom_states||{})[x.dataset.atom]||'');$('referenceVisibility').value=r.reference_visibility||'';$('confidence').value=r.confidence??.9;$('confidenceValue').value=Number(r.confidence??.9).toFixed(2);$('note').value=r.evidence_note||''}$('badge').textContent=r?.status==='completed'?'已完成':r?'草稿':'待审核';draw()}function applyFilter(){const f=$('filter').value;visible=rows.filter(r=>f==='all'||(f==='completed'&&r.status==='completed')||(f==='pending'&&r.status!=='completed'));$('picker').innerHTML='';visible.forEach(r=>{const o=document.createElement('option');o.value=r.index;o.textContent=`${String(r.index+1).padStart(2,'0')} | ${r.status} | ${r.query_stratum}`;$('picker').appendChild(o)});if(!visible.some(r=>r.index===index)&&visible.length)index=visible[0].index;$('picker').value=String(index)}async function load(i){capture();index=Math.max(0,Math.min(rows.length-1,i));const r=await fetch(`/api/item/${index}?reviewer_id=${encodeURIComponent(reviewer())}`);if(!r.ok){msg('加载失败',true);return}current=await r.json();render();$('picker').value=String(index);msg('')}async function refresh(open=true){const r=await fetch(`/api/state?reviewer_id=${encodeURIComponent(reviewer())}`),d=await r.json();rows=d.groups;queueHash=d.source_queue_sha256;$('progress').textContent=`已完成 ${d.completed} / ${d.total}`;applyFilter();if(open&&rows.length)await load(index)}async function save(status){capture();const body={annotation_id:current.annotation_id,reviewer_id:reviewer(),status,source_queue_sha256:queueHash,...draft};const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok){msg(d.error||'保存失败',true);return false}msg(status==='completed'?'已保存':'草稿已保存');await refresh(false);return true}$('image').onload=draw;$('confidence').oninput=e=>$('confidenceValue').value=Number(e.target.value).toFixed(2);$('picker').onchange=e=>load(Number(e.target.value));$('filter').onchange=()=>{applyFilter();if(visible.length)load(index)};$('reviewer').onchange=()=>refresh(true);$('prev').onclick=()=>{const p=visible.findIndex(r=>r.index===index);if(p>0)load(visible[p-1].index)};$('next').onclick=()=>{const p=visible.findIndex(r=>r.index===index);if(visible[p+1])load(visible[p+1].index)};$('draft').onclick=()=>save('draft');$('complete').onclick=async()=>{const old=index;if(await save('completed')){const n=rows.find(r=>r.index>old&&r.status!=='completed')||rows.find(r=>r.status!=='completed');if(n)load(n.index)}};$('clearBox').onclick=()=>{draft[$('drawMode').value==='reference'?'reference_bbox_xyxy':'corrected_target_bbox_xyxy']=null;draw()};$('canvas').onpointerdown=e=>{const r=$('canvas').getBoundingClientRect(),sx=$('canvas').width/r.width,sy=$('canvas').height/r.height;drag={x1:(e.clientX-r.left)*sx,y1:(e.clientY-r.top)*sy,x2:0,y2:0};$('canvas').setPointerCapture(e.pointerId)};$('canvas').onpointermove=e=>{if(!drag)return;const r=$('canvas').getBoundingClientRect(),sx=$('canvas').width/r.width,sy=$('canvas').height/r.height;drag.x2=(e.clientX-r.left)*sx;drag.y2=(e.clientY-r.top)*sy;const box=[Math.min(drag.x1,drag.x2),Math.min(drag.y1,drag.y2),Math.max(drag.x1,drag.x2),Math.max(drag.y1,drag.y2)];draft[$('drawMode').value==='reference'?'reference_bbox_xyxy':'corrected_target_bbox_xyxy']=box;draw()};$('canvas').onpointerup=()=>drag=null;refresh(true);
</script></body></html>'''

WORKFLOW_CSS = r'''
.image-layer{position:relative;width:100%;min-height:1px;line-height:0}.image-layer img{display:block;width:100%;height:auto}.image-layer canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.workflow-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-bottom:1px solid #d4dbe1;margin-bottom:18px}
.step-button{height:50px;border:0;border-bottom:3px solid transparent;border-radius:0;background:transparent;color:#66737e;display:flex;align-items:center;justify-content:center;gap:7px;padding:0 8px;font-size:13px}
.step-button.active{color:#176b52;border-bottom-color:#176b52}.step-button:disabled{cursor:not-allowed;opacity:.45}.step-index{display:inline-grid;place-items:center;width:23px;height:23px;border:1px solid currentColor;border-radius:50%;font-size:12px}.workflow-panel[hidden]{display:none}.workflow-panel h2{font-size:16px;margin:0 0 14px}.workflow-panel{min-height:290px}.legend{display:flex;gap:16px;align-items:center;margin:10px 0 2px;font-size:12px;color:#54616c}.legend span{display:inline-flex;gap:6px;align-items:center}.swatch{width:16px;height:4px;border-radius:0}.swatch.upstream{background:#ef4444}.swatch.corrected{background:#38bdf8}.swatch.reference{background:#facc15}.form>.actions{position:sticky;bottom:0;background:#fff;padding:12px 0 4px;border-top:1px solid #d9dfe4}.step-nav{min-width:76px}.step-nav[hidden],#complete[hidden]{display:none}.workflow-panel .section{border:0;padding:0;margin:0}@media(max-width:560px){.step-button{font-size:12px}.step-index{display:none}.workflow-steps{margin-bottom:12px}.workflow-panel{min-height:240px}}
'''

WORKFLOW_JS = r'''
function setupReviewWorkflow(){
  const form=document.querySelector('.form'),sections=[...form.querySelectorAll(':scope > .section')],actions=form.querySelector(':scope > .actions');
  if(sections.length!==3||!actions)return;
  const imageLayer=document.createElement('div'),stage=$('stage');imageLayer.className='image-layer';stage.insertBefore(imageLayer,$('image'));imageLayer.append($('image'),$('canvas'));
  const parserField=$('parserStatus').closest('.field'),stateField=$('evidenceState').closest('.field'),subtypeField=$('subtype').closest('.field'),referenceField=$('referenceVisibility').closest('.field');
  const workflow=document.createElement('div');workflow.className='workflow';workflow.innerHTML=`<nav class="workflow-steps" aria-label="审核步骤"><button type="button" class="step-button active" data-step="1"><span class="step-index">1</span><span>核对查询解析</span></button><button type="button" class="step-button" data-step="2" disabled><span class="step-index">2</span><span>标注原子证据</span></button><button type="button" class="step-button" data-step="3" disabled><span class="step-index">3</span><span>给出审核结论</span></button></nav><section class="workflow-panel" data-panel="1"><h2>步骤一：核对查询解析</h2></section><section class="workflow-panel" data-panel="2" hidden><h2>步骤二：标注红框内的证据</h2></section><section class="workflow-panel" data-panel="3" hidden><h2>步骤三：给出证据结论</h2></section>`;
  const panels=[...workflow.querySelectorAll('.workflow-panel')];panels[0].append(parserField);panels[1].append($('atoms'),referenceField);panels[2].append(stateField,subtypeField);while(sections[2].firstChild)panels[2].append(sections[2].firstChild);form.insertBefore(workflow,actions);sections.forEach(x=>x.remove());
  const legend=document.createElement('div');legend.className='legend';legend.innerHTML='<span><i class="swatch upstream"></i>上游原框</span><span><i class="swatch corrected"></i>目标校正框</span><span><i class="swatch reference"></i>关系参照框</span>';document.querySelector('.drawbar').before(legend);
  const back=document.createElement('button'),forward=document.createElement('button');back.type=forward.type='button';back.id='stepBack';forward.id='stepForward';back.className=forward.className='step-nav';back.textContent='上一步';forward.textContent='下一步';actions.insertBefore(back,$('draft'));actions.insertBefore(forward,$('draft'));$('complete').textContent='完成审核';
  const stepButtons=[...workflow.querySelectorAll('.step-button')];let step=1,unlocked=1;
  function syncApplicable(){const parsed=current?.stage_b_applicable_atoms||[],applicable=$('parserStatus').value==='correct'?parsed:['identity','attribute','action','relation'];$('atoms').querySelectorAll('select').forEach(x=>{const active=applicable.includes(x.dataset.atom);x.disabled=!active;x.closest('label').style.display=active?'flex':'none'});referenceField.hidden=!applicable.includes('relation')}
  function syncConclusion(){const wrong=$('evidenceState').value==='WRONG_INSTANCE';subtypeField.hidden=!wrong;if(!wrong)$('subtype').value=''}
  function validateStep(value){capture();if(value===1&&!$('parserStatus').value){msg('请先选择查询解析状态',true);return false}if(value===2){const missing=[...$('atoms').querySelectorAll('select:not(:disabled)')].some(x=>!x.value);if(missing){msg('请填写全部适用的原子证据',true);return false}const relation=$('atoms').querySelector('[data-atom="relation"]');if(relation&&!relation.disabled&&['supported','contradicted'].includes(relation.value)){if(!draft.reference_bbox_xyxy){msg('请在图中绘制黄色关系参照框',true);return false}if(!['visible','partially_visible'].includes($('referenceVisibility').value)){msg('请选择参照对象的可见性',true);return false}}}if(value===3){if(!$('evidenceState').value){msg('请选择最终证据状态',true);return false}if($('evidenceState').value==='WRONG_INSTANCE'&&!$('subtype').value){msg('请选择错误实例子类型',true);return false}}msg('');return true}
  window.goReviewStep=function(value){step=Math.max(1,Math.min(3,value));panels.forEach((x,i)=>x.hidden=i!==step-1);stepButtons.forEach((x,i)=>{x.classList.toggle('active',i===step-1);x.disabled=i+1>unlocked});back.hidden=step===1;forward.hidden=step===3;$('complete').hidden=step!==3;syncApplicable();syncConclusion();if(step===2&&!referenceField.hidden)$('drawMode').value='reference';if(step===3&&$('evidenceState').value==='WRONG_INSTANCE')$('drawMode').value='corrected'};
  window.resetReviewWorkflow=function(){unlocked=1;window.goReviewStep(1)};
  stepButtons.forEach(button=>button.onclick=()=>{const target=Number(button.dataset.step);if(target<=unlocked)window.goReviewStep(target)});back.onclick=()=>window.goReviewStep(step-1);forward.onclick=()=>{if(validateStep(step)){unlocked=Math.max(unlocked,step+1);window.goReviewStep(step+1)}};$('parserStatus').onchange=syncApplicable;$('evidenceState').onchange=syncConclusion;
  const originalComplete=$('complete').onclick;$('complete').onclick=async event=>{if(validateStep(3))await originalComplete(event)};window.resetReviewWorkflow();
}
'''

# 数据枚举保持稳定，所有面向审核人的文字在输出页面中使用中文。
HTML = (
    HTML.replace("V-SIGHT Drift Review", "V-SIGHT 视觉漂移审核")
    .replace('value="project_owner"', 'value="项目负责人"')
    .replace("||'project_owner'", "||'项目负责人'")
    .replace('alt="review image"', 'alt="审核图像"')
    .replace("Evidence State", "证据状态")
    .replace(">Parser<", ">查询解析<")
    .replace("Wrong-instance subtype", "错误实例子类型")
    .replace("Atom Evidence", "原子证据")
    .replace('value="correct">correct<', 'value="correct">正确<')
    .replace('value="incorrect">incorrect<', 'value="incorrect">错误<')
    .replace('value="uncertain">uncertain<', 'value="uncertain">不确定<')
    .replace('value="SUPPORTED_CORRECT">SUPPORTED_CORRECT<', 'value="SUPPORTED_CORRECT">支持且定位正确<')
    .replace('value="WRONG_INSTANCE">WRONG_INSTANCE<', 'value="WRONG_INSTANCE">错误实例绑定<')
    .replace('value="ABSENT_UNSUPPORTED">ABSENT_UNSUPPORTED<', 'value="ABSENT_UNSUPPORTED">目标缺失或证据不支持<')
    .replace('value="UNOBSERVABLE_AMBIGUOUS">UNOBSERVABLE_AMBIGUOUS<', 'value="UNOBSERVABLE_AMBIGUOUS">不可观测或歧义<')
    .replace('value="target_attribute">target_attribute<', 'value="target_attribute">目标属性绑定错误<')
    .replace('value="target_action">target_action<', 'value="target_action">目标动作或状态绑定错误<')
    .replace('value="target_reference_relation">target_reference_relation<', 'value="target_reference_relation">目标与参照关系绑定错误<')
    .replace('value="reference_identity">reference_identity<', 'value="reference_identity">参照对象身份错误<')
    .replace('value="role_reversal">role_reversal<', 'value="role_reversal">目标与参照角色反转<')
    .replace('value="correct_target_wrong_witness">correct_target_wrong_witness<', 'value="correct_target_wrong_witness">目标正确但证据对象错误<')
    .replace('value="other">other<', 'value="other">其他<')
    .replace('value="supported">supported<', 'value="supported">支持<')
    .replace('value="contradicted">contradicted<', 'value="contradicted">矛盾<')
    .replace('value="unobservable">unobservable<', 'value="unobservable">不可观测<')
    .replace('value="visible">visible<', 'value="visible">可见<')
    .replace('value="partially_visible">partially_visible<', 'value="partially_visible">部分可见<')
    .replace('value="not_visible">not_visible<', 'value="not_visible">不可见<')
    .replace(
        "<script>\nconst $=",
        "<script>\nconst STRATUM_ZH={relation:'关系',attribute:'属性',action:'动作或状态',object:'对象'};"
        "const FAMILY_ZH={directional:'方向','symmetric/proximity':'对称或邻近','support/contact':'支撑或接触','role/interaction':'角色或交互',between:'位于两者之间',unknown:'未知'};"
        "const STATUS_ZH={pending:'待审核',draft:'草稿',completed:'已完成'};const $=",
    )
    .replace(
        "`${current.annotation_id} · ${current.query_stratum} · family=${current.relation_family||'none'} · reference=${current.reference_phrase||'none'}`",
        "`${current.annotation_id} · ${STRATUM_ZH[current.query_stratum]||current.query_stratum} · 关系族=${FAMILY_ZH[current.relation_family]||'无'} · 参照=${current.reference_phrase||'无'}`",
    )
    .replace(
        "`${String(r.index+1).padStart(2,'0')} | ${r.status} | ${r.query_stratum}`",
        "`${String(r.index+1).padStart(2,'0')} | ${STATUS_ZH[r.status]||r.status} | ${STRATUM_ZH[r.query_stratum]||r.query_stratum}`",
    )
)
HTML = (
    HTML.replace("</style>", WORKFLOW_CSS + "</style>")
    .replace(
        "current=await r.json();render();$('picker')",
        "current=await r.json();render();window.resetReviewWorkflow?.();$('picker')",
    )
    .replace(
        "$('canvas').onpointerup=()=>drag=null;refresh(true);",
        "$('canvas').onpointerup=()=>drag=null;" + WORKFLOW_JS + "setupReviewWorkflow();refresh(true);",
    )
)


class Handler(BaseHTTPRequestHandler):
    server: "ReviewApp"

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/state":
            reviewer = (query.get("reviewer_id") or [DEFAULT_REVIEWER])[0]
            return self._json(self.server.state(reviewer))
        match = re.fullmatch(r"/api/item/(\d+)", parsed.path)
        if match:
            reviewer = (query.get("reviewer_id") or [DEFAULT_REVIEWER])[0]
            try:
                return self._json(self.server.item(int(match.group(1)), reviewer))
            except (IndexError, ValueError) as exc:
                return self._json({"error": str(exc)}, 404)
        match = re.fullmatch(r"/image/(\d+)", parsed.path)
        if match:
            try:
                row = self.server.rows[int(match.group(1))]
                image = self.server.image_dir / Path(str(row["image_filename"])).name
                body = image.read_bytes()
            except (IndexError, FileNotFoundError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(image))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urllib.parse.urlparse(self.path).path != "/api/save":
            self.send_error(404)
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            self._json({"record": self.server.save(payload)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)

    def log_message(self, format: str, *args: Any) -> None:
        return


class ReviewApp(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, rows, queue_path, output, image_dir):
        super().__init__(address, Handler)
        self.rows = rows
        self.queue_hash = sha256(queue_path)
        self.output = output
        self.image_dir = image_dir
        self.latest = latest_reviews(output)
        self.lock = threading.Lock()
        self.image_sizes = {}
        from PIL import Image
        for row in rows:
            name = Path(str(row["image_filename"])).name
            with Image.open(image_dir / name) as image:
                self.image_sizes[name] = (image.width, image.height)

    def state(self, reviewer: str) -> dict[str, Any]:
        groups = []
        completed = 0
        for index, row in enumerate(self.rows):
            saved = self.latest.get((str(row["annotation_id"]), reviewer))
            status = str(saved.get("status") if saved else "pending")
            completed += status == "completed"
            groups.append({
                "index": index, "annotation_id": row["annotation_id"],
                "query": row["query"], "query_stratum": row["query_stratum"],
                "status": status,
            })
        return {
            "total": len(groups), "completed": completed,
            "source_queue_sha256": self.queue_hash, "groups": groups,
        }

    def item(self, index: int, reviewer: str) -> dict[str, Any]:
        row = dict(self.rows[index])
        row["index"] = index
        row["saved_review"] = self.latest.get((str(row["annotation_id"]), reviewer))
        return row

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if payload.get("source_queue_sha256") != self.queue_hash:
                raise ValueError("审核队列哈希不一致")
            record = validate_submission(
                payload,
                {str(row["annotation_id"]): row for row in self.rows},
                self.image_sizes,
            )
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
            self.latest[(record["annotation_id"], record["reviewer_id"])] = record
            return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = args.queue if args.queue.is_absolute() else ROOT / args.queue
    images = args.images if args.images.is_absolute() else ROOT / args.images
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = read_queue(queue)
    app = ReviewApp((args.host, args.port), rows, queue, output, images)
    print(f"V-SIGHT 视觉漂移审核：http://{args.host}:{args.port}/", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        app.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
