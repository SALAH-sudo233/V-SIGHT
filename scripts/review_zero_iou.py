#!/usr/bin/env python3
"""Loopback-only visual review UI for the 127-group IoU=0 audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import mimetypes
import re
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/audits/zero_iou_127.template.jsonl"
DEFAULT_IMAGES = Path("/home/u2025141034/benchmark/benchmark_images")
DEFAULT_OUTPUT = ROOT / "data/audits/zero_iou_127.reviews.jsonl"

TASKS = ("t2_vqa_grounding", "t4_caption_grounding")
FAILURE_MODES = (
    "same_category_wrong_instance",
    "target_reference_role_swap",
    "wrong_category",
    "partial_or_oversized_region",
    "background_or_unannotated",
    "false_rejection",
    "annotation_or_gt_issue",
    "visually_ambiguous",
    "other",
)
PREFERRED_ACTIONS = ("keep", "switch", "reject", "both_wrong", "ambiguous")
AMBIGUITY_LEVELS = ("clear", "mild", "high", "unresolvable")
QUERY_SUPPORT = ("supported", "unsupported", "ambiguous")
BINDING_EVIDENCE = (
    "object_identity",
    "attribute",
    "action_or_state",
    "left_right_or_depth",
    "target_reference_relation",
    "count",
    "localization_tightness",
    "none_visible",
)
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_groups(manifest: Path, image_dir: Path) -> list[dict[str, Any]]:
    groups = read_jsonl(manifest)
    seen: set[str] = set()
    for index, group in enumerate(groups):
        base_id = str(group.get("base_sample_id") or "")
        if not base_id or base_id in seen:
            raise ValueError(f"group {index}: missing or duplicate base_sample_id")
        seen.add(base_id)
        cases = group.get("cases")
        if not isinstance(cases, list) or {case.get("task") for case in cases} != set(TASKS):
            raise ValueError(f"{base_id}: expected exactly one T2 and one T4 case")
        if len(cases) != len(TASKS):
            raise ValueError(f"{base_id}: duplicate task case")
        image_name = Path(str(group.get("image_filename") or "")).name
        if not image_name:
            raise ValueError(f"{base_id}: missing image_filename")
        group["image_filename"] = image_name
        group["image_exists"] = (image_dir / image_name).is_file()
    return groups


def load_latest_reviews(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        base_id = str(row.get("base_sample_id") or "")
        reviewer_id = str(row.get("reviewer_id") or "")
        if base_id and reviewer_id:
            latest[(base_id, reviewer_id)] = row
    return latest


def _short_text(value: Any, field: str, limit: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise ValueError(f"{field} exceeds {limit} characters")
    return text


def validate_submission(
    request: dict[str, Any],
    known_groups: set[str],
) -> dict[str, Any]:
    base_id = str(request.get("base_sample_id") or "")
    if base_id not in known_groups:
        raise ValueError("unknown base_sample_id")
    reviewer_id = str(request.get("reviewer_id") or "").strip()
    if not REVIEWER_PATTERN.fullmatch(reviewer_id):
        raise ValueError("reviewer_id must use 1-64 letters, digits, dot, dash, or underscore")
    status = str(request.get("status") or "draft")
    if status not in {"draft", "completed"}:
        raise ValueError("status must be draft or completed")
    query_support = request.get("query_support")
    if query_support not in (*QUERY_SUPPORT, None, ""):
        raise ValueError("unsupported query_support value")
    raw_cases = request.get("case_reviews")
    if not isinstance(raw_cases, dict):
        raise ValueError("case_reviews must be an object")

    cases: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        raw = raw_cases.get(task) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{task}: review must be an object")
        failure_mode = raw.get("failure_mode")
        preferred_action = raw.get("preferred_action")
        ambiguity = raw.get("ambiguity")
        evidence = raw.get("binding_evidence") or []
        if failure_mode not in (*FAILURE_MODES, None, ""):
            raise ValueError(f"{task}: invalid failure_mode")
        if preferred_action not in (*PREFERRED_ACTIONS, None, ""):
            raise ValueError(f"{task}: invalid preferred_action")
        if ambiguity not in (*AMBIGUITY_LEVELS, None, ""):
            raise ValueError(f"{task}: invalid ambiguity")
        if not isinstance(evidence, list) or any(item not in BINDING_EVIDENCE for item in evidence):
            raise ValueError(f"{task}: invalid binding_evidence")
        cases[task] = {
            "failure_mode": failure_mode or None,
            "preferred_action": preferred_action or None,
            "binding_evidence": sorted(set(evidence)),
            "ambiguity": ambiguity or None,
            "notes": _short_text(raw.get("notes"), f"{task}.notes"),
        }

    if status == "completed":
        if query_support not in QUERY_SUPPORT:
            raise ValueError("completed review requires query_support")
        for task, case in cases.items():
            missing = [
                field
                for field in ("failure_mode", "preferred_action", "ambiguity")
                if not case[field]
            ]
            if missing:
                raise ValueError(f"{task}: completed review missing {', '.join(missing)}")

    return {
        "base_sample_id": base_id,
        "reviewer_id": reviewer_id,
        "status": status,
        "query_support": query_support or None,
        "case_reviews": cases,
        "group_notes": _short_text(request.get("group_notes"), "group_notes"),
    }


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-SIGHT IoU=0 人工审计</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:#19222c;background:#eef1f4;color-scheme:light}
*{box-sizing:border-box}body{margin:0;min-width:320px}button,input,select,textarea{font:inherit;color:inherit}
header{height:64px;display:flex;align-items:center;gap:18px;padding:0 20px;background:#fff;border-bottom:1px solid #cbd2d9;position:sticky;top:0;z-index:20}
h1{font-size:18px;line-height:1;margin:0;white-space:nowrap}.metric{font-size:13px;color:#53606d}.spacer{flex:1}
.reviewer{display:flex;align-items:center;gap:8px;font-size:13px}.reviewer input{width:140px}
button,.button,select,input,textarea{border:1px solid #aeb8c2;background:#fff;border-radius:5px}
button{height:34px;padding:0 12px;cursor:pointer;font-weight:600}button:hover{background:#f2f5f7}button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:#176b52;border-color:#176b52;color:#fff}button.primary:hover{background:#125640}
input,select{height:34px;padding:0 9px}textarea{width:100%;min-height:70px;padding:9px;resize:vertical;line-height:1.4}
.toolbar{min-height:54px;display:flex;align-items:center;gap:8px;padding:9px 20px;background:#f8f9fa;border-bottom:1px solid #ccd3da;position:sticky;top:64px;z-index:19}
.toolbar select#groupPicker{min-width:390px;max-width:52vw}.status{font-size:13px;font-weight:650;min-width:160px;text-align:right}.ok{color:#176b52}.error{color:#b42318}
main{display:grid;grid-template-columns:minmax(460px,1.25fr) minmax(390px,.75fr);min-height:calc(100vh - 118px)}
.visual{padding:16px 18px 24px;border-right:1px solid #cbd2d9;min-width:0}.form{padding:16px 20px 32px;background:#fff;min-width:0}
.query{font-size:20px;font-weight:700;line-height:1.25;margin:0 0 6px}.submeta{display:flex;gap:14px;flex-wrap:wrap;color:#5a6672;font-size:12px;margin-bottom:12px}
.image-tools{height:38px;display:flex;align-items:center;gap:14px;background:#fff;border:1px solid #c7ced5;border-bottom:0;padding:0 10px;border-radius:6px 6px 0 0;font-size:12px}
.image-tools label{display:flex;align-items:center;gap:5px;margin:0}.image-tools input[type=checkbox]{width:15px;height:15px;margin:0}.image-tools input[type=range]{width:110px;height:auto;padding:0;border:0}
.image-viewport{background:#20262d;border:1px solid #9da8b2;border-radius:0 0 6px 6px;overflow:auto;max-height:calc(100vh - 260px);min-height:360px}
.image-stage{position:relative;width:100%;min-width:100%}.image-stage img{display:block;width:100%;height:auto}.image-stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.legend{display:flex;gap:15px;font-size:12px;margin-top:9px}.swatch{display:inline-block;width:12px;height:3px;margin-right:5px;vertical-align:middle}.gt{background:#16a34a}.base{background:#dc2626}.cand{background:#2563eb}
.task-tabs{display:grid;grid-template-columns:1fr 1fr;border:1px solid #aeb8c2;border-radius:6px;overflow:hidden;margin-bottom:14px}.task-tabs button{border:0;border-radius:0;height:46px;font-size:13px}.task-tabs button+button{border-left:1px solid #aeb8c2}.task-tabs button.active{background:#263846;color:#fff}
.case-meta{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}.case-meta div{padding:9px;background:#f3f5f7;border-left:3px solid #aab4bd;font-size:12px;line-height:1.4}.case-meta strong{display:block;font-size:13px}
.hidden{display:none!important}
.field{margin:0 0 14px}.field>label{display:block;font-size:12px;font-weight:750;margin-bottom:6px;text-transform:uppercase;color:#46525e}.field select{width:100%}
.evidence{display:grid;grid-template-columns:1fr 1fr;gap:7px}.evidence label{display:flex;align-items:center;gap:7px;padding:7px;border:1px solid #d1d7dc;border-radius:4px;font-size:12px}.evidence input{width:15px;height:15px;margin:0}
.divider{border:0;border-top:1px solid #d8dde2;margin:18px 0}.footer-actions{display:flex;align-items:center;gap:8px;margin-top:18px}.footer-actions .spacer{flex:1}.complete-badge{font-size:11px;padding:2px 6px;border-radius:3px;background:#dcefe8;color:#125640}.draft-badge{font-size:11px;padding:2px 6px;border-radius:3px;background:#fff0cc;color:#704b00}
@media(max-width:960px){header{height:auto;min-height:64px;flex-wrap:wrap;padding:12px}.toolbar{top:88px;padding:8px 12px;flex-wrap:wrap}.toolbar select#groupPicker{min-width:220px;max-width:100%;flex:1}main{grid-template-columns:1fr}.visual{border-right:0;border-bottom:1px solid #cbd2d9}.image-viewport{max-height:70vh}.form{padding:16px}.status{min-width:0}.reviewer input{width:110px}}
</style></head><body>
<header><h1>V-SIGHT IoU=0 人工审计</h1><span class="metric" id="progress">已完成 0 / 127</span><span class="spacer"></span><label class="reviewer">审核人 <input id="reviewer" value="reviewer_1" autocomplete="off"></label></header>
<div class="toolbar"><button id="prev" title="上一组">&larr;</button><button id="next" title="下一组">&rarr;</button><select id="filter"><option value="pending">待审核</option><option value="all">全部样本</option><option value="completed">已完成</option></select><select id="groupPicker"></select><span class="spacer"></span><span id="saveStatus" class="status"></span></div>
<main><section class="visual"><h2 class="query" id="query">正在加载</h2><div class="submeta"><span id="groupId"></span><span id="target"></span><span id="distractors"></span><span id="queryStatus"></span></div>
<div class="image-tools"><label><input type="checkbox" id="showGt" checked>真值框</label><label><input type="checkbox" id="showBase" checked>基线框</label><label><input type="checkbox" id="showCand" checked>候选框</label><label><input type="checkbox" id="showAuto">自动标签</label><span class="spacer"></span><label>缩放 <input type="range" id="zoom" min="100" max="220" value="100" step="10"><span id="zoomValue">100%</span></label></div>
<div class="image-viewport"><div class="image-stage" id="imageStage"><img id="image" alt="人工审计图像"><svg id="overlay"></svg></div></div><div class="legend"><span><i class="swatch gt"></i>真值框</span><span><i class="swatch base"></i>基线框</span><span><i class="swatch cand"></i>候选框</span></div></section>
<section class="form"><div class="task-tabs"><button id="tabT2" class="active">T2 直接定位</button><button id="tabT4">T4 描述后定位</button></div>
<div class="case-meta"><div><strong id="transition"></strong>结果变化</div><div><strong id="ious"></strong>基线 &rarr; 候选 IoU</div><div class="autoMeta hidden"><strong id="autoBase"></strong>基线框自动分类</div><div class="autoMeta hidden"><strong id="autoCand"></strong>候选框自动分类</div></div>
<div class="field"><label for="failureMode">失败类型</label><select id="failureMode"><option value="">请选择</option></select></div>
<div class="field"><label for="preferredAction">建议决策</label><select id="preferredAction"><option value="">请选择</option></select></div>
<div class="field"><label for="ambiguity">视觉歧义程度</label><select id="ambiguity"><option value="">请选择</option></select></div>
<div class="field"><label>实例绑定证据</label><div class="evidence" id="evidence"></div></div>
<div class="field"><label for="caseNotes">当前任务备注</label><textarea id="caseNotes"></textarea></div><hr class="divider">
<div class="field"><label for="querySupport">正表达是否得到图像支持？</label><select id="querySupport"><option value="">请选择</option><option value="supported">成立</option><option value="unsupported">不成立</option><option value="ambiguous">无法确定</option></select></div>
<div class="field"><label for="groupNotes">样本整体备注</label><textarea id="groupNotes"></textarea></div>
<div class="footer-actions"><span id="savedBadge"></span><span class="spacer"></span><button id="saveDraft">保存草稿</button><button id="complete" class="primary">完成并审核下一组</button></div></section></main>
<script>
const FAILURE_MODES=%FAILURE_MODES%;const ACTIONS=%ACTIONS%;const AMBIGUITIES=%AMBIGUITIES%;const EVIDENCE=%EVIDENCE%;
const TASKS=['t2_vqa_grounding','t4_caption_grounding'];const $=id=>document.getElementById(id);
const LABELS={
failure:{same_category_wrong_instance:'同类错误实例',target_reference_role_swap:'目标与参照物角色互换',wrong_category:'错误物体类别',partial_or_oversized_region:'局部框或过大框',background_or_unannotated:'背景或未标注区域',false_rejection:'错误拒答',annotation_or_gt_issue:'标注或真值框问题',visually_ambiguous:'视觉上无法唯一指代',other:'其他'},
action:{keep:'保留基线框',switch:'切换到候选框',reject:'拒答',both_wrong:'两个框都错误',ambiguous:'无法判断'},
ambiguity:{clear:'目标明确',mild:'轻度歧义',high:'高度歧义',unresolvable:'无法消解'},
evidence:{object_identity:'物体类别或身份',attribute:'颜色/材质/大小等属性',action_or_state:'动作或状态',left_right_or_depth:'左右/前后/远近',target_reference_relation:'目标与参照物关系',count:'数量或顺序',localization_tightness:'框的完整性与紧致度',none_visible:'没有可见区分证据'},
transition:{valid_zero_unresolved:'零 IoU 仍未解决',valid_zero_recovered:'零 IoU 已恢复',nonzero_regressed_to_zero:'从非零退化为零 IoU',nonzero_remained_nonzero:'仍为非零 IoU',false_rejection_unchanged:'仍然错误拒答'},
status:{pending:'待审核',draft:'草稿',completed:'已完成'},
structure:{relation:'关系表达',attribute:'属性表达',object_only:'仅物体表达'}
};
let summaries=[],visible=[],group=null,index=0,currentTask=TASKS[0],draft=null;
function emptyCase(){return{failure_mode:null,preferred_action:null,binding_evidence:[],ambiguity:null,notes:''}}
function reviewer(){return $('reviewer').value.trim()||'reviewer_1'}
function setStatus(text,error=false){$('saveStatus').textContent=text;$('saveStatus').className='status '+(error?'error':'ok')}
function optionize(select,values,labels){values.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=labels[v]||v;select.appendChild(o)})}
optionize($('failureMode'),FAILURE_MODES,LABELS.failure);optionize($('preferredAction'),ACTIONS,LABELS.action);optionize($('ambiguity'),AMBIGUITIES,LABELS.ambiguity);
EVIDENCE.forEach(v=>{const label=document.createElement('label');label.innerHTML=`<input type="checkbox" value="${v}">${LABELS.evidence[v]||v}`;$('evidence').appendChild(label)});
function newDraft(){return{query_support:null,case_reviews:Object.fromEntries(TASKS.map(t=>[t,emptyCase()])),group_notes:''}}
function captureCase(){if(!draft)return;const c=draft.case_reviews[currentTask];c.failure_mode=$('failureMode').value||null;c.preferred_action=$('preferredAction').value||null;c.ambiguity=$('ambiguity').value||null;c.binding_evidence=[...$('evidence').querySelectorAll('input:checked')].map(x=>x.value);c.notes=$('caseNotes').value;draft.query_support=$('querySupport').value||null;draft.group_notes=$('groupNotes').value}
function renderCase(){const c=draft.case_reviews[currentTask]||emptyCase();$('failureMode').value=c.failure_mode||'';$('preferredAction').value=c.preferred_action||'';$('ambiguity').value=c.ambiguity||'';$('caseNotes').value=c.notes||'';$('evidence').querySelectorAll('input').forEach(x=>x.checked=(c.binding_evidence||[]).includes(x.value));$('querySupport').value=draft.query_support||'';$('groupNotes').value=draft.group_notes||'';const data=group.cases.find(x=>x.task===currentTask);$('transition').textContent=LABELS.transition[data.transition]||data.transition;$('ious').textContent=`${data.baseline_iou.toFixed(3)} -> ${data.challenger_iou.toFixed(3)}`;$('autoBase').textContent=LABELS.failure[data.automatic_baseline_class]||'无';$('autoCand').textContent=LABELS.failure[data.automatic_challenger_class]||'无';$('tabT2').classList.toggle('active',currentTask===TASKS[0]);$('tabT4').classList.toggle('active',currentTask===TASKS[1]);drawBoxes()}
function svgNode(name,attrs){const node=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));return node}
function drawBox(svg,box,color,label){if(!Array.isArray(box))return;const [x1,y1,x2,y2]=box;svg.appendChild(svgNode('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:color,'stroke-width':4,'vector-effect':'non-scaling-stroke'}));const text=svgNode('text',{x:x1+5,y:Math.max(17,y1+17),fill:color,'font-size':16,'font-weight':800,stroke:'#fff','stroke-width':3,'paint-order':'stroke'});text.textContent=label;svg.appendChild(text)}
function drawBoxes(){if(!group)return;const img=$('image'),svg=$('overlay');if(!img.naturalWidth)return;svg.setAttribute('viewBox',`0 0 ${img.naturalWidth} ${img.naturalHeight}`);svg.innerHTML='';const c=group.cases.find(x=>x.task===currentTask);if($('showGt').checked)drawBox(svg,c.gt_box,'#16a34a','真值');if($('showBase').checked)drawBox(svg,c.baseline_box,'#dc2626','基线');if($('showCand').checked)drawBox(svg,c.challenger_box,'#2563eb','候选')}
function renderGroup(){draft=group.saved_review?JSON.parse(JSON.stringify(group.saved_review)):newDraft();if(!draft.case_reviews)draft.case_reviews=Object.fromEntries(TASKS.map(t=>[t,emptyCase()]));TASKS.forEach(t=>draft.case_reviews[t]??=emptyCase());$('query').textContent=group.query;$('groupId').textContent=group.base_sample_id;$('target').textContent=`目标类别：${group.target_category||'未知'}`;$('distractors').textContent=`同类干扰实例：${group.same_category_distractors}`;$('queryStatus').textContent=`表达结构：${LABELS.structure[group.expression_structure]||group.expression_structure}`;$('image').src='/image/'+encodeURIComponent(group.base_sample_id);$('image').onload=drawBoxes;$('savedBadge').innerHTML=group.saved_review?(group.saved_review.status==='completed'?'<span class="complete-badge">已完成</span>':'<span class="draft-badge">草稿</span>'):'';renderCase()}
function updateNav(){const pos=visible.findIndex(x=>x.index===index);$('prev').disabled=pos<=0;$('next').disabled=pos<0||pos>=visible.length-1}
async function loadGroup(i){captureCase();index=Math.max(0,Math.min(summaries.length-1,i));const r=await fetch(`/api/group/${index}?reviewer_id=${encodeURIComponent(reviewer())}`);if(!r.ok){setStatus('样本加载失败',true);return}group=await r.json();$('groupPicker').value=String(index);currentTask=TASKS[0];renderGroup();updateNav();setStatus('')}
function applyFilter(){const mode=$('filter').value;visible=summaries.filter(x=>mode==='all'||(mode==='completed'&&x.status==='completed')||(mode==='pending'&&x.status!=='completed'));const picker=$('groupPicker');picker.innerHTML='';visible.forEach(x=>{const o=document.createElement('option');o.value=x.index;o.textContent=`${String(x.index+1).padStart(3,'0')} | ${LABELS.status[x.status]||x.status} | ${x.base_sample_id}`;picker.appendChild(o)});if(!visible.some(x=>x.index===index)&&visible.length)index=visible[0].index;picker.value=String(index);updateNav()}
async function refreshState(load=true){const r=await fetch(`/api/state?reviewer_id=${encodeURIComponent(reviewer())}`);const data=await r.json();summaries=data.groups;$('progress').textContent=`已完成 ${data.completed} / ${data.total}`;applyFilter();if(load&&visible.length)await loadGroup(index)}
function navigate(delta){captureCase();const pos=visible.findIndex(x=>x.index===index);if(pos<0||!visible[pos+delta])return;loadGroup(visible[pos+delta].index)}
async function save(status){captureCase();const payload={base_sample_id:group.base_sample_id,reviewer_id:reviewer(),status,query_support:draft.query_support,case_reviews:draft.case_reviews,group_notes:draft.group_notes};const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await r.json();if(!r.ok){setStatus(data.error||'保存失败',true);return false}group.saved_review=data.record;draft=JSON.parse(JSON.stringify(data.record));$('savedBadge').innerHTML=status==='completed'?'<span class="complete-badge">已完成</span>':'<span class="draft-badge">草稿</span>';setStatus(status==='completed'?'审核已完成':'草稿已保存');await refreshState(false);return true}
$('tabT2').onclick=()=>{captureCase();currentTask=TASKS[0];renderCase()};$('tabT4').onclick=()=>{captureCase();currentTask=TASKS[1];renderCase()};$('prev').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);$('groupPicker').onchange=e=>loadGroup(Number(e.target.value));$('filter').onchange=()=>{applyFilter();if(visible.length)loadGroup(index)};$('saveDraft').onclick=()=>save('draft');$('complete').onclick=async()=>{const old=index;if(await save('completed')){const pending=summaries.find(x=>x.index>old&&x.status!=='completed')||summaries.find(x=>x.status!=='completed');if(pending)loadGroup(pending.index)}};
$('reviewer').onchange=()=>{localStorage.setItem('vsightReviewer',$('reviewer').value);index=0;refreshState(true)};['showGt','showBase','showCand'].forEach(id=>$(id).onchange=drawBoxes);$('showAuto').onchange=e=>document.querySelectorAll('.autoMeta').forEach(x=>x.classList.toggle('hidden',!e.target.checked));$('zoom').oninput=e=>{$('zoomValue').textContent=e.target.value+'%';$('imageStage').style.width=e.target.value+'%'};
$('reviewer').value=localStorage.getItem('vsightReviewer')||'reviewer_1';refreshState(true);
</script></body></html>"""


def html_page() -> bytes:
    replacements = {
        "%FAILURE_MODES%": json.dumps(FAILURE_MODES),
        "%ACTIONS%": json.dumps(PREFERRED_ACTIONS),
        "%AMBIGUITIES%": json.dumps(AMBIGUITY_LEVELS),
        "%EVIDENCE%": json.dumps(BINDING_EVIDENCE),
    }
    page = HTML
    for marker, value in replacements.items():
        page = page.replace(marker, value)
    return page.encode("utf-8")


class AuditServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        groups: list[dict[str, Any]],
        manifest: Path,
        image_dir: Path,
        output: Path,
    ) -> None:
        super().__init__(address, AuditHandler)
        self.groups = groups
        self.by_id = {group["base_sample_id"]: group for group in groups}
        self.known_groups = set(self.by_id)
        self.image_dir = image_dir
        self.output = output
        self.manifest_sha256 = sha256(manifest)
        self.latest = load_latest_reviews(output)
        self.lock = threading.Lock()


class AuditHandler(BaseHTTPRequestHandler):
    server: AuditServer

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[zero-iou-audit] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def reviewer_id(self, query: dict[str, list[str]]) -> str:
        value = (query.get("reviewer_id") or ["reviewer_1"])[0]
        return value if REVIEWER_PATTERN.fullmatch(value) else "reviewer_1"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path == "/":
            payload = html_page()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/api/state":
            reviewer = self.reviewer_id(query)
            summaries = []
            for index, group in enumerate(self.server.groups):
                review = self.server.latest.get((group["base_sample_id"], reviewer))
                summaries.append(
                    {
                        "index": index,
                        "base_sample_id": group["base_sample_id"],
                        "status": review.get("status", "pending") if review else "pending",
                    }
                )
            self.send_json(
                {
                    "reviewer_id": reviewer,
                    "total": len(summaries),
                    "completed": sum(row["status"] == "completed" for row in summaries),
                    "groups": summaries,
                }
            )
            return
        if path.startswith("/api/group/"):
            try:
                index = int(path.rsplit("/", 1)[1])
                group = dict(self.server.groups[index])
            except (ValueError, IndexError):
                self.send_json({"error": "unknown group index"}, HTTPStatus.NOT_FOUND)
                return
            reviewer = self.reviewer_id(query)
            group["saved_review"] = self.server.latest.get((group["base_sample_id"], reviewer))
            self.send_json(group)
            return
        if path.startswith("/image/"):
            base_id = urllib.parse.unquote(path.rsplit("/", 1)[1])
            group = self.server.by_id.get(base_id)
            if group is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            image_path = self.server.image_dir / group["image_filename"]
            if not image_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "image missing")
                return
            payload = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(image_path.name)[0] or "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urllib.parse.urlparse(self.path).path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise ValueError("request body must be between 1 byte and 1 MiB")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            cleaned = validate_submission(request, self.server.known_groups)
            record = {
                "schema_version": "vsight_zero_iou_human_review_v1",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_manifest_sha256": self.server.manifest_sha256,
                **cleaned,
            }
            with self.server.lock:
                append_jsonl(self.server.output, record)
                self.server.latest[(record["base_sample_id"], record["reviewer_id"])] = record
            self.send_json({"ok": True, "record": record})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    groups = load_groups(args.manifest, args.image_dir)
    missing = [group["base_sample_id"] for group in groups if not group["image_exists"]]
    cases = sum(len(group["cases"]) for group in groups)
    if args.check:
        print(
            json.dumps(
                {
                    "groups": len(groups),
                    "cases": cases,
                    "missing_images": missing,
                    "manifest_sha256": sha256(args.manifest),
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if missing:
        raise ValueError(f"cannot start: {len(missing)} audit images are missing")
    server = AuditServer(
        (args.host, args.port),
        groups,
        args.manifest,
        args.image_dir,
        args.output,
    )
    print(
        f"Loaded {len(groups)} groups / {cases} cases. "
        f"Open http://{args.host}:{args.port}/",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
