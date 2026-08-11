#!/usr/bin/env python3
"""Loopback-only reviewer for the E3 train binding queue."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
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
DEFAULT_QUEUE = ROOT / "data/e3/binding_annotation_queue/e3_binding_annotation_queue.train.jsonl.gz"
DEFAULT_IMAGES = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")
DEFAULT_OUTPUT = ROOT / "data/e3/binding_annotation_queue/e3_binding_reviews.jsonl"
DEFAULT_REVIEWER = "project_owner"
ANNOTATION_PROTOCOL = "single_project_owner"
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
STAGE_A = ("supported", "contradicted", "ambiguous")
ACTIONS = ("ACCEPT", "REJECT", "RELOCALIZE", "UNCERTAIN")
ATOMS = ("identity", "attribute", "action", "relation")
ATOM_STATES = ("supported", "contradicted", "unobservable")
REFERENCE_VISIBILITY = ("visible", "partially_visible", "not_visible")
ACTION_STAGE_A = {"ACCEPT": "supported", "REJECT": "contradicted", "RELOCALIZE": "supported"}


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
        key = str(row.get("annotation_id") or "")
        if not key or key in seen:
            raise ValueError(f"missing or duplicate annotation_id: {key}")
        seen.add(key)
        if row.get("data_split") != "train":
            raise ValueError("binding review queue must be train-only")
    return rows


def read_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_reviews(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {}
    for row in read_reviews(path):
        key = (str(row.get("annotation_id") or ""), str(row.get("reviewer_id") or ""))
        if all(key):
            latest[key] = row
    return latest


def append_review(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def valid_box(value: Any, width: int, height: int) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return False
    return (
        all(map(lambda v: v == v and abs(v) != float("inf"), (x1, y1, x2, y2)))
        and 0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    )


def validate_submission(
    payload: dict[str, Any], known: dict[str, dict[str, Any]], image_sizes: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    annotation_id = str(payload.get("annotation_id") or "")
    if annotation_id not in known:
        raise ValueError("unknown annotation_id")
    reviewer_id = str(payload.get("reviewer_id") or "").strip()
    if not REVIEWER_PATTERN.fullmatch(reviewer_id):
        raise ValueError("reviewer_id must use letters, digits, dot, dash, or underscore")
    if reviewer_id != DEFAULT_REVIEWER:
        raise ValueError("single_project_owner requires reviewer_id=project_owner")
    status = str(payload.get("status") or "draft")
    if status not in ("draft", "completed"):
        raise ValueError("invalid status")
    stage_a = payload.get("stage_a_target_status")
    action = payload.get("stage_c_action")
    if stage_a not in (*STAGE_A, None, ""):
        raise ValueError("invalid Stage A status")
    if action not in (*ACTIONS, None, ""):
        raise ValueError("invalid Stage C action")
    atoms = payload.get("stage_b_atoms") or {}
    if not isinstance(atoms, dict) or any(atom not in ATOMS for atom in atoms):
        raise ValueError("Stage B atoms must be a tri-state mapping")
    if any(value not in (*ATOM_STATES, None, "") for value in atoms.values()):
        raise ValueError("invalid Stage B atom state")
    confidence = payload.get("confidence")
    if confidence in (None, ""):
        confidence_value = None
    else:
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            raise ValueError("confidence must be numeric")
        if not 0 <= confidence_value <= 1:
            raise ValueError("confidence must be in [0, 1]")
    original = known[annotation_id]
    image_name = Path(str(original["image_filename"])).name
    width, height = image_sizes[image_name]
    corrected = payload.get("corrected_bbox_xyxy")
    if corrected not in (None, "") and not valid_box(corrected, width, height):
        raise ValueError("corrected bbox is outside the image or invalid")
    reference = payload.get("reference_bbox_xyxy")
    if reference not in (None, "") and not valid_box(reference, width, height):
        raise ValueError("reference bbox is outside the image or invalid")
    reference_visibility = payload.get("reference_visibility")
    if reference_visibility not in (*REFERENCE_VISIBILITY, None, ""):
        raise ValueError("invalid reference visibility")
    applicable = set(original.get("stage_b_applicable_atoms") or ATOMS)
    if status == "completed":
        if stage_a not in STAGE_A or action not in ACTIONS or confidence_value is None:
            raise ValueError("completed review requires Stage A, Stage C, and confidence")
        if action == "RELOCALIZE" and not valid_box(corrected, width, height):
            raise ValueError("RELOCALIZE requires a corrected bbox")
        if action != "UNCERTAIN" and ACTION_STAGE_A.get(str(action)) != stage_a:
            raise ValueError("Stage A and Stage C action are inconsistent")
        if any(atoms.get(atom) not in ATOM_STATES for atom in applicable):
            raise ValueError("completed review requires every applicable Stage B atom")
        applicable_states = [atoms.get(atom) for atom in applicable]
        if action == "ACCEPT" and any(value != "supported" for value in applicable_states):
            raise ValueError("ACCEPT requires supported applicable Stage B atoms")
        if action in {"REJECT", "RELOCALIZE"} and "contradicted" not in applicable_states:
            raise ValueError(f"{action} requires a contradicted Stage B atom")
        relation_state = atoms.get("relation")
        if "relation" in applicable and relation_state in {"supported", "contradicted"}:
            if not valid_box(reference, width, height):
                raise ValueError("observable relation atom requires an independent reference bbox")
            if reference_visibility not in {"visible", "partially_visible"}:
                raise ValueError("observable relation atom requires reference visibility")
    return {
        "schema_version": "vsight_e3_binding_review_v2",
        "annotation_protocol": ANNOTATION_PROTOCOL,
        "review_authority": "project_owner",
        "annotation_id": annotation_id,
        "reviewer_id": reviewer_id,
        "status": status,
        "stage_a_target_status": stage_a or None,
        "stage_b_atoms": {
            atom: atoms.get(atom) or None for atom in ATOMS if atom in applicable
        },
        "stage_b_note": str(payload.get("stage_b_note") or "")[:4000],
        "stage_c_action": action or None,
        "corrected_bbox_xyxy": corrected if corrected not in (None, "") else None,
        "reference_bbox_xyxy": reference if reference not in (None, "") else None,
        "reference_visibility": reference_visibility or None,
        "confidence": confidence_value,
        "review_note": str(payload.get("review_note") or "")[:4000],
        "source_queue_sha256": str(payload.get("source_queue_sha256") or ""),
        "reviewed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-SIGHT 绑定复核</title><style>
:root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#1b2733;background:#eef2f5}*{box-sizing:border-box}body{margin:0}button,input,select,textarea{font:inherit}header{height:62px;display:flex;align-items:center;gap:14px;padding:0 18px;background:#fff;border-bottom:1px solid #c9d1d8;position:sticky;top:0;z-index:5}h1{font-size:18px;margin:0}.grow{flex:1}.toolbar{display:flex;gap:8px;align-items:center;padding:9px 18px;background:#f8fafb;border-bottom:1px solid #c9d1d8;position:sticky;top:62px;z-index:4}button,select,input,textarea{border:1px solid #aeb9c3;border-radius:5px;background:#fff}button{height:34px;padding:0 12px;font-weight:650;cursor:pointer}button.primary{background:#176b52;color:#fff;border-color:#176b52}select,input{height:34px;padding:0 8px}.toolbar select{min-width:300px;max-width:45vw}.status{font-size:13px}.ok{color:#176b52}.error{color:#b42318}main{display:grid;grid-template-columns:minmax(500px,1.15fr) minmax(360px,.85fr);min-height:calc(100vh - 116px)}.visual{padding:16px;border-right:1px solid #c9d1d8}.form{padding:16px 20px;background:#fff}.query{font-size:20px;font-weight:750;margin:0 0 5px}.meta{font-size:12px;color:#5b6874;display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px}.stage{position:relative;background:#20262d;border:1px solid #96a2ad;border-radius:6px;overflow:auto;max-height:calc(100vh - 250px);line-height:0}.stage img{display:block;width:100%;height:auto}.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.hint{font-size:12px;color:#586572;margin-top:8px}.section{border-bottom:1px solid #d9dfe4;padding:0 0 14px;margin-bottom:15px}.section h2{font-size:14px;margin:0 0 9px}.choice{display:flex;gap:8px;flex-wrap:wrap}.choice label{display:flex;align-items:center;gap:5px;border:1px solid #d0d7dd;border-radius:4px;padding:7px 9px;font-size:13px}.atoms{display:grid;grid-template-columns:1fr 1fr;gap:7px}.atoms label{display:flex;gap:7px;align-items:center;padding:8px;border:1px solid #d0d7dd;border-radius:4px;font-size:13px}.field{margin:9px 0}.field label{display:block;font-size:12px;font-weight:700;margin-bottom:5px}.field select,.field textarea{width:100%}.field textarea{min-height:64px;padding:8px;resize:vertical}.actions{display:flex;gap:8px;align-items:center}.badge{font-size:12px}.danger{color:#b42318}@media(max-width:900px){main{grid-template-columns:1fr}.visual{border-right:0;border-bottom:1px solid #c9d1d8}.stage{max-height:65vh}.toolbar{top:62px;flex-wrap:wrap}.toolbar select{min-width:200px;max-width:100%;flex:1}}
</style></head><body><header><h1>V-SIGHT 绑定复核</h1><span id="progress" class="status">加载中</span><span class="grow"></span><label>审核人 <input id="reviewer" value="project_owner" readonly></label></header><div class="toolbar"><button id="prev" title="上一组">←</button><button id="next" title="下一组">→</button><select id="filter"><option value="pending">待审核</option><option value="all">全部</option><option value="completed">已完成</option></select><select id="picker"></select><span class="grow"></span><span id="saveStatus" class="status"></span></div><main><section class="visual"><h2 id="query" class="query">加载中</h2><div id="meta" class="meta"></div><div id="stage" class="stage"><img id="image" alt="待审核图像"><canvas id="canvas"></canvas></div><div class="field"><label>拖框类型</label><select id="drawMode"><option value="corrected">目标校正框（蓝）</option><option value="reference">独立参照框（黄）</option></select></div><div class="hint">红框为上游原框。关系原子可观测时必须独立绘制黄色参照框；RELOCALIZE 必须绘制蓝色目标校正框。不要参考 GT 或 IoU。</div></section><section class="form"><div class="section"><h2>Stage A：完整目标是否存在</h2><div class="choice" id="stageA"><label><input type="radio" name="stageA" value="supported">支持</label><label><input type="radio" name="stageA" value="contradicted">矛盾</label><label><input type="radio" name="stageA" value="ambiguous">歧义</label></div></div><div class="section"><h2>Stage B：逐原子三态证据</h2><div class="atoms" id="atoms"><label>身份/类别<select data-atom="identity"><option value="">未标注</option><option value="supported">supported</option><option value="contradicted">contradicted</option><option value="unobservable">unobservable</option></select></label><label>属性<select data-atom="attribute"><option value="">未标注</option><option value="supported">supported</option><option value="contradicted">contradicted</option><option value="unobservable">unobservable</option></select></label><label>动作/状态<select data-atom="action"><option value="">未标注</option><option value="supported">supported</option><option value="contradicted">contradicted</option><option value="unobservable">unobservable</option></select></label><label>目标-参照关系<select data-atom="relation"><option value="">未标注</option><option value="supported">supported</option><option value="contradicted">contradicted</option><option value="unobservable">unobservable</option></select></label></div><div class="field"><label>参照可见性</label><select id="referenceVisibility"><option value="">未标注</option><option value="visible">visible</option><option value="partially_visible">partially_visible</option><option value="not_visible">not_visible</option></select></div><div class="field"><label>证据备注</label><textarea id="stageBNote"></textarea></div></div><div class="section"><h2>Stage C：后处理动作</h2><div class="choice" id="action"><label><input type="radio" name="action" value="ACCEPT">ACCEPT</label><label><input type="radio" name="action" value="REJECT">REJECT</label><label><input type="radio" name="action" value="RELOCALIZE">RELOCALIZE</label><label><input type="radio" name="action" value="UNCERTAIN">UNCERTAIN</label></div><div class="field"><label for="confidence">置信度：<output id="confidenceValue">0.90</output></label><input id="confidence" type="range" min="0" max="1" step="0.01" value="0.90" style="width:100%"></div><div class="field"><label>复核备注</label><textarea id="reviewNote"></textarea></div></div><div class="actions"><span id="badge" class="badge"></span><span class="grow"></span><button id="draft">保存草稿</button><button id="complete" class="primary">完成并下一组</button></div></section></main><script>
const $=id=>document.getElementById(id),A=['supported','contradicted','ambiguous'],ACT=['ACCEPT','REJECT','RELOCALIZE','UNCERTAIN'],ATOM=['identity','attribute','action','relation'];let rows=[],visible=[],current=null,index=0,draft=null,drag=null,sourceHash='';const atomLabel={identity:'身份/类别',attribute:'属性',action:'动作/状态',relation:'目标-参照关系'};
function reviewer(){return $('reviewer').value.trim()||'project_owner'}function selectedRadio(name){return document.querySelector(`input[name="${name}"]:checked`)?.value||null}function setRadio(name,value){document.querySelectorAll(`input[name="${name}"]`).forEach(x=>x.checked=x.value===value)}function setStatus(s,e=false){$('saveStatus').textContent=s;$('saveStatus').className='status '+(e?'error':'ok')}
function clearForm(){setRadio('stageA',null);setRadio('action',null);$('atoms').querySelectorAll('select').forEach(x=>{x.value='';x.disabled=false});$('referenceVisibility').value='';$('stageBNote').value='';$('reviewNote').value='';$('confidence').value=.9;$('confidenceValue').value='0.90';draft={stage_a_target_status:null,stage_b_atoms:{},stage_b_note:'',stage_c_action:null,corrected_bbox_xyxy:null,reference_bbox_xyxy:null,reference_visibility:null,confidence:.9,review_note:''}}
function draw(){const c=$('canvas'),img=$('image');if(!img.naturalWidth)return;c.width=img.naturalWidth;c.height=img.naturalHeight;const ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const b=current.original_bbox_xyxy;ctx.strokeStyle='#ef4444';ctx.lineWidth=Math.max(3,c.width/400);ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);if(draft?.corrected_bbox_xyxy){const q=draft.corrected_bbox_xyxy;ctx.strokeStyle='#38bdf8';ctx.strokeRect(q[0],q[1],q[2]-q[0],q[3]-q[1])}if(draft?.reference_bbox_xyxy){const q=draft.reference_bbox_xyxy;ctx.strokeStyle='#facc15';ctx.strokeRect(q[0],q[1],q[2]-q[0],q[3]-q[1])}}
function render(){if(!current)return;$('query').textContent=current.query;$('meta').textContent=`${current.annotation_id} · ${current.query_stratum} · family=${current.relation_family||'无'} · reference=${current.reference_phrase||'无'}`;$('image').src='/image/'+current.index;$('badge').textContent=current.saved_review?.status==='completed'?'已完成':current.saved_review?'草稿':'待审核';clearForm();const applicable=current.stage_b_applicable_atoms||ATOM;$('atoms').querySelectorAll('select').forEach(x=>x.disabled=!applicable.includes(x.dataset.atom));const r=current.saved_review;if(r){setRadio('stageA',r.stage_a_target_status);setRadio('action',r.stage_c_action);$('atoms').querySelectorAll('select').forEach(x=>x.value=(r.stage_b_atoms||{})[x.dataset.atom]||'');$('referenceVisibility').value=r.reference_visibility||'';$('stageBNote').value=r.stage_b_note||'';$('reviewNote').value=r.review_note||'';draft={...r};$('confidence').value=r.confidence??.9;$('confidenceValue').value=Number(r.confidence??.9).toFixed(2)}draw();updateNav()}
function capture(){draft.stage_a_target_status=selectedRadio('stageA');draft.stage_b_atoms=Object.fromEntries([...$('atoms').querySelectorAll('select:not(:disabled)')].map(x=>[x.dataset.atom,x.value||null]));draft.reference_visibility=$('referenceVisibility').value||null;draft.stage_b_note=$('stageBNote').value;draft.stage_c_action=selectedRadio('action');draft.confidence=Number($('confidence').value);draft.review_note=$('reviewNote').value}
function applyFilter(){const f=$('filter').value;visible=rows.filter(x=>f==='all'||(f==='completed'&&x.status==='completed')||(f==='pending'&&x.status!=='completed'));$('picker').innerHTML='';visible.forEach(x=>{const o=document.createElement('option');o.value=x.index;o.textContent=`${String(x.index+1).padStart(3,'0')} | ${x.status} | ${x.query_stratum}`;$('picker').appendChild(o)});if(!visible.some(x=>x.index===index)&&visible.length)index=visible[0].index;$('picker').value=String(index);updateNav()}
function updateNav(){const p=visible.findIndex(x=>x.index===index);$('prev').disabled=p<=0;$('next').disabled=p<0||p>=visible.length-1}
async function load(i){capture();index=Math.max(0,Math.min(rows.length-1,i));const r=await fetch(`/api/item/${index}?reviewer_id=${encodeURIComponent(reviewer())}`);if(!r.ok){setStatus('加载失败',true);return}current=await r.json();render();$('picker').value=String(index);setStatus('')}
async function refresh(loadIt=true){const r=await fetch(`/api/state?reviewer_id=${encodeURIComponent(reviewer())}`);const d=await r.json();rows=d.groups;sourceHash=d.source_queue_sha256;$('progress').textContent=`已完成 ${d.completed} / ${d.total}`;applyFilter();if(loadIt&&rows.length)await load(index)}
async function save(status){capture();if(draft.stage_c_action==='RELOCALIZE'&&!draft.corrected_bbox_xyxy){setStatus('RELOCALIZE必须拖出校正框',true);return}if(['supported','contradicted'].includes(draft.stage_b_atoms.relation)&&!draft.reference_bbox_xyxy){setStatus('可观测关系必须拖出参照框',true);return}const p={annotation_id:current.annotation_id,reviewer_id:reviewer(),status,source_queue_sha256:sourceHash,...draft};const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});const d=await r.json();if(!r.ok){setStatus(d.error||'保存失败',true);return}current.saved_review=d.record;$('badge').textContent=status==='completed'?'已完成':'草稿';setStatus(status==='completed'?'已保存':'草稿已保存');await refresh(false)}
$('image').onload=draw;$('confidence').oninput=e=>$('confidenceValue').value=Number(e.target.value).toFixed(2);$('picker').onchange=e=>load(Number(e.target.value));$('filter').onchange=()=>{applyFilter();if(visible.length)load(index)};$('prev').onclick=()=>{const p=visible.findIndex(x=>x.index===index);if(p>0)load(visible[p-1].index)};$('next').onclick=()=>{const p=visible.findIndex(x=>x.index===index);if(p>=0&&visible[p+1])load(visible[p+1].index)};$('draft').onclick=()=>save('draft');$('complete').onclick=async()=>{const old=index;await save('completed');const n=rows.find(x=>x.index>old&&x.status!=='completed')||rows.find(x=>x.status!=='completed');if(n)load(n.index)};$('reviewer').onchange=()=>refresh(true);
$('canvas').onpointerdown=e=>{if(!current)return;const r=$('canvas').getBoundingClientRect(),sx=$('canvas').width/r.width,sy=$('canvas').height/r.height;drag={x1:(e.clientX-r.left)*sx,y1:(e.clientY-r.top)*sy,x2:(e.clientX-r.left)*sx,y2:(e.clientY-r.top)*sy};$('canvas').setPointerCapture(e.pointerId)};$('canvas').onpointermove=e=>{if(!drag)return;const r=$('canvas').getBoundingClientRect(),sx=$('canvas').width/r.width,sy=$('canvas').height/r.height;drag.x2=(e.clientX-r.left)*sx;drag.y2=(e.clientY-r.top)*sy;const q=[Math.min(drag.x1,drag.x2),Math.min(drag.y1,drag.y2),Math.max(drag.x1,drag.x2),Math.max(drag.y1,drag.y2)];draft[$('drawMode').value==='reference'?'reference_bbox_xyxy':'corrected_bbox_xyxy']=q;draw()};$('canvas').onpointerup=()=>{drag=null};refresh(true);
</script></body></html>'''


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server: "ReviewApp"

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path == "/":
            return self._html()
        if path == "/api/state":
            reviewer = (query.get("reviewer_id") or [DEFAULT_REVIEWER])[0]
            return self._json(self.server.state(reviewer))
        match = re.fullmatch(r"/api/item/(\d+)", path)
        if match:
            reviewer = (query.get("reviewer_id") or [DEFAULT_REVIEWER])[0]
            try:
                return self._json(self.server.item(int(match.group(1)), reviewer))
            except (IndexError, ValueError) as exc:
                return self._json({"error": str(exc)}, 404)
        match = re.fullmatch(r"/image/(\d+)", path)
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
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            record = self.server.save(payload)
            self._json({"record": record})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)

    def log_message(self, format: str, *args: Any) -> None:
        return


class ReviewApp(ReviewServer):
    def __init__(self, address, rows, queue_path, output, image_dir):
        super().__init__(address, Handler)
        self.rows = rows
        self.queue_path = queue_path
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
            status = saved.get("status", "pending") if saved else "pending"
            completed += status == "completed"
            groups.append({"index": index, "annotation_id": row["annotation_id"], "query": row["query"], "query_stratum": row["query_stratum"], "status": status})
        return {"total": len(groups), "completed": completed, "source_queue_sha256": self.queue_hash, "groups": groups}

    def item(self, index: int, reviewer: str) -> dict[str, Any]:
        row = dict(self.rows[index])
        row["index"] = index
        row["saved_review"] = self.latest.get((str(row["annotation_id"]), reviewer))
        row.pop("stage_a_target_status", None)
        row.pop("stage_b_atoms", None)
        row.pop("stage_c_action", None)
        return row

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if payload.get("source_queue_sha256") != self.queue_hash:
                raise ValueError("queue hash mismatch")
            record = validate_submission(payload, {str(row["annotation_id"]): row for row in self.rows}, self.image_sizes)
            append_review(self.output, record)
            self.latest[(record["annotation_id"], record["reviewer_id"])] = record
            return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = args.queue if args.queue.is_absolute() else ROOT / args.queue
    images = args.images if args.images.is_absolute() else ROOT / args.images
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = read_queue(queue)
    app = ReviewApp((args.host, args.port), rows, queue, output, images)
    print(f"V-SIGHT binding review: http://{args.host}:{args.port}/", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        app.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
