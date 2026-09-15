import json, datetime, collections, psutil
from pathlib import Path
root=Path(r'C:/Users/30796/Desktop/V-SIGHT-assets/data_quality_audit')
run=root/'auto_repair'/'runs'/'20260911T083146Z'
def read_json(p):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e: return {'_error':type(e).__name__+': '+str(e)}
def jsonl(p):
    out=[]
    try:
        with p.open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                try: out.append(json.loads(line))
                except Exception as e: out.append({'_parse_error':type(e).__name__, '_line':i})
    except Exception as e: print('READERR',p,e)
    return out
print('NOW',datetime.datetime.now().astimezone().isoformat())
for p in [root/'auto_repair'/'current_run.json',root/'auto_repair'/'process.json',run/'heartbeat.json']:
    print('FILE',p, 'mtime', datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat() if p.exists() else 'MISSING')
    print(read_json(p))
procs=[]
for p in psutil.process_iter(['pid','create_time','cmdline','name']):
    try:
        info=p.info; cmd=' '.join(info.get('cmdline') or [])
        if info['pid'] in [28452,760] or any(x in cmd.lower() for x in ['auto_repair\\runner.py','auto_repair/runner.py','auto_repair.runner','data_quality_audit']):
            procs.append((info['pid'],info['create_time'],info.get('name'),cmd[:500]))
    except (psutil.AccessDenied,psutil.NoSuchProcess): pass
print('PROCS',procs)
ar=jsonl(root/'audit_v2.jsonl'); print('AUDIT_LINES',len(ar)); print('AUDIT_KEYS',len({(str(r.get('set')),str(r.get('sid'))) for r in ar if r.get('sid') is not None})); print('AUDIT_SETS',collections.Counter(str(r.get('set')) for r in ar))
bykey={}
for r in ar:
    if r.get('sid') is not None: bykey[(str(r.get('set')),str(r.get('sid')))]=r
print('AUDIT_FINAL_BY_SET',collections.Counter(k[0] for k in bykey))
print('AUDIT_REPAIR_NEEDED',collections.Counter(str((r.get('audit') or {}).get('repair_needed')) for r in bykey.values()))
print('AUDIT_LIKELY_BAD',collections.Counter(str((r.get('audit') or {}).get('likely_bad_pair')) for r in bykey.values()))
for k in ['status','result','outcome','event','kind']:
    c=collections.Counter(str(r.get(k)) for r in ar if k in r)
    if c: print('AUDIT_FIELD',k,c)
err=[]
for r in ar:
    txt=json.dumps(r,ensure_ascii=False).lower()
    if any(x in txt for x in ['error','exception','timeout','parse']): err.append(r)
print('AUDIT_ERRORLIKE',len(err))
for fname in ['events.jsonl','corrected.jsonl']:
    rows=jsonl(run/fname); print(fname,'lines',len(rows),'mtime',datetime.datetime.fromtimestamp((run/fname).stat().st_mtime).astimezone().isoformat()); print(fname,'status',collections.Counter(r.get('status','<missing>') for r in rows))
    if fname=='events.jsonl':
        by=collections.defaultdict(list)
        for r in rows: by[r.get('content_id')].append(r)
        print('EVENTS_UNIQUE',len(by),'ATTEMPTED_RETRIES',sum(len(v)>1 for v in by.values()),'MAX_ATTEMPT',max([r.get('attempt',0) for r in rows] or [0])); print('EVENTS_FINAL',collections.Counter(v[-1].get('status') for v in by.values()))
    else: print('CORRECTED_UNIQUE',len({r.get('content_id') for r in rows if r.get('content_id') is not None}))
src=Path(r'C:/Users/30796/AppData/Local/Temp/grpo/pairs_dump.json')
if src.exists():
    data=json.loads(src.read_text(encoding='utf-8')); print('SOURCE_PAIRS',len(data)); print('SOURCE_KEYS',sorted(data[0].keys()) if data else [])
else: print('SOURCE_MISSING',src)
log=run/'runner.log'
if log.exists():
    lines=log.read_text(encoding='utf-8',errors='replace').splitlines(); print('LOG_LINES',len(lines),'mtime',datetime.datetime.fromtimestamp(log.stat().st_mtime).astimezone().isoformat()); print('LOG_TAIL'); [print(x[:500]) for x in lines[-15:]]
