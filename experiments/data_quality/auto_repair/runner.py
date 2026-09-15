"""Append-only, image-grounded repair. Same model, independent contexts; not human gold."""
import argparse, base64, datetime, hashlib, html, io, json, logging, os, re, time, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw
import yaml
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent
AUDIT=ROOT.parent/'audit_v2.jsonl'
DATA=Path('C:/Users/30796/AppData/Local/Temp/grpo')
MODEL='gpt-6-astra'
CHECKS=['bbox_valid','target_preserved','grammar_ok','no_internal_conflict','natural_language','single_factor_type_correct','reference_clear','absence_visually_certain','pass']
RULES='''Preserve the original physical target and original bbox; never guess or move the box. If target identity/box/reference is uncertain return needs_human. Positive must uniquely identify the boxed target in the entire image. Negative must have NO matching target anywhere in the entire image, with visibly supported absence; uncertainty fails. Both expressions must be natural grammatical plausible descriptions, not absurd shortcuts or internally contradictory about the same object. Four types: object replaces ONLY target object noun (retain modifiers); attribute changes ONLY ONE attribute (retain object and remaining context); relation retains target AND an actually visible uniquely identifiable reference, changing ONLY relation; co_occurrence adds ONLY an absent plausible companion entity to the positive. A corrected positive may disambiguate original wording, but must preserve the original physical target. Avoid synonyms, unverifiable hidden objects, profession guesses and multiple-factor changes. Relation direction is target relative to reference. Images supplied: original full scene then red candidate box. Red box is fallible, not proof.'''

def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def key(r):return (str(r['set']),str(r['sid']),str(r['ht']))
def content_id(r):return digest({k:r[k] for k in ('set','sid','ht','img','bbox','pos','neg')})
def accept(v,p):
 n='B' if p=='A' else 'A'
 enums=[role+suffix for role in ('A','B') for suffix in ('_matches_box','_has_match_anywhere','_unique')]
 return all(v.get(k) in ('yes','no','uncertain') for k in enums) and all(v.get(k) is True for k in CHECKS) and v.get(p+'_matches_box')=='yes' and v.get(p+'_has_match_anywhere')=='yes' and v.get(p+'_unique')=='yes' and v.get(n+'_matches_box')=='no' and v.get(n+'_has_match_anywhere')=='no' and isinstance(v.get('reason'),str) and bool(v['reason'].strip())
def readlines(p):
 if not p.exists():return []
 out=[]
 for line in p.read_text(encoding='utf8').splitlines():
  try:out.append(json.loads(line))
  except json.JSONDecodeError:continue
 return out

def write(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8'); os.replace(tmp,p)
def append(p,x):
 with p.open('a',encoding='utf8') as f:f.write(canon(x)+'\n'); f.flush(); os.fsync(f.fileno())
def flagged(v):
 return v.get('repair_needed')!='no' or v.get('likely_bad_pair')!='no' or v.get('A_matches_box')!='yes' or v.get('B_matches_box')!='no' or v.get('A_unique')!='yes' or v.get('text_conflict_A')!='no' or v.get('text_conflict_B')!='no' or bool(v.get('problem_types')) or v.get('reference_clear') not in ('yes','none')
def queue():
 source={key(r):r for r in json.loads((DATA/'pairs_dump.json').read_text(encoding='utf8'))}
 quarantine=set()
 for p in (ROOT.parent/'cleaning').glob('*/quarantine.jsonl'):
  quarantine.update(key(r) for r in readlines(p))
 evidence={}
 for a in readlines(AUDIT):
  if not isinstance(a.get('audit'),dict) or 'error' in a or 'ht' not in a:continue
  k=key(a); r=source.get(k)
  if r is None or any(a.get(f)!=r[f] for f in ('img','bbox','pos','neg')):continue
  if flagged(a['audit']) or k in quarantine:evidence[k]=a
 return [(source[k],a,k in quarantine) for k,a in sorted(evidence.items(),key=lambda kv:(kv[0] not in quarantine,kv[0]))]

def images(r,folder):
 path=DATA/'all_auditimgs'/Path(r['img']).name
 if not path.exists():raise FileNotFoundError('missing_source_image')
 raw=path.read_bytes(); im=Image.open(io.BytesIO(raw)).convert('RGB'); w,h=im.size; b=r['bbox']
 if len(b)!=4 or not all(isinstance(x,(int,float)) for x in b) or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError('invalid_original_bbox')
 d=ImageDraw.Draw(im); d.rectangle(b,outline='red',width=4); buf=io.BytesIO(); im.save(buf,format='JPEG',quality=93); boxed=buf.getvalue()
 (folder/'original.jpg').write_bytes(raw); (folder/'boxed.jpg').write_bytes(boxed)
 return [raw,boxed],[w,h]

def call(prompt,imgs,path):
 # Secret is read only into memory; never printed, saved or sent except auth header.
 home=Path('C:/Users/30796/AppData/Local/hermes')
 cfg=yaml.safe_load((home/'config.yaml').read_text(encoding='utf8'))['model']; secret=dotenv_values(home/'.env')[cfg['key_env']]
 messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(b).decode()}} for b in imgs]+[{'type':'text','text':prompt}]}]
 payload={'model':MODEL,'temperature':0,'max_tokens':1500,'messages':messages}
 provenance={'model_requested':MODEL,'input_hash':digest(payload),'prompt':prompt,'image_hashes':[hashlib.sha256(b).hexdigest() for b in imgs],'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 write(path.with_name(path.stem+'_input.json'),provenance)
 req=urllib.request.Request(cfg['base_url'].rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=210) as response:raw=json.load(response)
 except Exception as exc:
  write(path,{**provenance,'error_type':type(exc).__name__}); raise
 # Persist complete raw response before parsing, including usage and returned model.
 write(path,{**provenance,'raw_response':raw,'usage':raw.get('usage'),'model':raw.get('model')})
 txt=raw['choices'][0]['message']['content'].strip()
 if txt.startswith('```'):txt=re.sub(r'^```(?:json)?\s*|\s*```$','',txt)
 v=json.loads(txt)
 if not isinstance(v,dict):raise ValueError('schema_not_object')
 return v

def page(folder,r,g,v,status):
 def esc(x):return html.escape(str(x))
 body='<!doctype html><meta charset="utf-8"><title>V-SIGHT repair evidence</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}img{width:48%}td,th{padding:12px;border:1px solid #aaa}pre{white-space:pre-wrap}</style>'
 body+='<h1>'+esc(status)+'</h1><p>Same model, independent image-conditioned blind-order validation. Not human gold.</p><p>'+esc(key(r))+'</p><img src="original.jpg"><img src="boxed.jpg"><table><tr><th></th><th>Before</th><th>After</th></tr>'
 for label in ('pos','neg'):body+='<tr><th>'+label+'</th><td>'+esc(r[label])+'</td><td>'+esc(g.get(label,''))+'</td></tr>'
 body+='</table><h2>Generation</h2><pre>'+esc(json.dumps(g,ensure_ascii=False,indent=2))+'</pre><h2>Independent validation</h2><pre>'+esc(json.dumps(v,ensure_ascii=False,indent=2))+'</pre>'
 (folder/'before_after.html').write_text(body,encoding='utf8')

def repair(r,a,priority,run,attempt):
 cid=content_id(r); folder=run/'evidence'/cid/str(attempt); folder.mkdir(parents=True,exist_ok=True)
 result={'content_id':cid,'key':list(key(r)),'attempt':attempt,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence_dir':str(folder),'priority_quarantine':priority}
 write(folder/'source.json',{'source':r,'audit_evidence':a,'note':'Normalized structured A=pos B=neg. Raw letters ignored.'})
 try:
  imgs,size=images(r,folder)
  # No audit raw text or A/B proposals: their role mapping is unreliable.
  prompt=RULES+'\nRepair this reviewed pair. Use only image and original expressions, not audit proposals. Return ONLY JSON {"status":"proposed|needs_human","pos":"...","neg":"...","reason":"visible evidence and exact single-factor edit"}. Original: '+canon(r)+' Image size: '+canon(size)
  g=call(prompt,imgs,folder/'generation.json')
  if g.get('status')=='needs_human':result.update(status='needs_human',reason=g.get('reason')); page(folder,r,g,{},result['status']); return result
  if g.get('status')!='proposed' or any(not isinstance(g.get(k),str) or not g[k].strip() for k in ('pos','neg','reason')) or g['pos']==g['neg']:raise ValueError('invalid_generation_schema')
  if g['pos']==r['pos'] and g['neg']==r['neg']:
   result.update(status='needs_human',reason='generator_returned_unchanged_pair');page(folder,r,g,{},result['status']);return result
  p='B' if int(digest([cid,g['pos'],g['neg'],'blind-v1']),16)%2 else 'A'; n='B' if p=='A' else 'A'
  expressions={p:g['pos'],n:g['neg']}
  # New API request: no generation rationale, original labels or audit outcome.
  vp=RULES+'\nIndependently audit these UNLABELED expressions in randomized order. Do not presume either true. Target identity anchor (not a truth label): '+r['pos']+'\nType: '+r['ht']+' Bbox: '+canon(r['bbox'])+'\n'+canon(dict(sorted(expressions.items())))+'\nReturn ONLY JSON with A_matches_box,B_matches_box,A_has_match_anywhere,B_has_match_anywhere,A_unique,B_unique each "yes|no|uncertain"; '+','.join(CHECKS)+' each boolean; reason string with specific visible evidence. target_preserved means same original physical target, reference_clear is true for nonrelation or a visible unique relation reference. single_factor_type_correct requires the exact type definition. absence_visually_certain requires false expression to be demonstrably false across full image, not merely outside red box. pass only if all checks hold. Do not propose repairs.'
  v=call(vp,imgs,folder/'validation.json'); write(folder/'mapping.json',{'positive_label':p,'expressions':expressions,'method':'sha256 deterministic randomized order'})
  status='accepted' if accept(v,p) else 'validation_failed'
  page(folder,r,g,v,status); result.update(status=status,reason=v.get('reason'),positive_label=p)
  if status=='accepted':
   corrected={**r,'pos':g['pos'],'neg':g['neg'],'content_id':cid,'original_pos':r['pos'],'original_neg':r['neg'],'repair_version':run.name,'evidence_dir':str(folder),'validation':v,'validation_positive_label':p,'verification_kind':'same_model_independent_context_not_human_gold'}
   write(folder/'accepted.json',corrected)
   # accepted evidence precedes publication; startup also recovers accepted.json after crash.
   if cid not in {x['content_id'] for x in readlines(run/'corrected.jsonl')}:append(run/'corrected.jsonl',corrected)
  return result
 except (FileNotFoundError,ValueError) as e:
  result.update(status='needs_human' if str(e) in ('missing_source_image','invalid_original_bbox') else 'error',error_type=type(e).__name__,reason=str(e) if isinstance(e,FileNotFoundError) else 'validation_or_schema_error');return result
 except Exception as e:result.update(status='error',error_type=type(e).__name__);return result

class Lock:
 def __enter__(self):
  import msvcrt
  self.f=(ROOT/'runner.lock').open('a+b');self.f.seek(0)
  if self.f.read(1)==b'':self.f.write(b'0');self.f.flush()
  self.f.seek(0)
  try:msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
  except OSError:self.f.close();raise SystemExit('another auto_repair runner owns lock')
  return self
 def __exit__(self,*args):self.f.close()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--until-accepted',type=int,default=0);ap.add_argument('--max-items',type=int,default=0);ap.add_argument('--poll',type=int,default=45);args=ap.parse_args()
 with Lock():
  pointer=ROOT/'current_run.json'
  if pointer.exists():run=Path(json.loads(pointer.read_text())['run_dir'])
  else:
   run=ROOT/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run.mkdir(parents=True);write(pointer,{'run_dir':str(run)})
  logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(run/'runner.log',encoding='utf8'),logging.StreamHandler()])
  write(ROOT/'process.json',{'pid':os.getpid(),'run_dir':str(run),'watch':args.watch,'started':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  logging.info('START pid=%s watch=%s',os.getpid(),args.watch)
  published={x['content_id'] for x in readlines(run/'corrected.jsonl')}
  for f in (run/'evidence').glob('*/*/accepted.json'):
   x=json.loads(f.read_text(encoding='utf8'))
   if x['content_id'] not in published and accept(x['validation'],x['validation_positive_label']):append(run/'corrected.jsonl',x);published.add(x['content_id'])
  count=0
  while True:
   events=readlines(run/'events.jsonl');hist={}
   for e in events:hist.setdefault(e['content_id'],[]).append(e)
   done={cid for cid,ee in hist.items() if any(e['status'] in ('accepted','needs_human','validation_failed') for e in ee)}|published
   todo=[(r,a,p) for r,a,p in queue() if content_id(r) not in done and len(hist.get(content_id(r),[]))<3]
   write(run/'heartbeat.json',{'pid':os.getpid(),'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pending':len(todo),'published':len(readlines(run/'corrected.jsonl')),'events':len(events)})
   if not todo:
    if not args.watch:break
    time.sleep(args.poll);continue
   r,a,p=todo[0];cid=content_id(r);attempt=len(hist.get(cid,[]))+1
   logging.info('PROCESS %s attempt=%s',key(r),attempt)
   res=repair(r,a,p,run,attempt);append(run/'events.jsonl',res);count+=1
   logging.info('RESULT %s %s',key(r),res['status'])
   if res['status']=='error':time.sleep(min(30,5*attempt))
   if args.until_accepted and len(readlines(run/'corrected.jsonl'))>=args.until_accepted:break
   if args.max_items and count>=args.max_items:break
  logging.info('STOP completed_this_process=%s',count)
if __name__=='__main__':main()
