import os,json,base64,io,re,hashlib,random,time,concurrent.futures as cf
from PIL import Image,ImageDraw
import urllib.request
from dotenv import dotenv_values
import yaml
ROOT=r'C:\Users\30796\AppData\Local\Temp\grpo'; OUTDIR=r'C:\Users\30796\Desktop\V-SIGHT-assets\data_quality_audit'
os.makedirs(OUTDIR,exist_ok=True)
rows=json.load(open(os.path.join(ROOT,'pairs_dump.json'),encoding='utf8'))
imgdir=os.path.join(ROOT,'all_auditimgs')
cfg=yaml.safe_load(open(os.path.expanduser(r'~\AppData\Local\hermes\config.yaml'),encoding='utf8'))['model']; key=dotenv_values(os.path.expanduser(r'~\AppData\Local\hermes\.env'))[cfg['key_env']]
MODEL='gpt-6-astra'; URL=cfg['base_url'].rstrip('/')+'/chat/completions'
out=os.path.join(OUTDIR,'audit_v2.jsonl'); lock=None
PROMPT='''You are an independent quality auditor for a referring-expression grounding dataset. Do not assume either expression or the red box is correct. The red box is only a candidate target region. Image size is {w}x{h}. Two expressions A and B are shown in random order; one was originally intended positive and the other counterfactual, but you must ignore that.
A: {a}\nB: {b}
Return ONLY JSON with this schema:
{{"A_matches_box":"yes|no|uncertain","B_matches_box":"yes|no|uncertain","A_unique":"yes|no|uncertain","B_unique":"yes|no|uncertain","reference_clear":"yes|no|uncertain|none","text_conflict_A":"yes|no|uncertain","text_conflict_B":"yes|no|uncertain","likely_bad_pair":"yes|no|uncertain","problem_types":["text_conflict|box_mismatch|ambiguous_reference|non_unique|both_true|neither_true|other"],"repair_needed":"yes|no|uncertain","proposed_A":"short corrected expression or null","proposed_B":"short corrected expression or null","reason":"max 35 words"}}
Judge only visible evidence. A phrase can be false in the image without being self-contradictory. A standing person and a sitting person are not contradictory if they refer to different people; 'standing person sitting on the bench' is contradictory when it describes one person. Use uncertain when the image or reference is insufficient.'''
def imgdata(r):
 p=os.path.join(imgdir,os.path.basename(r['img'])); im=Image.open(p).convert('RGB'); w,h=im.size
 d=ImageDraw.Draw(im); b=r['bbox']; d.rectangle(b,outline='red',width=max(3,int(max(w,h)*.008)))
 z=io.BytesIO(); im.save(z,'JPEG',quality=86); return w,h,base64.b64encode(z.getvalue()).decode(),hashlib.sha256(z.getvalue()).hexdigest()
def one(r):
 try:
  w,h,b64,ih=imgdata(r); swap=hash(r['sid'])%2==0; a,b=(r['pos'],r['neg']) if not swap else (r['neg'],r['pos'])
  payload={'model':MODEL,'temperature':0,'max_tokens':300,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+b64}},{'type':'text','text':PROMPT.format(w=w,h=h,a=a,b=b)}]}]}
  req=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
  raw=json.load(urllib.request.urlopen(req,timeout=150)); txt=raw['choices'][0]['message']['content']; m=re.search(r'\{.*\}',txt,re.S); v=json.loads(m.group()) if m else {'parse_error':txt}
  if swap:
   for x,y in [('A_matches_box','B_matches_box'),('A_unique','B_unique'),('text_conflict_A','text_conflict_B'),('proposed_A','proposed_B')]:v[x],v[y]=v.get(y),v.get(x)
  return {'set':r['set'],'sid':r['sid'],'ht':r['ht'],'img':r['img'],'bbox':r['bbox'],'pos':r['pos'],'neg':r['neg'],'image_size':[w,h],'image_hash':ih,'model':MODEL,'audit':v,'raw':txt,'usage':raw.get('usage')}
 except Exception as e:return {'set':r.get('set'),'sid':r.get('sid'),'error':type(e).__name__+': '+str(e)[:180]}
def main():
 done=set()
 if os.path.exists(out):
  for l in open(out,encoding='utf8'):
   try:
    x=json.loads(l); done.add((x.get('set'),x.get('sid')))
   except:pass
 todo=[r for r in rows if (r['set'],r['sid']) not in done and os.path.exists(os.path.join(imgdir,os.path.basename(r['img'])))]
 print('total',len(rows),'done',len(done),'todo',len(todo),flush=True)
 with open(out,'a',encoding='utf8') as f,cf.ThreadPoolExecutor(max_workers=4) as ex:
  futures={ex.submit(one,r):(r.get('set'),r.get('sid')) for r in todo}
  for i,fu in enumerate(cf.as_completed(futures),1):
   key=futures[fu]
   try: x=fu.result()
   except BaseException as e: x={'set':key[0],'sid':key[1],'error':'executor:'+type(e).__name__+': '+str(e)[:180]}
   f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
   if i%5==0:print('progress',i,'/',len(todo),flush=True)
 print('DONE',out)
if __name__=='__main__':main()
