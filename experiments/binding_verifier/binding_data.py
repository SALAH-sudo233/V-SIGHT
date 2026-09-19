"""Construction-label pilot, not independent process supervision."""
import copy,hashlib,json,os,random
from collections import defaultdict
TYPES=('object','co_occurrence','attribute','relation')
SUFFIX=('Answer ONLY one JSON object with exactly two fields: '
 '"binding": "MATCH" or "MISMATCH" or "NA", and "decision": "KEEP" or "REJECT". '
 'MATCH means the phrase accurately describes the specified region. '
 'MISMATCH means an attribute or relation is incorrect. '
 'NA means an object identity or co-occurrence claim is incorrect. '
 'KEEP only if the whole phrase is accurate, otherwise REJECT. '
 'Example output syntax: {"binding":"MATCH","decision":"KEEP"}. No prose or markdown.')
def image_key(row):return os.path.basename(row['images'][0])
def convert(row,sft=False):
 x=copy.deepcopy(row)
 assert x['solution'] in ('KEEP','REJECT') and x['htype'] in TYPES
 messages=x['messages'];assert messages[0]['role']=='user'
 text=messages[0]['content'];marker='Answer with ONLY a JSON object:'
 assert text.count(marker)==1,'Unexpected source prompt; do not silently replace'
 x['messages']=[{'role':'user','content':text.split(marker)[0]+SUFFIX}]
 binding='MATCH' if x['solution']=='KEEP' else ('MISMATCH' if x['htype'] in ('attribute','relation') else 'NA')
 x['binding_gold']=binding
 x['prompt_id']=hashlib.sha256(json.dumps([image_key(x),x['messages'][0]['content'],x['solution'],x['htype']],ensure_ascii=False).encode()).hexdigest()
 if sft:x['messages'].append({'role':'assistant','content':json.dumps({'binding':binding,'decision':x['solution']},separators=(',',':'))})
 return x

def prepare(rows,train_per_stratum=250,sanity_images=32,seed=42):
 rng=random.Random(seed);images=sorted({image_key(r) for r in rows});rng.shuffle(images)
 assert len(images)>sanity_images
 held=set(images[:sanity_images]);pools=defaultdict(list)
 for row in rows:
  if image_key(row) not in held:pools[(row['htype'],row['solution'])].append(row)
 train=[]
 for ht in TYPES:
  for sol in ('KEEP','REJECT'):
   pool=pools[(ht,sol)];rng.shuffle(pool)
   assert len(pool)>=train_per_stratum
   train.extend(convert(r,True) for r in pool[:train_per_stratum])
 sanity=[convert(r,False) for r in rows if image_key(r) in held]
 rng.shuffle(train);sanity.sort(key=lambda r:r['prompt_id'])
 assert not {image_key(r) for r in train}&{image_key(r) for r in sanity}
 return train,sanity
