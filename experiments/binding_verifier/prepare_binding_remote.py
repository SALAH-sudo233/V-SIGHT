import json,subprocess,shutil,hashlib,sys
from pathlib import Path
w=Path.home()/'SVD/grpo_verifier';sys.path.insert(0,str(w))
from binding_data import prepare,convert,image_key
src=w/'grpo_1996_swift_v2.jsonl';rows=[json.loads(l) for l in src.open()]
train,sanity=prepare(rows)
out=w/'binding_pilot_v1';out.mkdir(exist_ok=False)
held={image_key(r) for r in sanity}
grpo=[convert(r) for r in rows if image_key(r) not in held]
for name,data in [('sft',train),('sanity',sanity),('grpo',grpo)]:
 assert all((w/r['images'][0]).exists() for r in data)
 with (out/(name+'.jsonl')).open('x') as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
manifest={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'n_sft':len(train),'n_sanity':len(sanity),'n_grpo':len(grpo),'sft_images':len({image_key(r) for r in train}),'sanity_images':len(held),'image_overlap':len(held&{image_key(r) for r in train}),'schema_gate':0.95,'seed':42,'label_status':'Construction-label ablation only; not independently verified binding or process labels. No target supervision.','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
