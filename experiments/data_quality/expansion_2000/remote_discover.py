import json,pickle,pathlib,hashlib
p=pathlib.Path('/home/u2025141034/models/LENS/data/refcoco'); ex=json.load(open('/tmp/vsight_expansion_exclusions.json')); ban=set(sum(ex.values(),[])); strict=json.load(open('/home/u2025141034/benchmark/repaired/refcocog_2000_heldout.strict.json'))
import re
for r in strict:
 for v in r.values():
  if isinstance(v,str):
   m=re.search(r'COCO_train2014_(\d{12})',v)
   if m: ban.add(int(m[1]))
refs=pickle.load(open(p/'refs(google).p','rb')); inst=json.load(open(p/'instances.json')); anns={a['id']:a for a in inst['annotations']}; ims={a['id']:a for a in inst['images']}; selected=[]; used=set()
for r in refs:
 if r['split']!='train' or r['image_id'] in ban or r['image_id'] in used:continue
 a=anns[r['ann_id']]; im=ims[r['image_id']]
 if a['bbox'][2]*a['bbox'][3]<im['width']*im['height']*.08:continue
 used.add(r['image_id']); selected.append({'ref':r,'annotation':a,'image':im})
 if len(selected)==12:break
out={'sources':{str(p/f):hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['refs(google).p','instances.json']},'exclusion_count':len(ban),'candidates':selected}
pathlib.Path('/tmp/vsight_expansion_candidates.json').write_text(json.dumps(out));print([(x['ref']['image_id'],[s['sent'] for s in x['ref']['sentences']]) for x in selected])
