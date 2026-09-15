import json,pathlib,re,subprocess
R=pathlib.Path('C:/Users/30796/Desktop/V-SIGHT-assets'); O=R/'data_quality_audit/expansion_2000'; T=pathlib.Path('C:/Users/30796/AppData/Local/Temp'); PY='C:/Users/30796/AppData/Local/hermes/hermes-agent/venv/Scripts/python'; SSH=str(T/'ssh_tool.py')
def ssh(*a):
 p=subprocess.run([PY,SSH,'vlm1',*a],capture_output=True,text=True); print(p.stdout); assert p.returncode==0,p.stderr
rows=json.load(open(T/'grpo/pairs_dump.json')); splits=json.load(open(R/'experiments_v2/e0_audit/manifests/binding_v2_splits.json'))
sets={s:sorted({int(re.search(r'(\d{12})',x['img'])[1]) for x in rows if x['set']==s}) for s in {x['set'] for x in rows}}; sets['pilot']=splits['roles']['pilot_human_dev']['coco_ids']; (O/'exclusion_sets.json').write_text(json.dumps(sets))
ssh('put',str(O/'exclusion_sets.json'),'/tmp/vsight_expansion_exclusions.json')
remote='''import json,pickle,pathlib,hashlib
p=pathlib.Path('/home/u2025141034/models/LENS/data/refcoco'); ex=json.load(open('/tmp/vsight_expansion_exclusions.json')); ban=set(sum(ex.values(),[])); strict=json.load(open('/home/u2025141034/benchmark/repaired/refcocog_2000_heldout.strict.json'))
import re
for r in strict:
 for v in r.values():
  if isinstance(v,str):
   m=re.search(r'COCO_train2014_(\\d{12})',v)
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
'''
(O/'remote_discover.py').write_text(remote); ssh('put',str(O/'remote_discover.py'),'/tmp/vsight_expansion_discover.py'); ssh('run','python /tmp/vsight_expansion_discover.py'); ssh('get','/tmp/vsight_expansion_candidates.json',str(O/'source_candidates.json'))
d=json.load(open(O/'source_candidates.json')); (O/'images').mkdir(exist_ok=True)
for x in d['candidates']:
 fn='COCO_train2014_%012d.jpg'%x['ref']['image_id'];ssh('get','/home/u2025141034/models/LENS/data/refcoco/train2014/'+fn,str(O/'images'/fn))
ssh('get','/home/u2025141034/benchmark/repaired/refcocog_1996_heldout.manual_v2.json',str(O/'original_train1996.json'))
