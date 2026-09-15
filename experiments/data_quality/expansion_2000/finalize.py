import pathlib,json,re,hashlib,collections,html,datetime
from PIL import Image
O=pathlib.Path(__file__).parent; R=O.parent.parent
IDS=[472639,478262,224420,119815]; TYPES=['object','co_occurrence','attribute','relation']; SUFFIX=dict(zip(TYPES,['obj','cooc','attr','rel']))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,d):(O/name).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
ex=json.load(open(O/'exclusion_sets.json')); pilot_sources={}; pilot_ids=set(ex['pilot'])
for p in (R/'golden_review').rglob('*'):
 if p.suffix not in ['.jsonl','.txt']:continue
 text=p.read_text(encoding='utf8'); found={int(i) for i in re.findall(r'COCO_(?:train|val)2014_(\d{12})',text)}
 if p.name=='pilot_image_ids.txt':found|={int(i) for i in re.findall(r'\b\d+\b',text)}
 if p.suffix=='.jsonl':
  def walk(v):
   if isinstance(v,dict):
    for k,w in v.items():
     if k in ['image_id','coco_image_id','coco_id'] and str(w).isdigit():found.add(int(w))
     walk(w)
   elif isinstance(v,list):
    for w in v:walk(w)
  for line in text.splitlines():
   if line.strip():walk(json.loads(line))
 pilot_sources[str(p)]={'sha256':sha(p),'image_ids':sorted(found),'count':len(found)};pilot_ids|=found
ex['pilot_union']=sorted(pilot_ids)
strict_path=O/'original_2000_strict_exclusion_evidence.json'
strict_rows=json.load(open(strict_path))
strict_ids={int(re.search(r'(\d{12})',r['image_filename'])[1]) for r in strict_rows}
assert len(strict_ids)==2000
ex['original2000_including_excluded4']=sorted(strict_ids)
ex['previously_excluded4']=sorted(strict_ids-set(ex['1996']))
assert len(ex['previously_excluded4'])==4
for role,ids in ex.items():assert not set(IDS)&set(ids),(role,set(IDS)&set(ids))
orig=json.load(open(O/'original_train1996.json'));assert len(orig)==7984
orig_ids={int(re.search(r'(\d{12})',r['image_filename'])[1]) for r in orig};assert len(orig_ids)==1996 and not set(IDS)&orig_ids
adds=[]; comparisons=[]; n_pass=0
for iid in IDS:
 c=json.load(open(O/('candidate_%d.json'%iid))); a=c['audit']; assert a['overall_pass'] is True and len(a['pairs'])==4
 assert {p['type'] for p in a['pairs']}==set(TYPES)
 for p in a['pairs']:
  pos=c['audit_order_positive'][p['type']]; neg='A' if pos=='B' else 'B'
  for k in ['pass','positive_unique','bbox_valid','grammar_ok','single_factor_type_correct','reference_clear']:assert p[k] is True,(iid,p)
  assert p[pos+'_matches_box']=='yes' and p[neg+'_matches_box']=='no' and p[pos+'_has_match_anywhere']=='yes' and p[neg+'_has_match_anywhere']=='no'
  n_pass+=1
 v=c['proposed_after']; src=c['source']; b=src['annotation']['bbox']; bbox=[b[0],b[1],b[0]+b[2],b[1]+b[3]]; fn='COCO_train2014_%012d.jpg'%iid; im=Image.open(O/'images'/fn); w,h=im.size;assert 0<=bbox[0]<bbox[2]<=w and 0<=bbox[1]<bbox[3]<=h
 base='expansion_ref%d_COCO_train2014_%012d'%(src['ref']['ref_id'],iid)
 for ht in TYPES:
  row={'source':'RefCOCOg','image_filename':fn,'chosen':v['positive'],'rejected':v['negative'][ht],'sample_id':base+'__'+SUFFIX[ht],'hallucination_type':ht,'positive_text':v['positive'],'negative_text':v['negative'][ht],'positive_bbox':bbox,'positive_bbox_format':'xyxy','gt_bbox_xyxy':bbox,'chosen_bbox_xyxy':bbox,'base_sample_id':base,'pair_id':base+'::'+ht,'difficulty_tier':ht,'expansion_origin':'RefCOCOg refs(google).p train','expansion_method':'image_conditioned_generation_independent_blinded_image_review','pair_metadata':{'schema_version':'roh_vcd_strict_pair_v2','base_sample_id':base,'positive_is_shared_with_group':True,'positive_bbox_is_shared_with_group':True,'pair_alignment_validated':True,'semantic_status':'vlm_image_review_passed_not_human_gold','negative_origin':'generated_strict_counterfactual','source_ref_id':src['ref']['ref_id'],'source_ann_id':src['ref']['ann_id'],'audit_file':'verification_%d.json'%iid}}
  adds.append(row)
 comparisons.append({'image_id':iid,'image':fn,'image_sha256':sha(O/'images'/fn),'source_sentences':[s['sent'] for s in src['ref']['sentences']],'source_bbox_xywh':b,'final_bbox_xyxy':bbox,'generation_before':c['generation_before'],'final_after':v,'audit':'verification_%d.json'%iid})
# Content-duplicate check against every locally available original train/dev image.
selected_hash={sha(O/'images'/('COCO_train2014_%012d.jpg'%iid)):iid for iid in IDS};assert len(selected_hash)==4
checked=0; collisions=[]
for p in pathlib.Path('C:/Users/30796/AppData/Local/Temp/grpo/all_auditimgs').glob('*.jpg'):
 checked+=1; h=sha(p)
 if h in selected_hash:collisions.append({'selected':selected_hash[h],'existing':str(p)})
assert not collisions
allrows=orig+adds; cnt=collections.Counter(r['hallucination_type'] for r in allrows);assert len(allrows)==8000 and all(cnt[t]==2000 for t in TYPES)
assert len({r['image_filename'] for r in allrows})==2000 and len({r['sample_id'] for r in allrows})==8000 and allrows[:7984]==orig
assert len(adds)==16 and n_pass==16
manifest={'schema':'vsight_training_expansion_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'new_four_images_vlm_image_audit_passed','not_human_gold':True,'scope':'2000 is TRAINING, not independent heldout. Existing 1996 unchanged; not a claim that ongoing full audit has passed.','original_images':1996,'original_pairs':7984,'added_images':4,'added_pairs':16,'merged_images':2000,'merged_pairs':8000,'type_counts':dict(cnt),'selected_coco_ids':IDS,'overlap_by_role':{k:sorted(set(IDS)&set(v)) for k,v in ex.items()},'exclusion_counts':{k:len(set(v)) for k,v in ex.items()},'pilot_evidence':pilot_sources,'existing_local_image_hash_checked':checked,'image_hash_collisions':collisions,'hash_check_limit':'Local train/dev cache only; canonical COCO image-id isolation also checked for pilot. Does not certify arbitrary perceptual near-duplicate absence.','source_pool':json.load(open(O/'source_candidates.json'))['sources'],'original_train_file_sha256':sha(O/'original_train1996.json'),'blinded_image_review_pass_pairs':n_pass,'reviewer_model':'gpt-6-astra','independence_limit':'Separate context from generation; same model family, not independent human or cross-model consensus.'}
dump('new4_pairs16.json',adds);dump('refcocog_train2000.expanded_v1.json',allrows);dump('before_after.json',comparisons);dump('isolation_evidence.json',{'exclusions':ex,'pilot_sources':pilot_sources,'overlaps':manifest['overlap_by_role']});manifest['merged_sha256']=sha(O/'refcocog_train2000.expanded_v1.json');manifest['additions_sha256']=sha(O/'new4_pairs16.json');dump('manifest.json',manifest)
parts=['<!doctype html><meta charset="utf-8"><title>V-SIGHT training expansion: 4 new images</title><style>body{font:16px sans-serif;max-width:1400px;margin:auto}img{max-width:48%}pre{white-space:pre-wrap}section{border-top:2px solid #888;padding:20px}</style><h1>1996 → 2000 training expansion / 16 added pairs</h1><p>New additions passed separate-context actual-image VLM audit. NOT human gold. Original 1996 unchanged and not certified by this expansion.</p>']
for c in comparisons:
 parts.append('<section><h2>COCO %d</h2><img src="images/%s"><img src="images/boxed_%s"><h3>Original pool expressions → final positive/negative</h3><pre>%s</pre></section>'%(c['image_id'],c['image'],c['image'],html.escape(json.dumps(c,ensure_ascii=False,indent=2))))
(O/'before_after.html').write_text('\n'.join(parts),encoding='utf8')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['pilot_evidence','source_pool']},ensure_ascii=False,indent=2))
