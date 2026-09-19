import json,os,sys
root=sys.argv[1];out=sys.argv[2];names=['VisionReasoner','Vision-R1','TreeVGR','UniVG-R1']; data={}
for name in names:
 p=f'{root}/{name}/records.jsonl'; d={}
 for line in open(p):
  x=json.loads(line)
  if x.get('task')!='t2_vqa_grounding':continue
  k=(x.get('base_sample_id'),x.get('query_role'),x.get('hallucination_type'),x.get('query'))
  d[k]=x
 data[name]=d
base=set(data[names[0]])
print('keys',*[len(data[n]) for n in names],'intersection',len(set.intersection(*(set(data[n]) for n in names))))
os.makedirs(out,exist_ok=True)
for name in names:
 rows=[]; valid=0
 for k,x in data[name].items():
  box=x.get('pred_bbox_xyxy');
  if not (isinstance(box,list) and len(box)==4 and all(isinstance(v,(int,float)) for v in box)):continue
  valid+=1;img=x.get('image_filename') or ('COCO_train2014_'+x['base_sample_id'].split('COCO_train2014_')[-1]+'.jpg')
  rows.append({'record_id':f'{name}__{x["base_sample_id"]}__{x["query_role"]}__{x["hallucination_type"]}','query':x['query'],'image_filename':img,'upstream_box_xyxy':box,'upstream_model':name,'gold':'KEEP' if x.get('query_role')=='positive' else 'REJECT','htype':x.get('hallucination_type')})
 with open(f'{out}/{name}.jsonl','w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(name,'valid',valid,'written',len(rows))
