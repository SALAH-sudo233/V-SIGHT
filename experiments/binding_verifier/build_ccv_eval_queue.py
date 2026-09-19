import json,os,sys
src=sys.argv[1];out=sys.argv[2]
rows=json.load(open(src));n=0
with open(out,'w') as f:
 for r in rows:
  img=os.path.basename(r['image_filename']);ht=r.get('hallucination_type','?')
  if r.get('positive_text'):
   f.write(json.dumps({'record_id':f"{r.get('image_id',r.get('image_filename'))}__pos",'query':r['positive_text'],'image_filename':img,'gold':'KEEP','htype':ht,'source':'500dev'},ensure_ascii=False)+'\n');n+=1
  if r.get('negative_text') and r['negative_text']!=r.get('positive_text'):
   f.write(json.dumps({'record_id':f"{r.get('image_id',r.get('image_filename'))}__neg__{ht}",'query':r['negative_text'],'image_filename':img,'gold':'REJECT','htype':ht,'source':'500dev'},ensure_ascii=False)+'\n');n+=1
print(n)
