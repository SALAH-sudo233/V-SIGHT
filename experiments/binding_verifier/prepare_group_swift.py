import json,sys,os
src,dst,imgdir=sys.argv[1:4]
out=[]
for r in json.load(open(src,encoding='utf-8')):
 for q,sol in ((r['pos'],'KEEP'),(r['neg'],'REJECT')):
  x0,y0,x1,y1=[int(v) for v in r['bbox']]
  out.append({'messages':[{'role':'user','content':f'<image>Look at the region [{x0},{y0},{x1},{y1}]. Does the phrase "{q}" correctly and accurately describe the object in that region? Answer with ONLY a JSON object: {{"binding":"MATCH" or "MISMATCH" or "NA", "decision":"KEEP" or "REJECT"}}.'}], 'images':[os.path.join(imgdir,r['img'])], 'solution':sol,'htype':r['ht'],'hallucination_group':r['hallucination_group'],'sid':r['sid']})
with open(dst,'w',encoding='utf-8') as f:
 for x in out:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print('rows',len(out),'missing_images',sum(not os.path.exists(x['images'][0]) for x in out))
