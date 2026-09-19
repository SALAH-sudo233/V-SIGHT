import json,sys,os,random
src,dst,imgdir,mode=sys.argv[1:5]
rows=json.load(open(src,encoding='utf-8'))
if mode=='stage2':
    boh=[r for r in rows if r['hallucination_group']=='BOH']
    roh=[r for r in rows if r['hallucination_group']=='ROH']
    random.Random(44).shuffle(boh); rows=roh+boh[:max(1,len(roh)//5)]
else:
    random.Random(44).shuffle(rows)
out=[]
for r in rows:
 for q,sol in ((r['pos'],'KEEP'),(r['neg'],'REJECT')):
  x0,y0,x1,y1=[int(v) for v in r['bbox']]
  out.append({'messages':[{'role':'user','content':f'<image>Look at the region [{x0},{y0},{x1},{y1}]. Does the phrase "{q}" correctly and accurately describe the object in that region? Answer with ONLY a JSON object: {{"binding":"MATCH" or "MISMATCH" or "NA", "decision":"KEEP" or "REJECT"}}.'}], 'images':[os.path.join(imgdir,r['img'])], 'solution':sol,'htype':r['ht'],'hallucination_group':r['hallucination_group'],'stage':mode,'sid':r['sid']})
with open(dst,'w',encoding='utf-8') as f:
 for x in out:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print(mode,'source_pairs',len(rows),'prompts',len(out),'missing_images',sum(not os.path.exists(x['images'][0]) for x in out))
