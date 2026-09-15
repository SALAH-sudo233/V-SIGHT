import argparse,json,os,glob,torch
from collections import defaultdict
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
B={'MATCH','MISMATCH','NA'};D={'KEEP','REJECT'}
PROMPT=('Look at the proposed region [{x0},{y0},{x1},{y1}]. Does the phrase "{q}" correctly and accurately describe the object in that region? Answer with ONLY this JSON object and no other text: {{"binding":"MATCH" or "MISMATCH" or "NA","decision":"KEEP" or "REJECT"}}. Use MATCH when the phrase is correctly bound to the region, MISMATCH when an attribute or relation is wrong, and NA when the requested object/co-occurrence is absent. KEEP only if fully accurate.')
def parse(s):
 try:o=json.loads(s.strip())
 except:return None,None
 if not isinstance(o,dict) or set(o)!={'binding','decision'}:return None,None
 b,d=o['binding'],o['decision']
 if not isinstance(b,str) or not isinstance(d,str):return None,None
 b,d=b.upper(),d.upper();return (b,d) if b in B and d in D else (None,None)
def iou(a,b):
 if not a or not b:return 0
 x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]); inter=max(0,x1-x0)*max(0,y1-y0)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter) if aa+bb-inter else 0
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ccv',required=True);ap.add_argument('--queue',required=True);ap.add_argument('--dev',required=True);ap.add_argument('--model',required=True);ap.add_argument('--lora',default='');ap.add_argument('--out',required=True);ap.add_argument('--shard',type=int);ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
 qrows={r['record_id']:r for r in (json.loads(x) for x in open(args.queue))}; dev=json.load(open(args.dev)); gt={os.path.basename(r['image_filename']):r.get('gt_bbox_xyxy') or r.get('positive_bbox') for r in dev}
 rows=[]
 for fn in sorted(glob.glob(args.ccv+'/shard_*.jsonl')):
  for line in open(fn):
   r=json.loads(line); ev=r['detector_evidence']; props=ev.get('proposals',[]); targets=[p for p in props if not p.get('is_reference')];
   if not targets:targets=props
   if not targets: rows.append((r,None));continue
   p=max(targets,key=lambda x:float(x.get('score') or 0));rows.append((r,p))
 if args.shard is not None: rows=[x for i,x in enumerate(rows) if i%args.shards==args.shard]
 proc=AutoProcessor.from_pretrained(args.model,trust_remote_code=True,min_pixels=256*28*28,max_pixels=768*28*28)
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda:0')
 if args.lora:model=PeftModel.from_pretrained(model,args.lora)
 model.eval();out=[]
 for j,(r,p) in enumerate(rows):
  src=qrows[r['record_id']]; q=r['query'];img=os.path.join(os.path.dirname(args.queue),os.path.basename(r['image_filename']))
  if not os.path.exists(img):img=os.path.join(os.environ.get('IMG_DIR',''),os.path.basename(r['image_filename']))
  pred=None
  if p is not None:
   box=p['bbox_xyxy'];msg=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':PROMPT.format(x0=int(box[0]),y0=int(box[1]),x1=int(box[2]),y1=int(box[3]),q=q)}]}]
   inp=proc.apply_chat_template([msg],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to('cuda:0')
   with torch.no_grad():o=model.generate(**inp,max_new_tokens=32,do_sample=False)
   txt=proc.decode(o[0][inp['input_ids'].shape[1]:],skip_special_tokens=True);b,d=parse(txt);pred={'binding':b,'decision':d}
  gtbox=gt.get(os.path.basename(r['image_filename']));iv=iou(p.get('bbox_xyxy') if p else None,gtbox);gold=src['gold'];goldb='MATCH' if gold=='KEEP' else ('MISMATCH' if src.get('htype') in {'attribute','relation'} else 'NA')
  out.append({'record_id':r['record_id'],'gold':gold,'gold_binding':goldb,'htype':src.get('htype'),'proposal':p,'proposal_iou':iv,'proposal_recall':iv>=0.5,'pred':pred})
  if (j+1)%100==0:print(j+1,flush=True)
 os.makedirs(os.path.dirname(args.out),exist_ok=True);json.dump(out,open(args.out,'w'),ensure_ascii=False)
 print('DONE',args.out,len(out))
if __name__=='__main__':main()
