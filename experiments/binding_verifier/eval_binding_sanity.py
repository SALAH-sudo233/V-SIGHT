import argparse,json,re,torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
V={'MATCH','MISMATCH','NA'};D={'KEEP','REJECT'}
def parse(s):
 raw=s.strip()
 if not raw.startswith('{') or not raw.endswith('}'):return None,None
 m=re.fullmatch(r'\{\s*"binding"\s*:\s*"([A-Z]+)"\s*,\s*"decision"\s*:\s*"([A-Z]+)"\s*\}',raw)
 if not m:return None,None
 b,d=m.groups();return (b,d) if b in V and d in D else (None,None)
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--model',required=True);a.add_argument('--lora',required=True);a.add_argument('--out',required=True);x=a.parse_args();rows=[json.loads(l) for l in open(x.data)]
 p=AutoProcessor.from_pretrained(x.model,trust_remote_code=True,min_pixels=256*28*28,max_pixels=768*28*28);m=Qwen2_5_VLForConditionalGeneration.from_pretrained(x.model,dtype=torch.bfloat16,device_map='cuda:0');from peft import PeftModel;m=PeftModel.from_pretrained(m,x.lora);m.eval();res=[]
 from PIL import Image
 for i,r in enumerate(rows):
  msg=[{'role':'user','content':[{'type':'image','image':r['images'][0]},{'type':'text','text':r['messages'][0]['content']}]}];inp=p.apply_chat_template([msg],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to('cuda:0')
  with torch.no_grad():o=m.generate(**inp,max_new_tokens=40,do_sample=False)
  s=p.decode(o[0][inp['input_ids'].shape[1]:],skip_special_tokens=True);b,d=parse(s);res.append({'htype':r['htype'],'solution':r['solution'],'gold_binding':r['binding_gold'],'pred_binding':b,'pred_decision':d,'valid':b is not None,'correct_binding':b==r['binding_gold'],'correct_decision':d==r['solution'],'text':s});print(i+1,flush=True)
 json.dump(res,open(x.out,'w'),ensure_ascii=False,indent=2)
if __name__=='__main__':main()
