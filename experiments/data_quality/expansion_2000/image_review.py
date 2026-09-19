import pathlib,json,base64,io,re,hashlib,urllib.request,concurrent.futures,time
from PIL import Image,ImageDraw
from dotenv import dotenv_values
import yaml
O=pathlib.Path(__file__).parent
cfg=yaml.safe_load(open('C:/Users/30796/AppData/Local/hermes/config.yaml',encoding='utf8'))['model']; key=dotenv_values('C:/Users/30796/AppData/Local/hermes/.env')[cfg['key_env']]
def call(prompt,imgs,out):
 content=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(b).decode()}} for b in imgs]+[{'type':'text','text':prompt}]
 payload={'model':'gpt-6-astra','temperature':0,'max_tokens':2400,'messages':[{'role':'user','content':content}]}
 for attempt in range(3):
  try:
   req=urllib.request.Request(cfg['base_url'].rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
   raw=json.load(urllib.request.urlopen(req,timeout=230)); text=raw['choices'][0]['message']['content']; v=json.loads(re.search(r'\{.*\}',text,re.S)[0]); break
  except Exception as e:
   if attempt==2:raise
   time.sleep(3)
 evidence={'model':'gpt-6-astra','prompt':prompt,'images_sha256':[hashlib.sha256(b).hexdigest() for b in imgs],'response':raw,'parsed':v}
 out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8');return v
GEN='''You construct high-quality image-grounded counterfactual pairs. First image is original, second same image with candidate target bbox in red. Verify target bbox using real image; do not trust supplied source expressions. Need ONE shared positive expression uniquely identifying boxed whole target, and four different negatives. Each negative must be grammatically natural, non-self-contradictory, clearly false for this target AND no valid matching target anywhere in full image. Negatives must differ from positive in exactly the designated semantic factor, preserving other clauses where possible. object: replace target noun with an absent plausible category (NOT add an object). co_occurrence: retain target and add a clearly absent companion object; reference must be visually checkable absent. attribute: change one visibly clear color/material/attribute to a false one, no other changes. relation: change/reverse a spatial relation between this target and an EXISTING identifiable reference; preserve target and reference identities; do not use absent references, imagined actions or impossible comic scenarios. Thus shared positive MUST contain a visibly certain target attribute and a clear spatial relation to existing reference. Avoid depth/occlusion ambiguities. If impossible mark usable false. Return JSON {"usable":true/false,"target_description":"...","bbox_valid":true/false,"positive":"...","negative":{"object":"...","co_occurrence":"...","attribute":"...","relation":"..."},"evidence":{"object":"...","co_occurrence":"...","attribute":"...","relation":"..."}}. Source (fallible): '''
def generate(x):
 iid=x['ref']['image_id']; fn='COCO_train2014_%012d.jpg'%iid; p=O/'images'/fn; im=Image.open(p).convert('RGB'); b=x['annotation']['bbox']; bbox=[b[0],b[1],b[0]+b[2],b[1]+b[3]]; ImageDraw.Draw(im).rectangle(bbox,outline='red',width=3); z=io.BytesIO(); im.save(z,format='JPEG',quality=95); (O/'images'/('boxed_'+fn)).write_bytes(z.getvalue()); out=O/('generation_%d.json'%iid)
 if out.exists():return json.load(open(out))['parsed']
 v=call(GEN+json.dumps({'source_sentences':x['ref']['sentences'],'bbox_xyxy':bbox,'size':im.size}),[p.read_bytes(),z.getvalue()],out); print(iid,v,flush=True);return v
if __name__=='__main__':
 data=json.load(open(O/'source_candidates.json'))['candidates']; ids=[526754,472639,478262,119815,224420,568562]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(generate,[x for x in data if x['ref']['image_id'] in ids]))
