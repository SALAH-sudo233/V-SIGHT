import json,re
from swift.rewards import orms
from swift.rewards.orm import ORM
VALID_B={'MATCH','MISMATCH','NA'}; VALID_D={'KEEP','REJECT'}
def parse_binding(text):
 raw=text.strip()
 if not raw.startswith('{') or not raw.endswith('}'): return None,None
 try:
  def hook(items):
   ks=[k for k,_ in items]
   if len(ks)!=len(set(ks)): raise ValueError()
   return dict(items)
  o=json.loads(raw,object_pairs_hook=hook)
 except Exception:return None,None
 if not isinstance(o,dict) or set(o)!={'binding','decision'}:return None,None
 b,d=o['binding'],o['decision']
 if not isinstance(b,str) or not isinstance(d,str):return None,None
 b,d=b.upper(),d.upper()
 return (b,d) if b in VALID_B and d in VALID_D else (None,None)
def one(text,sol,ht,group):
 b,d=parse_binding(text); sol=str(sol).upper(); ht=str(ht); group=str(group)
 gb='MATCH' if sol=='KEEP' and group=='ROH' else ('MISMATCH' if sol=='REJECT' and group=='ROH' else 'NA')
 # Preserve decision balance; emphasize ROH binding/decision equally.
 w=1.5 if group=='ROH' else 1.0
 return (0.1 if b and d else 0.0, w if b==gb and group=='ROH' else 0.0, w if d==sol else (-0.2 if d is None else 0.0))
def rewards(completions,solution,htype,hallucination_group,**kwargs):
 vals=[]
 for i,c in enumerate(completions):
  sol=solution[i] if isinstance(solution,list) else solution
  ht=htype[i] if isinstance(htype,list) else htype
  gr=hallucination_group[i] if isinstance(hallucination_group,list) else hallucination_group
  vals.append(one(c,sol,ht,gr))
 return vals
class F(ORM):
 def __call__(self,completions,**kw): return [x[0] for x in rewards(completions,**kw)]
class B(ORM):
 def __call__(self,completions,**kw): return [x[1] for x in rewards(completions,**kw)]
class D(ORM):
 def __call__(self,completions,**kw): return [x[2] for x in rewards(completions,**kw)]
orms['vsight_group_format']=F; orms['vsight_group_binding']=B; orms['vsight_group_decision']=D
