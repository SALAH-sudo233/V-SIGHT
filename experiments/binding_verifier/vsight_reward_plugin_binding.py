import json,re
from swift.rewards import orms
from swift.rewards.orm import ORM
VALID_B={'MATCH','MISMATCH','NA'};VALID_D={'KEEP','REJECT'}
def parse_binding(text):
 raw=text.strip()
 if not raw.startswith('{') or not raw.endswith('}'):
  return None,None
 def pairs(items):
  keys=[k for k,_ in items]
  if len(keys)!=len(set(keys)): raise ValueError('duplicate key')
  return dict(items)
 try:o=json.loads(raw,object_pairs_hook=pairs)
 except Exception:return None,None
 if not isinstance(o,dict) or set(o)!= {'binding','decision'}:return None,None
 b=o['binding'];d=o['decision']
 if not isinstance(b,str) or not isinstance(d,str):return None,None
 b=b.upper();d=d.upper()
 return (b,d) if b in VALID_B and d in VALID_D else (None,None)
def gold_binding(sol,ht):
 sol=str(sol).upper();ht=str(ht)
 return ('MATCH' if sol=='KEEP' else ('MISMATCH' if ht in {'attribute','relation'} else 'NA'),sol)
def binding_rewards(completions,solution,htype,**kwargs):
 if not isinstance(solution,list):solution=[solution]*len(completions)
 if not isinstance(htype,list):htype=[htype]*len(completions)
 out=[]
 for text,sol,ht in zip(completions,solution,htype):
  b,d=parse_binding(text);gb,gd=gold_binding(sol,ht)
  out.append({'format':0.1 if b and d else 0.0,'binding':1.5 if b==gb and gb!='NA' else 0.0,'decision':1.0 if d==gd else (-0.2 if d is None else 0.0)})
 return out
class Format(ORM):
 def __call__(self,completions,**kwargs):return [binding_rewards([c],['KEEP'],['object'])[0]['format'] for c in completions]
class Binding(ORM):
 def __call__(self,completions,solution,htype=None,**kwargs):return [x['binding'] for x in binding_rewards(completions,solution,htype or ['unknown']*len(completions))]
class Decision(ORM):
 def __call__(self,completions,solution,htype=None,**kwargs):return [x['decision'] for x in binding_rewards(completions,solution,htype or ['unknown']*len(completions))]
orms['vsight_binding_format']=Format;orms['vsight_binding']=Binding;orms['vsight_binding_decision']=Decision
