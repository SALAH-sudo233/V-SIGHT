import json
from swift.rewards import orms
from swift.rewards.orm import ORM
VALID_B={'MATCH','MISMATCH','NA'}; VALID_D={'KEEP','REJECT'}
def valid(t):
 try:
  def h(items):
   ks=[k for k,_ in items]
   if len(ks)!=len(set(ks)): raise ValueError()
   return dict(items)
  o=json.loads(t.strip(),object_pairs_hook=h)
  if set(o)!={'binding','decision'}: return False
  b,d=o['binding'],o['decision']
  return isinstance(b,str) and isinstance(d,str) and b.upper() in VALID_B and d.upper() in VALID_D
 except Exception:return False
class Format(ORM):
 def __call__(self,completions,**kw): return [0.1 if valid(c) else -0.2 for c in completions]
class Decision(ORM):
 def __call__(self,completions,solution,**kw):
  ss=solution if isinstance(solution,list) else [solution]*len(completions)
  out=[]
  for c,s in zip(completions,ss):
   try:
    o=json.loads(c.strip()); d=o.get('decision') if isinstance(o,dict) else None
    out.append(1.0 if isinstance(d,str) and d.upper()==str(s).upper() else (-0.2 if d is None else 0.0))
   except: out.append(-0.2)
  return out
orms['vsight_decision_only_format']=Format
orms['vsight_decision_only']=Decision
