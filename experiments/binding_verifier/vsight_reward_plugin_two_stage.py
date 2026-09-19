import json
from swift.rewards import orms
from swift.rewards.orm import ORM
VALID_B={'MATCH','MISMATCH','NA'}; VALID_D={'KEEP','REJECT'}
def parse(t):
 try:
  def h(items):
   k=[x[0] for x in items]
   if len(k)!=len(set(k)): raise ValueError()
   return dict(items)
  o=json.loads(t.strip(),object_pairs_hook=h)
  if set(o)!= {'binding','decision'}: return None,None
  b,d=o['binding'].upper(),o['decision'].upper()
  return (b,d) if b in VALID_B and d in VALID_D else (None,None)
 except:return None,None
def score(t,sol,ht,stage):
 b,d=parse(t); sol=str(sol).upper(); roh=str(ht) in {'attribute','relation'}
 gb='MATCH' if sol=='KEEP' and roh else ('MISMATCH' if sol=='REJECT' and roh else 'NA')
 # Fixed-scale rewards; stage2 increases only the penalty for ROH false KEEP.
 false_keep = roh and sol=='REJECT' and d=='KEEP'
 dp = -1.5 if (stage=='stage2' and false_keep) else (1.0 if d==sol else (-0.2 if d is None else 0.0))
 return (0.1 if b and d else 0.0, 1.0 if (roh and b==gb) else 0.0, dp)
def allscores(completions,solution,htype,stage='stage1',**kw):
 return [score(c, solution[i] if isinstance(solution,list) else solution, htype[i] if isinstance(htype,list) else htype, stage[i] if isinstance(stage,list) else stage) for i,c in enumerate(completions)]
class F(ORM):
 def __call__(self,completions,stage='stage1',**kw):return [x[0] for x in allscores(completions,stage=stage,**kw)]
class B(ORM):
 def __call__(self,completions,stage='stage1',**kw):return [x[1] for x in allscores(completions,stage=stage,**kw)]
class D(ORM):
 def __call__(self,completions,stage='stage1',**kw):return [x[2] for x in allscores(completions,stage=stage,**kw)]
orms['vsight_two_format']=F; orms['vsight_two_binding']=B; orms['vsight_two_decision']=D
