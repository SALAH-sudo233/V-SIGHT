import json,sys
from swift.rewards import orms
from swift.rewards.orm import ORM
VALID_B={'MATCH','MISMATCH','NA'}; VALID_D={'KEEP','REJECT'}
def parse(t):
    try:
        def hook(items):
            keys=[k for k,_ in items]
            if len(keys)!=len(set(keys)): raise ValueError('duplicate key')
            return dict(items)
        o=json.loads(t.strip(),object_pairs_hook=hook)
        if not isinstance(o,dict) or set(o)!={'binding','decision'}: return None,None
        b,d=o['binding'],o['decision']
        if not isinstance(b,str) or not isinstance(d,str): return None,None
        b,d=b.upper(),d.upper()
        return (b,d) if b in VALID_B and d in VALID_D else (None,None)
    except Exception:return None,None
def score(text,sol,ht):
    b,d=parse(text); sol=str(sol).upper(); roh=str(ht) in {'attribute','relation'}
    gb='MATCH' if sol=='KEEP' and roh else ('MISMATCH' if sol=='REJECT' and roh else 'NA')
    return (0.1 if b is not None and d is not None else 0.0, 1.5 if roh and b==gb else 0.0, 1.0 if d==sol else (-0.2 if d is None else 0.0))
def rows(completions,solution,htype,**kw):
    ss=solution if isinstance(solution,list) else [solution]*len(completions)
    hs=htype if isinstance(htype,list) else [htype]*len(completions)
    return [score(c,s,h) for c,s,h in zip(completions,ss,hs)]
class F(ORM):
    def __call__(self,completions,**kw): return [x[0] for x in rows(completions,**kw)]
class B(ORM):
    def __call__(self,completions,solution,htype,**kw): return [x[1] for x in rows(completions,solution,htype,**kw)]
class D(ORM):
    def __call__(self,completions,solution,htype,**kw): return [x[2] for x in rows(completions,solution,htype,**kw)]
orms['vsight_symmetric_format']=F; orms['vsight_symmetric_binding']=B; orms['vsight_symmetric_decision']=D
