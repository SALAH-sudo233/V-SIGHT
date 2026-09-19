#!/usr/bin/env python3
"""Structured binding reward for V-SIGHT GRPO.

Auxiliary labels are derived only from the existing counterfactual construction:
positive -> PRESENT/MATCH/KEEP; BOH negative -> ABSENT/NA/REJECT;
ROH negative -> PRESENT/MISMATCH/REJECT. These are construction labels, not
independent human annotations of target/reference roles.
"""
import json,re
from typing import List,Optional,Tuple
from swift.rewards import orms
from swift.rewards.orm import ORM

VALID_TARGET={'PRESENT','ABSENT'}
VALID_BINDING={'MATCH','MISMATCH','NA'}
VALID_DECISION={'KEEP','REJECT'}

def parse_structured(text:str)->Tuple[Optional[str],Optional[str],Optional[str]]:
 m=re.search(r'\{[^{}]*\}',text,re.S)
 if not m:return None,None,None
 try:o=json.loads(m.group(0))
 except Exception:return None,None,None
 t=str(o.get('target','')).upper();b=str(o.get('binding','')).upper();d=str(o.get('decision','')).upper()
 return (t if t in VALID_TARGET else None,
         b if b in VALID_BINDING else None,
         d if d in VALID_DECISION else None)

def labels(solution,htype):
 sol=str(solution).upper(); ht=str(htype)
 target='PRESENT' if sol=='KEEP' or ht in {'attribute','relation'} else 'ABSENT'
 binding='MATCH' if sol=='KEEP' else ('MISMATCH' if ht in {'attribute','relation'} else 'NA')
 return target,binding,sol

def structured_rewards(completions,solution,htype,**kwargs):
 out=[]
 if not isinstance(solution,list):solution=[solution]*len(completions)
 if not isinstance(htype,list):htype=[htype]*len(completions)
 for text,gold,kind in zip(completions,solution,htype):
  t,b,d=parse_structured(text); gt,gb,gd=labels(gold,kind)
  fmt=0.1 if t and b and d else 0.0
  rt=0.5 if t==gt else 0.0
  rb=1.5 if b==gb and gb!='NA' else 0.0
  rd=1.0 if d==gd else (-0.2 if d is None else 0.0)
  out.append({'format':fmt,'target':rt,'binding':rb,'decision':rd,'total':fmt+rt+rb+rd})
 return out

class VSightStructuredFormat(ORM):
 def __call__(self,completions,**kwargs)->List[float]:return [0.1 if all(parse_structured(c)) else 0.0 for c in completions]
class VSightStructuredTarget(ORM):
 def __call__(self,completions,solution,htype=None,**kwargs)->List[float]:return [x['target'] for x in structured_rewards(completions,solution,htype or ['unknown']*len(completions))]
class VSightStructuredBinding(ORM):
 def __call__(self,completions,solution,htype=None,**kwargs)->List[float]:return [x['binding'] for x in structured_rewards(completions,solution,htype or ['unknown']*len(completions))]
class VSightStructuredDecision(ORM):
 def __call__(self,completions,solution,**kwargs)->List[float]:
  return [x['decision'] for x in structured_rewards(completions,solution,kwargs.get('htype',['unknown']*len(completions)))]

orms['vsight_struct_format']=VSightStructuredFormat
orms['vsight_struct_target']=VSightStructuredTarget
orms['vsight_struct_binding']=VSightStructuredBinding
orms['vsight_struct_decision']=VSightStructuredDecision
