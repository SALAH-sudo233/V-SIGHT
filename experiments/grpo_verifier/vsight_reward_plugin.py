"""ms-swift external reward plugin for the V-SIGHT KEEP/REJECT binding verifier.

Register with:  swift rlhf --external_plugins vsight_reward_plugin.py \
                          --reward_funcs vsight_format vsight_decision

Fully verifiable reward, no reward model:
  vsight_format   : +0.3 if completion is a valid typed-JSON with decision+confidence
  vsight_decision : +1.0 if parsed decision == gold `solution`; -0.2 if unparseable; else 0
The dataset's extra `solution` column is passed to __call__ as the `solution` kwarg.
"""
import json
import re
from typing import List

from swift.rewards import orms
from swift.rewards.orm import ORM


def _parse_decision(text: str):
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            dec = str(obj.get("decision", "")).upper()
            ok = dec in ("KEEP", "REJECT") and "confidence" in obj
            return (dec if dec in ("KEEP", "REJECT") else None), ok
        except Exception:
            pass
    d = re.search(r'"?decision"?\s*:\s*"?(KEEP|REJECT)"?', text, re.I)
    return (d.group(1).upper() if d else None), False


class VSightFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        out = []
        for c in completions:
            _, ok = _parse_decision(c)
            out.append(0.3 if ok else 0.0)
        return out


class VSightDecision(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out = []
        for c, gold in zip(completions, solution):
            dec, _ = _parse_decision(c)
            if dec is None:
                out.append(-0.2)
            elif dec == str(gold).upper():
                out.append(1.0)
            else:
                out.append(0.0)
        return out


orms['vsight_format'] = VSightFormat
orms['vsight_decision'] = VSightDecision
