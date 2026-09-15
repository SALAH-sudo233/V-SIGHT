"""Read-only raw-score evaluation on frozen E0 image roles; no fitting."""
import json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent

def cid(s):
    match = re.search(r'(\d{12})', str(s))
    if not match: raise ValueError(s)
    return int(match[1])

def load(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def auc(p, n):
    if not p or not n: return None
    a, b = np.asarray(p)[:,None], np.asarray(n)[None,:]
    return float(((a>b)+0.5*(a==b)).mean())

def threshold(scores, budget=.03):
    if not scores: return None
    return max(t for t in sorted(set(scores)) if sum(x<t for x in scores)/len(scores)<=budget)

def correct(r): return bool(r['label_exists']) and (r.get('iou') or 0)>=.5

def evaluate(cal, dev):
    pc = [r['p_correct'] for r in cal if correct(r)]
    dp = [r['p_correct'] for r in dev if correct(r)]
    t = threshold(pc)
    out = {'n_cal':len(cal),'n_dev':len(dev),'n_cal_correct':len(pc),'n_dev_correct':len(dp), 'threshold':t,
           'realFNR':sum(p<t for p in dp)/len(dp) if dp and t is not None else None}
    for name, types in {'BOH':{'object','co_occurrence'}, 'ROH':{'relation','attribute'},'relation':{'relation'},'attribute':{'attribute'}}.items():
        neg = [r['p_correct'] for r in dev if not r['label_exists'] and r['htype'] in types]
        out[name] = {'n_negative':len(neg),'AUROC':auc(dp,neg),'catch':sum(p<t for p in neg)/len(neg) if neg and t is not None else None}
    return out

def main():
    manifest = json.loads((ROOT.parent/'e0_audit/manifests/binding_v2_splits.json').read_text())
    roles={k:set(v['coco_ids']) for k,v in manifest['roles'].items()}
    calids, devids = roles['dev500_calibration'],roles['dev500_development']
    assert not calids & devids
    raw={name:load(ROOT/'results'/fn) for name,fn in [('old','old_b1b_3b_eval.jsonl'),('e6','e6_500dev_eval.jsonl')]}
    keyed={}
    for name,rows in raw.items():
        assert all('p_correct' in r for r in rows), 'Missing score/error row'
        keyed[name]={(r['model'],r['sample_id']):r for r in rows}
        assert len(keyed[name])==len(rows), 'Duplicate keys'
    common=set(keyed['old'])&set(keyed['e6'])
    mismatches=[]
    for key in common:
        if any(keyed['old'][key].get(f)!=keyed['e6'][key].get(f) for f in ['htype','label_exists','iou']): mismatches.append(key)
    assert not mismatches, mismatches[:5]
    train=load(ROOT/'e6_vlm_rows.jsonl')
    trainids={cid(r['image_filename']) for r in train if r['split']=='train'}
    e2=load(ROOT.parent/'e2_oracle/e2_items.jsonl')
    e2ids={cid(r['img']) for r in e2}
    dev500=set.union(*(roles[k] for k in ['dev500_fusion_fit','dev500_calibration','dev500_development']))
    out={'protocol':'Raw P(yes); threshold on calibration only, evaluation on development only; paired common keys. No new VLM inference.',
         'audit':{'raw_counts':{k:len(v) for k,v in raw.items()},'common_keys':len(common),'old_only':len(set(keyed['old'])-common),'e6_only':len(set(keyed['e6'])-common),'metadata_mismatch':len(mismatches),'candidate_query_equality':'Not auditable from score files; source records still required', 'prepared_train_images':len(trainids),'train_dev500_overlap':len(trainids&dev500),'train_E2_overlap':len(trainids&e2ids),'E2_total_images':len(e2ids), 'note':'Train overlap uses prepared train split; actual balanced training subset may be smaller.'},'models':{}}
    assert not trainids&dev500, 'Train/dev500 contamination'
    for model in sorted({k[0] for k in common}):
        keys=sorted(k for k in common if k[0]==model)
        out['models'][model]={}
        for name in ['old','e6']:
            rows=[keyed[name][k] for k in keys]
            out['models'][model][name]=evaluate([r for r in rows if cid(r['sample_id']) in calids],[r for r in rows if cid(r['sample_id']) in devids])
    dst=ROOT/'results/e6_frozen_calibration.json'
    dst.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
