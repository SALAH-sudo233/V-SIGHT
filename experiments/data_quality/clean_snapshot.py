"""Conservative snapshot triage; never changes original data or live audit."""
import json,collections,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).parent
SOURCE=Path('C:/Users/30796/AppData/Local/Temp/grpo/pairs_dump.json')
FIELDS=['A_matches_box','B_matches_box','A_unique','reference_clear','text_conflict_A','text_conflict_B','likely_bad_pair','repair_needed']
def verdict(a):
    required={'A_matches_box':'yes','B_matches_box':'no','A_unique':'yes','text_conflict_A':'no','text_conflict_B':'no','likely_bad_pair':'no','repair_needed':'no'}
    issues=[k for k,v in required.items() if a.get(k)!=v]
    if a.get('reference_clear') not in ('yes','none'):issues.append('reference_clear')
    if a.get('problem_types')!=[]:issues.append('problem_types')
    return issues

def main():
    raw=(ROOT/'audit_v2.jsonl').read_bytes(); groups=collections.defaultdict(list); bad=0
    for line in raw.splitlines():
        try:
            r=json.loads(line); groups[(r.get('set'),r.get('sid'))].append(r)
        except (ValueError,TypeError):bad+=1
    rows=json.loads(SOURCE.read_text(encoding='utf8')); bins=collections.defaultdict(list)
    for r in rows:
        key=(r['set'],r['sid']); attempts=groups.get(key,[])
        valid=[x for x in attempts if not x.get('error') and isinstance(x.get('audit'),dict) and all(f in x['audit'] for f in FIELDS)]
        rec=dict(r)
        if not valid:
            status='retry_required' if attempts else 'unaudited'
            rec['cleaning_reasons']=[status]
        else:
            fingerprints={json.dumps({f:x['audit'].get(f) for f in FIELDS},sort_keys=True) for x in valid}
            issues=sorted(set(v for x in valid for v in verdict(x['audit'])))
            if len(fingerprints)>1:issues.append('repeated_audits_disagree')
            status='quarantine' if issues else 'provisional_keep'
            rec['cleaning_reasons']=issues
            rec['audit_attempts']=len(attempts)
            rec['audit_evidence']=[{'audit':x['audit'],'model':x.get('model')} for x in valid]
        rec['cleaning_status']=status;bins[status].append(rec)
    assert sum(map(len,bins.values()))==len(rows)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dst=ROOT/'cleaning'/stamp;dst.mkdir(parents=True,exist_ok=False)
    for status in ['provisional_keep','quarantine','retry_required','unaudited']:
        (dst/(status+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in bins[status]),encoding='utf8')
    manifest={'source_pairs':len(rows),'snapshot_sha256':hashlib.sha256(raw).hexdigest(),'malformed_or_partial_lines':bad,'counts':{k:len(v) for k,v in bins.items()},'by_set':{k:dict(collections.Counter(r['set'] for r in v)) for k,v in bins.items()},'note':'Provisional screening only, NOT gold/verified repairs. No original data modified. New4 separately audited expansion not included. Audit A/B fields are normalized positive/negative; raw reason A/B may refer to original swapped order.'}
    (dst/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    (dst/'README.md').write_text('# 阶段清洗快照\n\nprovisional_keep：现有审核字段一致、暂可保留，非人工金标。\nquarantine：疑似错误、含混或重复审核分歧，隔离复核，未删除。\nretry_required：请求错误或审核结构不完整。\nunaudited：尚未审核，不视为干净数据。\n\n原始数据及在线审核日志未改动；修复建议未自动采纳。新增4图独立存于 expansion_2000，未混入此快照。\n',encoding='utf8')
    print(json.dumps({'directory':str(dst),**manifest},ensure_ascii=False,indent=2))
if __name__=='__main__':
    assert verdict({'A_matches_box':'yes','B_matches_box':'no','A_unique':'yes','reference_clear':'none','text_conflict_A':'no','text_conflict_B':'no','likely_bad_pair':'no','repair_needed':'no','problem_types':[]})==[]
    assert verdict({})
    main()
