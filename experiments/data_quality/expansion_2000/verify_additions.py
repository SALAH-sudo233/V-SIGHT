from image_review import *
IDS=[472639,478262,224420,119815]
P='''Independently audit each A/B referring-expression pair against BOTH attached actual images (original full scene and red candidate bbox). Do NOT infer labels from ordering; they are randomized. Assess full image, not only box. Accept only: exactly one expression uniquely identifies the boxed target; the other has NO matching target anywhere in image, is grammatical and not internally contradictory. Check bbox coverage and target identity (red box is fallible). Type object must replace target noun only, co_occurrence add absent companion, attribute change only attribute, relation change only spatial relation to an EXISTING unique reference while retaining target/reference. Flag synonym substitutions (monitor/television), ambiguity, insufficient absence evidence, non-unique positives, both-true, type mismatch, multi-factor changes. For each pair return {"type":...,"A_matches_box":"yes|no|uncertain","B_matches_box":"yes|no|uncertain","A_has_match_anywhere":"yes|no|uncertain","B_has_match_anywhere":"yes|no|uncertain","positive_unique":true/false,"bbox_valid":true/false,"grammar_ok":true/false,"single_factor_type_correct":true/false,"reference_clear":true/false,"pass":true/false,"reason":"specific visible evidence"}. Return JSON {"pairs":[...],"overall_pass":true/false}. For reference_clear true if relation reference visible unambiguous; false for ambiguous. Pairs: '''
def verify(iid):
 x=next(x for x in json.load(open(O/'source_candidates.json'))['candidates'] if x['ref']['image_id']==iid); v=json.load(open(O/('generation_%d.json'%iid)))['parsed']; before=json.loads(json.dumps(v))
 if iid==478262:v['negative']['object']=v['positive'].replace('woman','mannequin')
 if iid==472639:v['negative']['object']=v['positive'].replace('monitor','microwave')
 # Keep roof absence visibly checkable, no hidden-scene claims needed.
 pairs=[]; mapping={}
 for ht,neg in v['negative'].items():
  swap=int(hashlib.sha256((str(iid)+ht).encode()).hexdigest(),16)%2==0
  pairs.append({'type':ht,'A':neg if swap else v['positive'],'B':v['positive'] if swap else neg}); mapping[ht]='B' if swap else 'A'
 fn='COCO_train2014_%012d.jpg'%iid
 out=O/('verification_%d.json'%iid)
 audit=call(P+json.dumps(pairs),[(O/'images'/fn).read_bytes(),(O/'images'/('boxed_'+fn)).read_bytes()],out)
 rec={'image_id':iid,'source':x,'generation_before':before,'proposed_after':v,'audit_order_positive':mapping,'audit':audit}
 (O/('candidate_%d.json'%iid)).write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf8'); print(iid,json.dumps(audit),flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(verify,IDS))
