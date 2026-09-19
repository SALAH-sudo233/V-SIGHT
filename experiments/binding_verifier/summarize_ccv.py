import json,glob,os,collections,sys
base=sys.argv[1]; paths=glob.glob(base+'/ccv_eval/verifier_*.json')+glob.glob(base+'/upstream_ccv_eval_results/*.json')
for p in paths:
 rows=json.load(open(p)); c=collections.defaultdict(lambda:[0,0,0,0,0]);
 for r in rows:
  h=r.get('htype','?'); g='BOH' if h in ('object','co_occurrence') else 'ROH'; pr=r.get('pred') or {}; b=pr.get('binding');d=pr.get('decision'); gd=r.get('gold'); gb=r.get('gold_binding'); grp=['ALL',g]
  for x in grp:
   c[x][0]+=1;c[x][1]+=b is not None;c[x][2]+=d==gd;c[x][3]+=b==gb;c[x][4]+=r.get('proposal_iou',0)>=.5
 print(os.path.basename(p), 'N',len(rows), ' '.join(f'{g}:parse={v[1]/v[0]:.3f},dec={v[2]/v[0]:.3f},bind={v[3]/v[0]:.3f},iou50={v[4]/v[0]:.3f}' for g,v in c.items()))
