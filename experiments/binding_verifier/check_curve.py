import json,sys,statistics
p=sys.argv[1]
rows=[]
for line in open(p,encoding='utf-8',errors='ignore'):
 try:
  x=json.loads(line)
  if 'global_step/max_steps' in x: rows.append(x)
 except: pass
print('rows',len(rows),'steps',rows[0].get('global_step/max_steps'),rows[-1].get('global_step/max_steps'))
for a,b in [(1,20),(21,40),(41,60),(61,80),(81,999)]:
 z=rows[a-1:min(b,len(rows))]
 if z:
  def vals(k): return [float(x[k]) for x in z if k in x]
  print(a,b,'reward_mean',round(statistics.mean(vals('reward')),3),'format',round(statistics.mean(vals('rewards/F/mean')),3),'binding',round(statistics.mean(vals('rewards/B/mean')),3),'decision',round(statistics.mean(vals('rewards/D/mean')),3),'kl',round(statistics.mean(vals('kl')),4),'zero_std',round(statistics.mean(vals('frac_reward_zero_std')),3),'clip',round(statistics.mean(vals('clip_ratio/region_mean')),4))
print('last', {k:rows[-1].get(k) for k in ['global_step/max_steps','reward','reward_std','rewards/F/mean','rewards/B/mean','rewards/D/mean','kl','frac_reward_zero_std','clip_ratio/region_mean']})
