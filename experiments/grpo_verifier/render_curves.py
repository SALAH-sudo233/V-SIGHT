#!/usr/bin/env python3
"""Render GRPO reward convergence curves from ~/grpo_reward_seq.json."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

seq = json.load(open(os.path.expanduser("~/grpo_reward_seq.json")))
OUT = os.path.expanduser("~/SVD/grpo_verifier/figs/grpo_reward_curves.png")

def smooth(ys, w=25):
    return [sum(ys[max(0,i-w):i+1])/(i-max(0,i-w)+1) for i in range(len(ys))]

fig, axL = plt.subplots(figsize=(9,5)); axR = axL.twinx()
colors={'v1':'#1f77b4','v2':'#2ca02c','v3':'#d62728'}
for name in ['v1','v2']:
    s=seq[name]; xs=[p[0] for p in s]; ys=smooth([p[1] for p in s])
    axL.plot(xs,ys,color=colors[name],lw=1.8,label=f'{name} total reward (L)')
s=seq['v3']; xs=[p[0] for p in s]; ys=smooth([p[1] for p in s])
axR.plot(xs,ys,color=colors['v3'],lw=1.8,ls='--',label='v3 log-score (R)')
axL.axvline(600,color='gray',ls=':',lw=1)
axL.text(650,axL.get_ylim()[0]+0.05,'~600 steps: ROH saturates',fontsize=9,color='gray')
axL.set_xlabel('training step'); axL.set_ylabel('total reward  (v1/v2, higher=better)')
axR.set_ylabel('log-score reward  (v3, ->0 better)')
axL.set_title('GRPO reward convergence (moving avg w=25)')
h1,l1=axL.get_legend_handles_labels(); h2,l2=axR.get_legend_handles_labels()
axL.legend(h1+h2,l1+l2,loc='center right',fontsize=9); axL.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT,dpi=120); print("wrote",OUT)
