#!/usr/bin/env python3
"""GRPO reward curves — publication style: faint raw + bold moving-avg, per-run panels."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

seq = json.load(open(os.path.expanduser("~/grpo_reward_seq.json")))
OUT = os.path.expanduser("~/SVD/grpo_verifier/figs/grpo_reward_curves.png")

def smooth(ys, w):
    return [sum(ys[max(0,i-w):i+1])/(i-max(0,i-w)+1) for i in range(len(ys))]

W=80
runs=[('v1','#1f77b4','v1: binary reward'),
      ('v2','#2ca02c','v2: ROH-weighted + Brier'),
      ('v3','#d62728','v3: log-score (proper)')]
fig,axes=plt.subplots(1,3,figsize=(15,4.2))
for ax,(name,c,title) in zip(axes,runs):
    s=seq[name]; xs=[p[0] for p in s]; ys=[p[1] for p in s]
    ax.plot(xs,ys,color=c,alpha=0.18,lw=0.7)              # raw (noise)
    ax.plot(xs,smooth(ys,W),color=c,lw=2.2)               # moving avg
    # clip y to 2nd–98th pct so heavy-tail spikes don't squash the trend
    sv=sorted(ys); lo=sv[int(0.02*len(sv))]; hi=sv[int(0.98*len(sv))-1]
    pad=0.1*(hi-lo+1e-6); ax.set_ylim(lo-pad,hi+pad)
    ax.axvline(600,color='gray',ls=':',lw=1)
    ax.set_title(title,fontsize=11); ax.set_xlabel('training step')
    ax.grid(alpha=0.25)
    if name=='v3': ax.set_ylabel('log-score reward (->0 better)')
    else: ax.set_ylabel('total reward (higher better)')
    ax.text(620, ax.get_ylim()[0]+0.06*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            '~600: ROH saturates', fontsize=8, color='gray')
fig.suptitle(f'GRPO reward convergence — faint=raw, bold=moving avg (w={W})',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(OUT,dpi=120); print("wrote",OUT)
