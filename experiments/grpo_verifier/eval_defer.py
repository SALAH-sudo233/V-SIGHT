#!/usr/bin/env python3
"""Decision-accuracy + DEFER-effectiveness eval on 500-dev.

Beyond accuracy, this dumps per-sample (decision, confidence, correct, group) and
computes the DEFER working curve — the metric that actually says whether the
verifier can drive the agentic flywheel:

  For a DEFER threshold tau (DEFER if confidence < tau):
    error_recall@tau  = P(conf<tau | wrong)   # wrong cases caught for human review (higher=better)
    false_defer@tau   = P(conf<tau | correct) # correct cases needlessly sent to human (lower=better)
    auto_acc@tau      = accuracy among auto-passed (conf>=tau) cases (flywheel's autonomous quality)
    defer_rate@tau    = fraction sent to human

A verifier good for the flywheel: high error_recall at a tau where defer_rate stays modest,
i.e. its wrong answers really do carry low confidence.
"""
import json, os, re, sys, argparse, torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def parse(text):
    dec, conf = None, None
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            d = str(o.get("decision","")).upper()
            if d in ("KEEP","REJECT"): dec = d
            c = o.get("confidence", None)
            if isinstance(c,(int,float)): conf = max(0.0,min(1.0,float(c)))
        except Exception: pass
    if dec is None:
        d = re.search(r'"?decision"?\s*:\s*"?(KEEP|REJECT)"?', text, re.I)
        if d: dec = d.group(1).upper()
    if conf is None:
        c = re.search(r'"?confidence"?\s*:\s*([01](?:\.\d+)?)', text, re.I)
        if c: conf = float(c.group(1))
    return dec, conf

PROMPT=("Look at the region [{x0},{y0},{x1},{y1}]. Does the phrase \"{q}\" "
        "correctly and accurately describe the object in that region? "
        "Answer with ONLY a JSON object: {{\"decision\": \"KEEP\" or \"REJECT\", "
        "\"confidence\": a number 0-1}}. KEEP if fully accurate; REJECT if any part is wrong.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dev",required=True); ap.add_argument("--img_dir",required=True)
    ap.add_argument("--model",required=True); ap.add_argument("--lora",default="")
    ap.add_argument("--n",type=int,default=0); ap.add_argument("--dump",default="")
    a=ap.parse_args()
    rows=json.load(open(a.dev)); items=[]
    for r in rows:
        bbox=r.get("gt_bbox_xyxy") or r.get("positive_bbox")
        if not bbox: continue
        img=os.path.join(a.img_dir, os.path.basename(r["image_filename"]))
        if not os.path.exists(img): continue
        ht=r.get("hallucination_type","?")
        if r.get("positive_text"): items.append((img,r["positive_text"],bbox,"KEEP",ht))
        if r.get("negative_text") and r["negative_text"]!=r.get("positive_text"):
            items.append((img,r["negative_text"],bbox,"REJECT",ht))
    if a.n: items=items[:a.n]

    proc=AutoProcessor.from_pretrained(a.model,trust_remote_code=True,min_pixels=256*28*28,max_pixels=768*28*28)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,device_map="cuda:0")
    if a.lora:
        from peft import PeftModel; model=PeftModel.from_pretrained(model,a.lora)
    model.eval()

    recs=[]
    for i,(img,q,bbox,gold,ht) in enumerate(items):
        x0,y0,x1,y1=[int(v) for v in bbox]
        msg=[{"role":"user","content":[{"type":"image","image":img},
              {"type":"text","text":PROMPT.format(x0=x0,y0=y0,x1=x1,y1=y1,q=q)}]}]
        inp=proc.apply_chat_template([msg],add_generation_prompt=True,tokenize=True,
                                     return_dict=True,return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out=model.generate(**inp,max_new_tokens=48,do_sample=False)
        txt=proc.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True)
        dec,conf=parse(txt)
        grp="BOH" if ht in ("object","co_occurrence") else "ROH"
        recs.append({"gold":gold,"dec":dec,"conf":conf if conf is not None else -1,
                     "correct":int(dec==gold),"grp":grp})
        if (i+1)%400==0: print(f"  {i+1}/{len(items)}",flush=True)

    if a.dump:
        with open(a.dump,"w") as f:
            for r in recs: f.write(json.dumps(r)+"\n")

    def acc(rs): return sum(r["correct"] for r in rs)/len(rs) if rs else float('nan')
    print(f"=== N={len(recs)} lora={a.lora or 'ZEROSHOT'} ===")
    for g in ("ALL","BOH","ROH"):
        rs=[r for r in recs if g=="ALL" or r["grp"]==g]
        print(f"  {g}: acc={acc(rs):.4f} (n={len(rs)})")

    # DEFER working curve — only over samples with a parsed numeric confidence
    conf_recs=[r for r in recs if r["conf"]>=0]
    print(f"=== DEFER curve (over {len(conf_recs)} conf-parsed samples) ===")
    print("  tau | defer_rate | error_recall | false_defer | auto_acc | auto_ROH_acc")
    for tau in (0.3,0.4,0.5,0.6,0.7,0.8,0.9):
        deferred=[r for r in conf_recs if r["conf"]<tau]
        auto=[r for r in conf_recs if r["conf"]>=tau]
        wrong=[r for r in conf_recs if r["correct"]==0]
        corr=[r for r in conf_recs if r["correct"]==1]
        er = sum(1 for r in wrong if r["conf"]<tau)/len(wrong) if wrong else float('nan')
        fd = sum(1 for r in corr if r["conf"]<tau)/len(corr) if corr else float('nan')
        aacc = acc(auto) if auto else float('nan')
        auto_roh=[r for r in auto if r["grp"]=="ROH"]
        aroh = acc(auto_roh) if auto_roh else float('nan')
        dr = len(deferred)/len(conf_recs)
        print(f"  {tau:.1f} |   {dr:.3f}    |    {er:.3f}     |    {fd:.3f}    |  {aacc:.3f}  |   {aroh:.3f}")

if __name__=="__main__":
    main()
