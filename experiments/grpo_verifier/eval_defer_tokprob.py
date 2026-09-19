#!/usr/bin/env python3
"""DEFER eval using DECISION-TOKEN probability as confidence (not self-reported).

Instead of trusting the model's spoken "confidence" field (which collapses to 1.0),
we read the model's internal probability over the KEEP vs REJECT decision:
we force the JSON prefix '{"decision": "' then look at the next-token logits,
compare P(" KEEP"/"KEEP") vs P(" REJECT"/"REJECT"), softmax -> p_keep.
  decision = argmax; confidence = max(p_keep, 1-p_keep)  (prob of chosen class)
This is the model's real uncertainty, immune to the spoken-confidence prior.
Then the same DEFER working curve as eval_defer.py.
"""
import json, os, re, sys, argparse, torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
    tok=proc.tokenizer

    # candidate token ids for KEEP / REJECT (first subword, with and w/o leading space)
    def first_ids(word):
        ids=set()
        for s in (word, " "+word):
            t=tok.encode(s, add_special_tokens=False)
            if t: ids.add(t[0])
        return ids
    keep_ids=first_ids("KEEP"); rej_ids=first_ids("REJECT")

    recs=[]
    for i,(img,q,bbox,gold,ht) in enumerate(items):
        x0,y0,x1,y1=[int(v) for v in bbox]
        msg=[{"role":"user","content":[{"type":"image","image":img},
              {"type":"text","text":PROMPT.format(x0=x0,y0=y0,x1=x1,y1=y1,q=q)}]}]
        # render prompt then append the forced JSON prefix so next token = the decision word
        text=proc.apply_chat_template([msg],add_generation_prompt=True,tokenize=False)
        if isinstance(text,list): text=text[0]
        text=text+'{"decision": "'
        from PIL import Image as _Img
        inp=proc(text=[text], images=[_Img.open(img).convert('RGB')],
                 return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            logits=model(**inp).logits[0,-1,:]   # next-token logits
        lk=max(logits[t].item() for t in keep_ids)
        lr=max(logits[t].item() for t in rej_ids)
        import math
        pk=math.exp(lk)/(math.exp(lk)+math.exp(lr))
        dec="KEEP" if pk>=0.5 else "REJECT"
        conf=max(pk,1-pk)
        grp="BOH" if ht in ("object","co_occurrence") else "ROH"
        recs.append({"gold":gold,"dec":dec,"conf":conf,"correct":int(dec==gold),"grp":grp})
        if (i+1)%400==0: print(f"  {i+1}/{len(items)}",flush=True)

    if a.dump:
        with open(a.dump,"w") as f:
            for r in recs: f.write(json.dumps(r)+"\n")
    def acc(rs): return sum(r["correct"] for r in rs)/len(rs) if rs else float('nan')
    print(f"=== N={len(recs)} lora={a.lora or 'ZEROSHOT'} (token-prob confidence) ===")
    for g in ("ALL","BOH","ROH"):
        rs=[r for r in recs if g=="ALL" or r["grp"]==g]
        print(f"  {g}: acc={acc(rs):.4f} (n={len(rs)})")
    print("=== DEFER curve ===")
    print("  tau | defer_rate | error_recall | false_defer | auto_acc | auto_ROH_acc")
    for tau in (0.55,0.6,0.65,0.7,0.75,0.8,0.9):
        auto=[r for r in recs if r["conf"]>=tau]
        wrong=[r for r in recs if r["correct"]==0]; corr=[r for r in recs if r["correct"]==1]
        er=sum(1 for r in wrong if r["conf"]<tau)/len(wrong) if wrong else float('nan')
        fd=sum(1 for r in corr if r["conf"]<tau)/len(corr) if corr else float('nan')
        aacc=acc(auto) if auto else float('nan')
        aroh=[r for r in auto if r["grp"]=="ROH"]; aracc=acc(aroh) if aroh else float('nan')
        dr=1-len(auto)/len(recs)
        print(f"  {tau:.2f} |   {dr:.3f}    |    {er:.3f}     |    {fd:.3f}    |  {aacc:.3f}  |   {aracc:.3f}")

if __name__=="__main__":
    main()
