import pathlib,shutil,hashlib,json
w=pathlib.Path('/home/u2025141034/SVD/grpo_verifier')
run=next((w/'runs/binding_grpo_v1').glob('v0-*'))
dest=w/'saved_ckpts/binding_grpo_v1';dest.mkdir(exist_ok=True)
for ck in sorted(run.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1])):
 step=ck.name.split('-')[-1];out=dest/f'ck{step}';
 if out.exists():shutil.rmtree(out)
 shutil.copytree(ck,out)
 print(out,hashlib.sha256((out/'adapter_model.safetensors').read_bytes()).hexdigest())
