#!/usr/bin/env python3
"""Write an SFT or DPO config from the upstream TSR configs.

Only paths and the per-GPU batch split change; the global batch stays 32 on any number of GPUs.
  sft: template settings/sft/config/config.yaml (upstream: 8 GPUs x 4 x accumulation 1)
       -> per-device 2, accumulation 32 / (2 * ngpu). Sequences are packed to 2048 tokens, so this is equivalent.
  dpo: template settings/sr/config/dpo/config_full_iter0.yaml (upstream: 8 GPUs x 2 x accumulation 2).
       The TSR DPO configs are identical to it except paths (and iter0_tmp, which uses per-device 1, i.e. a global
       batch of 16 on 8 GPUs; we use 32 for every DPO run). -> per-device 1, accumulation 32 / ngpu.

  make_cfg.py {sft,dpo} --name N --init MODEL --data DIR --out_dir DIR --ngpu G --cfg OUT.yaml [--max_steps S]
"""
import argparse, os, yaml

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["sft", "dpo"])
ap.add_argument("--name", required=True); ap.add_argument("--init", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--out_dir", required=True); ap.add_argument("--ngpu", type=int, required=True)
ap.add_argument("--cfg", required=True); ap.add_argument("--max_steps", type=int, default=-1)
a = ap.parse_args()
tsr = os.environ["TSR_DIR"]

if a.mode == "sft":
    c = yaml.safe_load(open(f"{tsr}/settings/sft/config/config.yaml"))
    assert c["per_device_train_batch_size"] * c["gradient_accumulation_steps"] * 8 == 32
    micro = 2
    c.update({"per_device_eval_batch_size": 2, "save_strategy": "no"})
else:
    c = yaml.safe_load(open(f"{tsr}/settings/sr/config/dpo/config_full_iter0.yaml"))
    assert c["per_device_train_batch_size"] * c["gradient_accumulation_steps"] * 8 == 32
    micro = 1
    c.update({"report_to": ["tensorboard"]})
if 32 % (micro * a.ngpu):
    raise SystemExit(f"global batch 32 is not divisible by {micro} x {a.ngpu} GPUs; use 1, 2, 4, 8, 16 or 32 GPUs")
c.update({"model_name_or_path": a.init, "dataset_mixer": {a.data: 1.0}, "output_dir": a.out_dir,
          "hub_model_id": a.name, "per_device_train_batch_size": micro,
          "gradient_accumulation_steps": 32 // (micro * a.ngpu), "max_steps": a.max_steps})
os.makedirs(os.path.dirname(a.cfg), exist_ok=True)
yaml.safe_dump(c, open(a.cfg, "w"), sort_keys=False)
print(a.cfg)
