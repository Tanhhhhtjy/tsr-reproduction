"""Freeze one native SFT/DPO config for the best TSR profile."""
import argparse
import json
import hashlib
import os
from pathlib import Path

import yaml

PREFIX = 'Llama-3.1-8B-mix-sft-epoch3'
NAMES = {'M0': PREFIX, 'Mf0': PREFIX+'-temporal-iterjudge-iter0_tmp',
         'M1': PREFIX+'-temporal-iterjudge-iter0', 'Mf1': PREFIX+'-temporal-iterjudge-iter1_tmp',
         'M2': PREFIX+'-temporal-iterjudge-iter1'}
TEMPLATES = {'config': ('M0', None, None), 'config_full_iter0_tmp': ('Mf0','M0','iter0_tmp'),
             'config_full_iter0': ('M1','M0','iter0'), 'config_full_iter1_tmp': ('Mf1','M1','iter1_tmp'),
             'config_full_iter1': ('M2','M1','iter1')}


def configure(mode, template, run, base, gpus):
    template = Path(template).resolve()
    run = Path(run).resolve()
    root = Path(__file__).resolve().parents[1]
    stage, parent, dataset = TEMPLATES[template.stem]
    if (mode == 'sft') != (stage == 'M0'):
        raise ValueError('Training mode/template mismatch')
    config = yaml.safe_load(template.read_text())
    # Fail even if a future edit changes the stock template to initialize M2/Mf1 from Mf0/Mf1.
    if stage in ('Mf1','M2') and Path(config['model_name_or_path']).name != NAMES['M1']:
        raise ValueError(f'{stage} must initialize from M1, never from a future model')
    micro = 2 if stage == 'M0' else 4
    if not 1 <= gpus <= 8 or 32 % (gpus * micro):
        raise ValueError('GPU/micro-batch combination cannot preserve global batch 32')
    work = run / 'trial'
    output = work / 'models' / NAMES[stage]
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to reuse or overwrite an existing model; use a fresh run')
    init = base if parent is None else str(work / 'models' / NAMES[parent])
    data = root / 'data/sft_data' if dataset is None else work / 'data' / dataset
    config.update(model_name_or_path=init, dataset_mixer={str(data):1.0}, output_dir=str(output),
                  hub_model_id=stage, per_device_train_batch_size=micro,
                  gradient_accumulation_steps=32//(micro*gpus), seed=42, data_seed=42,
                  torch_dtype='bfloat16', attn_implementation='flash_attention_2',
                  save_strategy='no', save_only_model=True, save_safetensors=True,
                  logging_nan_inf_filter=False, push_to_hub=False, max_steps=-1)
    local_root = Path(os.environ.get('TSR_TENSORBOARD_LOCAL_ROOT','/tmp/tsr-tensorboard'))
    run_key = hashlib.sha256(str(run).encode()).hexdigest()[:12]
    config['logging_dir'] = str(local_root / run_key / stage)
    if stage == 'M0':
        config.update(num_train_epochs=3, learning_rate=2e-6, per_device_eval_batch_size=2,
                      gradient_checkpointing=True)
    else:
        config.update(num_train_epochs=1, learning_rate=5e-7, beta=0.1, do_eval=False,
                      gradient_checkpointing=False, optim='adamw_torch',
                      dataloader_drop_last=True, logging_steps=1, report_to=['tensorboard'])
    path = work / 'configs' / (stage+'.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    path.with_suffix('.initialization.json').write_text(json.dumps({
        'stage':stage,'initialization_stage':'Base' if parent is None else parent,
        'model_name_or_path':init,'output_dir':str(output),'native_template':str(template),
        'global_batch':32,'gpus':gpus,'micro_batch':micro,'accumulation':32//(micro*gpus)},indent=2)+'\n')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['sft','dpo'])
    parser.add_argument('template', type=Path)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--gpus', type=int, required=True)
    args = parser.parse_args()
    print(configure(args.mode,args.template,args.run_dir,args.base_model,args.gpus))
