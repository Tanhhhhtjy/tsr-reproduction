"""Freeze this official checkout and prepare SN5 training/scoring Jobs; never submit."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'pytorch/pytorch@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            hasher.update(chunk)
    return hasher.hexdigest()



def bundle_files(root,tracked):
    root=Path(root).resolve()
    files=set(filter(None,tracked))
    manifest=json.loads((root/'data/manifest.json').read_text())
    for rel,expected in manifest['files'].items():
        path=root/rel
        if not path.resolve().is_relative_to(root/'data'):
            raise ValueError('Unsafe dataset path')
        if not path.is_file() or digest(path)!=expected['sha256']:
            raise ValueError('Dataset checksum differs or file is missing: '+rel+'; run scripts/download_data.py')
        files.add(rel)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--base-model',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--credentials',type=Path,help='Private mode-0600 judge JSON; never copied into bundle')
    parser.add_argument('--python',default=sys.executable)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,42}',args.run_name):
        parser.error('Invalid run name')
    run = ROOT/'work/runs'/args.run_name
    if run.exists():
        parser.error('Run exists; refusing reuse')
    base = args.base_model.resolve()
    if not (base/'config.json').is_file():
        parser.error('Base model directory is incomplete')
    reference = json.loads(args.reference.read_text())
    if len(reference)!=805 or len({r['instruction'] for r in reference})!=805:
        parser.error('Reference must contain exactly 805 unique questions')
    if args.credentials and args.credentials.stat().st_mode & 0o077:
        parser.error('Credential file must be owner-readable only')
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    # Require committed source so runtime code and published patch have a stable identity.
    dirty = subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if dirty.strip():
        parser.error('Commit source before freezing a formal run')
    tracked = bundle_files(ROOT,tracked)
    bundle = run/'bundle'
    for rel in filter(None,tracked):
        src,dst = ROOT/rel,bundle/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    (run/'inputs').mkdir()
    shutil.copy2(args.reference,run/'inputs/alpaca_reference.json')
    dump(run/'source-sha256.json',{rel:digest(bundle/rel) for rel in filter(None,tracked)})
    dump(run/'base-model-manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)}
         for p in sorted(base.iterdir()) if p.is_file() and (p.suffix in ('.json','.safetensors') or p.name=='tokenizer.model')})
    ds = {'train_batch_size':'auto','train_micro_batch_size_per_gpu':'auto',
          'gradient_accumulation_steps':'auto','gradient_clipping':'auto','bf16':{'enabled':'auto'},
          'zero_optimization':{'stage':2,'overlap_comm':False,'contiguous_gradients':True,
                               'reduce_bucket_size':50000000,'allgather_bucket_size':50000000},
          'steps_per_print':1000000,'wall_clock_breakdown':False}
    dump(run/'deepspeed.json',ds)
    accelerate = yaml.safe_load((ROOT/'settings/tsr/config/deepspeed_zero3.yaml').read_text())
    accelerate['deepspeed_config']={'deepspeed_config_file':str(run/'deepspeed.json'),'zero3_init_flag':False}
    accelerate.pop('mixed_precision',None)
    (run/'accelerate.yaml').write_text(yaml.safe_dump(accelerate,sort_keys=False))
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dump(run/'run-plan.json',{'run_dir':str(run),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit':commit,'official_upstream':'a9a379d','pipeline':'native official SFT, inference, formatters and DPO',
        'gpus':8,'node':'any-sn5','base_directory':str(base),'global_batch':32,'seed':42,
        'initialization':{'M0':'Base','Mf0':'M0','M1':'M0','Mf1':'M1','M2':'M1'},
        'pair_policy':'future_tie_charcap_distinct','max_chosen_characters':2500,
        'strict_gap_all_stages':True,'drop_invalid_scores':True,'distinct_stripped_text':True,
        'source_priorities':{'chosen_tie':'future','rejected_tie':'current'},
        'generation_temperature':1,'self_judge_temperature':0,
        'eval':{'judge':'gpt-4o-2024-08-06','reference':'gpt4_1106_preview',
                'model_outputs':['M2_future_tie_charcap_distinct'],'generation_temperature':0,'questions':805},
        'retention':'scalar metrics/configs/canonical pairs/final stage models/evaluation answers; delete consumed raw caches; no optimizer checkpoints'})
    mount=[{'name':'fs','mountPath':'/mnt/sn-007'},{'name':'shm','mountPath':'/dev/shm'}]
    volumes=[{'name':'fs','persistentVolumeClaim':{'claimName':'youtu-sn2-007'}},
             {'name':'shm','emptyDir':{'medium':'Memory','sizeLimit':'64Gi'}}]
    env=[{'name':'TSR_PYTHON','value':str(Path(args.python).absolute())},
         {'name':'SFT_BASE_MODEL','value':str(base)}]
    train={'apiVersion':'batch/v1','kind':'Job','metadata':{'name':'zaninzeng-'+args.run_name,
         'namespace':'sn5-system-intern','labels':{'owner':'zaninzeng','app':'tsr-official-best'}},
        'spec':{'backoffLimit':0,'activeDeadlineSeconds':172800,'ttlSecondsAfterFinished':604800,
         'template':{'metadata':{'labels':{'owner':'zaninzeng','app':'tsr-official-best','run':args.run_name}},
          'spec':{'restartPolicy':'Never','automountServiceAccountToken':False,
           'affinity':{'nodeAffinity':{'requiredDuringSchedulingIgnoredDuringExecution':{'nodeSelectorTerms':[
               {'matchExpressions':[{'key':'group','operator':'In','values':['shennong-5','shennong-5-dev']},
                 {'key':'node.kubernetes.io/instance-type','operator':'In','values':['ml.p5e.48xlarge','ml.p5en.48xlarge']}]}]}}},
           'tolerations':[{'key':key,'operator':'Exists','effect':'NoSchedule'} for key in ['nvidia.com/gpu','role','glm5sft']],
           'containers':[{'name':'train','image':IMAGE,'command':['bash',str(bundle/'scripts/sn5_best_bootstrap.sh'),str(run)],
               'env':env,'resources':{'requests':{'cpu':'32','memory':'512Gi','nvidia.com/gpu':'8'},
                 'limits':{'cpu':'128','memory':'1536Gi','nvidia.com/gpu':'8'}},'volumeMounts':mount}],
           'volumes':volumes}}}}
    dump(run/'job.json',train)
    if args.credentials:
        score=json.loads(json.dumps(train))
        score['metadata']['name']='zaninzeng-'+args.run_name+'-score'
        spec=score['spec']['template']['spec'];spec.pop('affinity');spec.pop('tolerations')
        container=spec['containers'][0]
        container.update(name='score',resources={'requests':{'cpu':'2','memory':'8Gi'},'limits':{'cpu':'4','memory':'16Gi'}},
            command=[str(Path(args.python).absolute()),str(bundle/'scripts/score_when_ready.py'),
                     '--run',str(run),'--credentials',str(args.credentials.resolve()),'--api-concurrency','16'])
        container['env']=[{'name':'PYTHONDONTWRITEBYTECODE','value':'1'}]
        dump(run/'score-job.json',score)
    print(run)


if __name__=='__main__':
    main()
