"""Validate native best-profile artifacts and retain compact metrics."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from prepare_training_config import NAMES


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def pairs(path):
    stats = json.loads((path/'stats.json').read_text())
    if not (stats['n_prompts'] == 5000 and stats['n_pairs'] > 0 and stats['gap_min'] > 0
            and stats['retained_tie_pairs'] == stats['identical_pair'] == stats['chosen_over_character_limit_pairs'] == 0
            and stats['max_chosen_characters'] == 2500 and stats['drop_invalid_scores']):
        raise ValueError('Preference dataset rules/counts differ from the best profile')
    count = 0
    for line in (path/'train.json').open():
        row = json.loads(line)
        chosen,rejected = row['chosen'][-1]['content'],row['rejected'][-1]['content']
        if len(chosen)>2500 or chosen.strip()==rejected.strip():
            raise ValueError('Invalid canonical preference text')
        count += 1
    if count != stats['n_pairs']:
        raise ValueError('Canonical preference row count differs')


def model(run, stage):
    path = run/'trial/models'/NAMES[stage]
    state = json.loads((path/'trainer_state.json').read_text())
    if state['global_step'] <= 0:
        raise ValueError('No optimizer updates')
    for row in state.get('log_history',[]):
        if any(isinstance(value,float) and not math.isfinite(value) for value in row.values()):
            raise ValueError('Non-finite training metric')
    indices = list(path.glob('*.index.json'))
    weights = set()
    for index in indices:
        weights.update(path/name for name in json.loads(index.read_text()).get('weight_map',{}).values())
    if not weights:
        weights = set(path.glob('model.safetensors'))
    if not weights or any(not p.is_file() or p.stat().st_size==0 for p in weights):
        raise ValueError('Missing final native weights')
    save(run/'trial/metrics'/f'{stage}.json',dict(stage=stage,global_step=state['global_step'],
         epoch=state.get('epoch'),log_history=state.get('log_history',[]),weight_bytes=sum(p.stat().st_size for p in weights)))


def cleanup(run, phase):
    if phase not in ('iter0','iter1'):
        raise ValueError('Unexpected cleanup phase')
    index = int(phase[-1])
    for stage in (f'Mf{index}', f'M{index+1}'):
        if not (run/'trial/metrics'/f'{stage}.json').is_file():
            raise ValueError('Refusing to remove caches before completed training')
    directory = run/'trial/candidates'/phase
    files = sorted(directory.glob('*.json'))
    manifest = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Unexpected raw cache entry')
        manifest.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(run/'trial/metrics'/f'{phase}-consumed-caches.json',manifest)
    for path in files:
        path.unlink()


if __name__=='__main__':
    command,path,*extra = sys.argv[1:]
    path = Path(path).resolve()
    if command=='status':
        save(path/'status.json',dict(phase=extra[0],updated_unix=time.time()))
    elif command=='failure':
        save(path/'status.json',dict(phase='failed',updated_unix=time.time()))
        save(path/'FAILED.json',dict(failed=True,updated_unix=time.time()))
    elif command=='pairs':
        pairs(path)
    elif command=='model':
        model(path,extra[0])
    elif command=='cleanup':
        cleanup(path,extra[0])
    else:
        raise SystemExit('Unknown validation command')
