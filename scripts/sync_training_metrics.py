"""Copy closed node-local TensorBoard events to shared experiment storage."""
import os
from pathlib import Path
import shutil
import sys

import yaml


def sync(config_path,run):
    config_path=Path(config_path).resolve()
    run=Path(run).resolve()
    if not config_path.is_relative_to(run/'trial/configs'):
        raise ValueError('Config does not belong to this run')
    config=yaml.safe_load(config_path.read_text())
    local=Path(config['logging_dir']).resolve()
    allowed=Path(os.environ.get('TSR_TENSORBOARD_LOCAL_ROOT','/tmp/tsr-tensorboard')).resolve()
    if not local.is_relative_to(allowed):
        raise ValueError('TensorBoard output must be node-local')
    destination=run/'trial/metrics/tensorboard'/config_path.stem
    if local.is_dir():
        shutil.copytree(local,destination,dirs_exist_ok=True)
    return destination


if __name__=='__main__':
    print(sync(*sys.argv[1:]))
