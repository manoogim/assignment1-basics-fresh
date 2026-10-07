from argparse import ArgumentParser
import os

from tests.nn_yaml import load_yaml_config, resolve_runs_folder

"""
Rename a checkpoint in runs folder to 'saved_ckpt.pt'.
This is appropriate for resuming training with option -rc, --resume_ckpt <full-path-to-saved_ckpt.pt>
"""
def main(args):
    _, cfg = load_yaml_config(args.config)
    runs_folder = resolve_runs_folder(cfg)
    old_name = os.path.join(runs_folder, args.ckpt)
    if not os.path.exists(old_name):
        raise Exception(f'Cannot rename - file does not exist {old_name}')
    new_name = os.path.join(runs_folder, 'saved_ckpt.pt')
    os.replace(old_name, new_name)
    print(f'use this training option: --resume_from {new_name}')

if __name__ == '__main__':
    parser = ArgumentParser(description="Rename a ckpt which is located in the standard wandb project runs folder.")
    parser.add_argument('-c', '--config', type=str, default='tests/config/abla_batch/cs336_owt.yaml', help='Path to the YAML configuration file which determines the relevant wandb project.')
    parser.add_argument('-cp', '--ckpt', type = str, default='last_ckpt.pt', help='Name of the checkpoint to rename.')
    parser.add_argument('-su', '--suffix', type=str, default='40mm', help='Last segment of runs folder name is token budget, ex: 40mm, 70mm, 327mm etc')
    args = parser.parse_args()
    main(args)