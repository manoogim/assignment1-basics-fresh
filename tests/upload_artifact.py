from argparse import ArgumentParser
import os

import wandb

from tests.nn_yaml import load_yaml_config, resolve_runs_folder, resolve_wandb_name

def build_metadata(wand_run):
    summary = {key: value for key, value in wand_run.summary._as_dict().items() if key.startswith(('init','final','best')) }
    source = {'git sha': wand_run.settings.git_commit, 'git url': wand_run.settings.git_remote_url}
    run = {'run_id': wand_run.id, 'run_name': wand_run.name, 'run_project': wand_run.project, 'run_path': wand_run.path}
    metadata = {'run': run, 'source': source, 'summary' : summary}
    return metadata

def zzupdate_artifact_metadata(wand_run, rich_metadata):
    artifact = wand_run.Api().artifact("manoogim-personal/abla_batch_367/best_ckpt:v0")
    artifact.metadata = rich_metadata
    artifact.description = "Best checkpoint selected by minimum validation loss."
    artifact.save()

def upload_artifact(wand_run,  wandb_artifact_name, local_artifact_path, metadata={}, artifact_type = 'model' ):
    artifact = wandb.Artifact(name=wandb_artifact_name, type = artifact_type, metadata=metadata)
    base_name = os.path.basename(local_artifact_path)
    artifact.add_file(local_artifact_path, name=base_name)
    wand_run.log_artifact(artifact)
    try:
        artifact.wait() # wait without short timeout
        print(f'Uploaded artifact: {wandb_artifact_name}')
    except Exception as ex:
        print(f'Error while waiting to upload weights to wandb. {ex}')


def upload_dataset():
    entity="manoogim-personal"
    wandb_project="abla_batch_367"
    local_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\out\tinystories_GPT4'
    name = 'tinystories_tokens'

    with wandb.init(entity=entity, project=wandb_project, job_type="prepare-dataset") as run:
        dataset = wandb.Artifact(name, type="dataset")
        dataset.add_dir(local_path)
        run.log_artifact(dataset)

def upload_best_ckpt():
    entity="manoogim-personal"
    wandb_project="abla_batch_367"
    artifact_type = 'model'
    wandb_artifact_name = 'best_ckpt'
    local_artifact_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\runs\zz_327mm_b136\best_ckpt.pt'
    metadata={'path': local_artifact_path}

    with wandb.init(entity=entity, project=wandb_project) as wandb_run:
        upload_artifact(wandb_run, wandb_artifact_name,  metadata, local_artifact_path, artifact_type=artifact_type)

def upload_artifact_manually(wandb_project, wandb_artifact_name, local_artifact_path, metadata={}):
    entity="manoogim-personal"
    artifact_type = 'model'
 
    with wandb.init(entity=entity, project=wandb_project) as wandb_run:
        upload_artifact(wandb_run, wandb_artifact_name, local_artifact_path, metadata, artifact_type=artifact_type)

# def upload_artifact_from_args(args, metadata = {}) :
#     _, cfg = load_yaml_config(args.config)
#     wandb_project = resolve_wandb_name(cfg)
#     runs_folder = resolve_runs_folder(cfg, args.suffix)
#     local_artifact_path = os.path.join(runs_folder, args.ckpt)
#     wandb_artifact_name = args.ckpt
#     upload_artifact_manually(wandb_project, wandb_artifact_name, local_artifact_path, metadata)


def download_artifact(wandb_project, artifact_name):
    run = wandb.init( entity="manoogim-personal", job_type="manual download")

    artifact = run.use_artifact( f"manoogim-personal/{wandb_project}/{artifact_name}", type="model",)
    checkpoint_path = artifact.download() 
    print(f'Downloaded {checkpoint_path}')
    return checkpoint_path

if __name__ == '__main__':
    desc = 'Upload a ckpt which is located in the standard runs folder of wandb project.'
    epilog = (
        'Example:\n\n'
        'python tests/upload_artifact.py -c tests/config/gpt2_tiny.yaml -cp saved_ckpt.pt -su327mm'
    )
    parser = ArgumentParser(description=desc, epilog = epilog)
    parser.add_argument('-c', '--config', type=str, default='tests/config/gpt2_tiny.yaml', help='Path to the YAML configuration file which determines the relevant wandb project.')
    parser.add_argument('-cp', '--ckpt', type = str, default='ckpt_a.pt', help='Name of the checkpoint to upload.')
    parser.add_argument('-su', '--suffix', type=str, default='aruns/owt_learning29_128_ga32_lr0_0055_0mm_b128/last_ckpt.pt', help='Last segment of runs folder to correctly reconstruct path to runs folder, must be name of the token budget, ex: 40mm, 70mm, 327mm etc')
    args = parser.parse_args()

    # upload_artifact_manually('owt_learning','last_ckpt.pt',args.suffix)
    cpp = download_artifact('owt_learning', 'last_649:v0')
    # print(f'Downloaded artifact: {cpp}')

