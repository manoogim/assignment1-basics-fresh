import os

import wandb

def build_metadata(wand_run):
    summary = {key: value for key, value in wand_run.summary._as_dict().items() if key.startswith(('init','final','best')) }
    source = {'git sha': wand_run.settings.git_commit, 'git url': wand_run.settings.git_remote_url}
    run = {'run_id': wand_run.id, 'run_name': wand_run.name, 'run_project': wand_run.project, 'run_path': wand_run.path}
    metadata = {'run': run, 'source': source, 'summary' : summary}
    return metadata

def update_artifact_metadata(wand_run, rich_metadata):
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
    

if __name__ == '__main__':
    upload_dataset()

