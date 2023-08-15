import os
import wandb
import joblib
from dotenv import find_dotenv, load_dotenv


def init_wandb_run(name_script, job_type, group=None, id_run=None):
    """Setting up Weights & Biases for Tracking and Registry"""
    load_dotenv(find_dotenv())
    WANDB_API_KEY = os.environ["WANDB_API_KEY"]
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT,
                     entity=WANDB_ENTITY,
                     group=group,
                     name=name_script,
                     job_type=job_type,
                     id=id_run)
    return run


def log_wandb_artifact(run, name_artifact, type_artifact,
                       bucket_name=None, file_name=None, path_to_log=None):
    """Log an object stored locally or in S3 bucket as W&B artifact (only metadata)"""
    load_dotenv(find_dotenv())  # Load from .env -> AWS_ENDPOINT_URL
    artifact = wandb.Artifact(name=name_artifact, type=type_artifact)
    if bucket_name is not None:  # S3 bucket
        artifact.add_reference(f's3://{bucket_name}/{path_to_log}') # folder location
    else:                        # Local
        artifact.add_file(f'{path_to_log}/{file_name}')
    run.log_artifact(artifact)


def download_wandb_artifact(name_artifact, path_to_download):
    """Download (locally) and use an artifact stored on W&B"""
    load_dotenv(find_dotenv())
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
    api = wandb.Api()
    artifact = api.artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{name_artifact}:latest')
    artifact.download(root=path_to_download)


def configure_sweep(search_space):
    load_dotenv(find_dotenv())
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
    sweep_id = wandb.sweep(search_space,
                           entity=WANDB_ENTITY,
                           project=WANDB_PROJECT)
    return sweep_id


def run_sweep(sweep_id, target_function=None, n_sweeps=None):
    load_dotenv(find_dotenv())
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
    wandb.agent(sweep_id,
                function=target_function,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                count=n_sweeps)


def get_artifact_name(path):
    """Get the W&B Artifact name from local path
    E.g.: /data/processed -> processed-data"""
    path_parts = path.split('/')
    artifact_name = '-'.join(path_parts[-2:][::-1])
    return artifact_name


def select_best_models_from_sweep(n_best):

    load_dotenv(find_dotenv())
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
    WANDB_SWEEP_ID = os.environ["WANDB_SWEEP_ID"]
    api = wandb.Api()
    sweep = api.sweep(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_SWEEP_ID}')
    runs = sweep.runs

    # Create a list to store models and their attributes
    model_list = []
    # Iterate over runs and collect model information
    for run in runs:
        model_info = {
            "run_id": run.id,
            "roc_auc": run.history()['val.roc_auc'].iloc[-1]  # Get the last recorded AUC
        }
        model_list.append(model_info)

    # Sort the model_list based on the best validation AUC
    sorted_models = sorted(model_list,
                           key=lambda x: x["roc_auc"], reverse=True)
    # Add the alias 'best' to the 'n_best' first models
    for m in sorted_models[:n_best]:
        model_artifact = api.artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/model_{m["run_id"]}:latest')
        model_artifact.aliases.append('best')
        model_artifact.save()


def download_best_models(path):

    load_dotenv(find_dotenv())
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
    WANDB_SWEEP_ID = os.environ["WANDB_SWEEP_ID"]
    api = wandb.Api()
    sweep = api.sweep(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_SWEEP_ID}')
    runs = sweep.runs

    # Iterate over runs and download best models
    best_models = {}
    for run in runs:
        model_artifact = api.artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/model_{run.id}:latest')
        if 'best' in model_artifact.aliases:
            model_artifact.download(root=path)
            best_models[run.id] = joblib.load(f'{path}/model_{run.id}.pkl')
    return best_models


def promote_model_to_registry(run, model):

    load_dotenv(find_dotenv())
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
    model_artifact = run.use_artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/model_{model["id"]}:latest')
    model_artifact.link(f'{WANDB_ENTITY}/model-registry/model_{model["id"]}',
                        aliases=['staging'])
    wandb.log({'roc_auc': model['auc']})