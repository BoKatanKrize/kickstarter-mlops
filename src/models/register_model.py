import os
import logging

import wandb
from dotenv import find_dotenv, load_dotenv, set_key

from cli import gather_register_model
from utils.wandb import download_wandb_artifact, \
                        get_artifact_name,  \
                        select_best_models_from_sweep, \
                        download_best_models, \
                        promote_model_to_registry
from utils.io import load_data, save_pipe
from utils.aws_s3 import save_to_s3_bucket
from .train import prepare_data

from sklearn.metrics import roc_auc_score


def evaluate(X, y, models):
    """Evaluate trained models on the test set"""
    load_dotenv(find_dotenv())
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
    WANDB_ENTITY = os.environ["WANDB_ENTITY"]
    api = wandb.Api()
    best_auc = 0.0
    for key, model in models.items():
        y_proba = model.predict_proba(X['test'])
        roc_auc = roc_auc_score(y['test'], y_proba[:, 1])
        # Add the metric AUC (test) to the runs associated with best models
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{key}")
        run.summary["roc_auc_test"] = roc_auc
        run.summary.update()
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_key = key
            best_model = model
    return {'id': best_key, 'model': best_model, 'auc': best_auc}


def main(params):
    """Download the trained models from W&B, evaluate best models
       from Sweeps on the test set and record the most performant
       models in W&B Model Registry"""

    logger = logging.getLogger(__name__)

    logger.info(f'Downloading processed train/val/test data from W&B...')
    info_data = dict(params["info_data"])
    info_pipe = dict(params["info_pipe"])
    # Download dataset
    download_wandb_artifact(name_artifact=get_artifact_name(info_data['path_local_in']),
                            path_to_download=info_data['path_local_in'])

    logger.info(f'Preparing train/val/test data...')
    kicks_split_processed, year, month = load_data(info_data,
                                                   is_split=True)
    X, y = prepare_data(kicks_split_processed)

    logger.info(f'Downloading best models from W&B sweep...')
    # Adds the 'best' alias to the most performant models
    select_best_models_from_sweep(params['n_best'])
    best_models = download_best_models(info_pipe['path_local_in'])

    logger.info(f'Evaluating performance best models on test set...')
    best_model = evaluate(X, y, best_models)

    logger.info(f'Promoting best model to Model Registry...')
    promote_model_to_registry(best_model)
    # Store registered model name in .ENV
    set_key(find_dotenv(), "WANDB_REGISTERED_MODELS",
            f'model_{best_model["id"]}')

    logger.info(f'Saving Registered model locally...')
    info_pipe = save_pipe(best_model['model'],
                          info_pipe, suffix=best_model['id'])

    logger.info(f'Saving Registered model in S3 Bucket (LocalStack)...')
    save_to_s3_bucket(params["s3_bucket_name"],
                      info_pipe=info_pipe)


def wrapper_poetry():
    """ So that we can call this script using Poetry"""

    # -------------- #
    # Logging config #
    # -------------- #
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # ------------------------------ #
    # Load parameters from cli.py #
    # ------------------------------ #
    params = gather_register_model(standalone_mode=False)

    # --------------------------------------- #
    # Model prediction and W&B Model Registry #
    # --------------------------------------- #
    main(params)


if __name__ == '__main__':
    wrapper_poetry()