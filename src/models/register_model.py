import logging

import wandb

from config import gather_register_model
from utils.wandb import init_wandb_run, download_wandb_artifact, \
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
    best_auc = 0.0
    for key, model in models.items():
        y_proba = model.predict_proba(X['test'])
        roc_auc = roc_auc_score(y['test'], y_proba[:, 1])
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
    # Initialize W&B run
    run = init_wandb_run(name_script='register_model',
                         job_type='registry',
                         id_run='5')
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

    wandb.finish()

    logger.info(f'Saving best model locally...')
    info_pipe = save_pipe(best_model['model'],
                          info_pipe, suffix=best_model['id'])

    logger.info(f'Saving best model in S3 Bucket (LocalStack)...')
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
    # Load parameters from config.py #
    # ------------------------------ #
    params = gather_register_model(standalone_mode=False)

    # --------------------------------------- #
    # Model prediction and W&B Model Registry #
    # --------------------------------------- #
    main(params)


if __name__ == '__main__':
    wrapper_poetry()