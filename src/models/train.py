import yaml
import wandb
import logging
from functools import partial
import dotenv

from cli import gather_train
from utils.wandb import init_wandb_run, download_wandb_artifact, \
                        configure_sweep, \
                        run_sweep, log_wandb_artifact, \
                        get_artifact_name
from utils.io import load_data, save_pipe
from utils.aws_s3 import save_to_s3_bucket

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, log_evaluation

from sklearn.metrics import f1_score, precision_score, \
                            recall_score, roc_auc_score

from wandb.xgboost import WandbCallback
from wandb.lightgbm import wandb_callback

counter = 0


def prepare_data(df_split):
    """Extract the target column and divide X,y"""
    target = 'state'
    keys = ['train', 'val', 'test']
    y = {key: df_split[key][target] for key in keys}
    X = {key: df_split[key].drop(target, axis=1) for key in keys}
    return X, y


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        sweep_config = yaml.safe_load(file)
    return sweep_config


def train_single_sweep(X, y, info_pipe, s3_bucket, seed):

    global counter
    with init_wandb_run(name_script='sweep', job_type='training', group='sweeps') as run:

        counter += 1
        run.name = f'{run.name}-{run.id}-{counter}'
        cfg = run.config

        if cfg['model_name'] == 'xgboost':
            params = {
                'objective': 'binary:logistic',
                'learning_rate': cfg['learning_rate'],
                'max_depth': cfg['max_depth'],
                'n_estimators': cfg['n_estimators'],
                'gamma': cfg['gamma'],
                'colsample_bytree': cfg['colsample_bytree'],
                'subsample': cfg['subsample'],
                'subsample_freq': cfg['subsample_freq'],
                'early_stopping_rounds': 40,
                'seed': seed
            }
            model = XGBClassifier(**params)
            callbacks = [WandbCallback()]

        elif cfg['model_name'] == 'lightgbm':
            params = {
                'objective': 'binary',
                'num_leaves': cfg['num_leaves'],       # <-- Only LightGBM
                'learning_rate': cfg['learning_rate'],
                'max_depth': cfg['max_depth'],
                'num_iterations': cfg['n_estimators'],        # <-- != naming
                'min_gain_to_split': cfg['gamma'],            # <-- != naming
                'feature_fraction': cfg['colsample_bytree'],  # <-- != naming
                'bagging_fraction': cfg['subsample'],         # <-- != naming
                'bagging_freq': cfg['subsample_freq'],        # <-- != naming
                'early_stopping_rounds': 40,
                'seed': seed,
                'verbosity': -1,
            }
            model = LGBMClassifier(**params)
            callbacks = [wandb_callback(), log_evaluation()]

        eval_set = [(X['train'], y['train']), (X['val'], y['val'])]
        model.fit(X['train'], y['train'],
                  eval_set=eval_set,
                  eval_metric=['logloss', 'auc'], # <-- AUC is used for early stopping
                  callbacks=callbacks)

        keys = ['train','val']
        # predicted value
        y_pred = {key: model.predict(X[key]) for key in keys}
        # predicted probability
        y_proba = {key: model.predict_proba(X[key]) for key in keys}

        f1 = {key: f1_score(y[key], y_pred[key]) for key in keys}
        precision = {key: precision_score(y[key], y_pred[key]) for key in keys}
        recall = {key: recall_score(y[key], y_pred[key]) for key in keys}
        roc_auc = {key: roc_auc_score(y[key], y_proba[key][:,1]) for key in keys}

        wandb.log({
            "train": {
                'f1_score': f1['train'],
                'precision': precision['train'],
                'recall': recall['train'],
                'roc_auc': roc_auc['train']
                     },
            "val": {
                'f1_score': f1['val'],
                'precision': precision['val'],
                'recall': recall['val'],
                'roc_auc': roc_auc['val']
            },
        })

        # Save model from current sweep locally
        info_pipe = save_pipe(model, info_pipe, suffix=run.id)

        # Save current model to S3 Bucket
        model_name = f'{info_pipe["prefix_name"]}_{run.id}'
        info_tmp = {'fnames': [f'{model_name}.pkl'],
                    'path_local_out': info_pipe["path_local_out"],
                    'path_s3_out': info_pipe["path_s3_out"]
                    }
        save_to_s3_bucket(s3_bucket,
                          info_pipe=info_tmp)

        # Save the model to W&B
        log_wandb_artifact(run,
                           name_artifact=model_name,
                           type_artifact='model',
                           bucket_name=s3_bucket,
                           path_to_log=info_pipe["path_s3_out"],
                           name_file=f'{model_name}.pkl')

        run.finish()


def main(params):
    """Download the processed train/val/set from W&B and perform
    hyperparameter optimization evaluating XGBoost and LightGBM
    (Sweeps) and log the trained models to W&B"""

    logger = logging.getLogger(__name__)

    logger.info(f'Downloading processed train/val/test data from W&B...')
    info_data = dict(params["info_data"])
    info_pipe = dict(params["info_pipe"])

    download_wandb_artifact(name_artifact=get_artifact_name(info_data['path_local_in']),
                            path_to_download=info_data['path_local_in'])

    logger.info(f'Loading train/val/test data into Pandas...')
    kicks_split_processed, year, month = load_data(info_data,
                                                   is_split=True)

    logger.info(f'Dividing train/val/test data into features (X) and target (y)...')
    X, y = prepare_data(kicks_split_processed)

    logger.info(f'Defining W&B Sweep Configuration...')
    SWEEP_CONFIG = load_yaml(params['sweep_config'])
    sweep_id = configure_sweep(SWEEP_CONFIG)

    logger.info(f'Running W&B Sweeps (comparing XGBoost vs LightGBM)...')
    target_function = partial(train_single_sweep,
                              X=X, y=y, info_pipe=info_pipe,
                              s3_bucket = params["s3_bucket_name"],
                              seed=params['seed'])
    run_sweep(sweep_id, target_function=target_function, n_sweeps=params['n_sweeps'])

    # Store sweep_id in .ENV to be used later (no easy way to access it otherwise)
    dotenv.set_key(dotenv.find_dotenv(), "WANDB_SWEEP_ID", sweep_id)


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
    params = gather_train(standalone_mode=False)

    # ------------ #
    # Train models #
    # ------------ #
    main(params)


if __name__ == '__main__':
    wrapper_poetry()