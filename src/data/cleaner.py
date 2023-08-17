import wandb
import logging
import warnings

from cli import gather_cleaner
from utils.aws_s3 import save_to_s3_bucket, load_from_s3_bucket
from utils.pipelines import ColumnDropperTransformer, \
                            DropRowsWithSameIDTransformer, \
                            to_datetime_transformer, \
                            category_transformation, \
                            calculate_usd_goal, \
                            calculate_usd_pledged
from utils.io import load_data, save_data, save_pipe
from utils.wandb import init_wandb_run, log_wandb_artifact, \
                        get_artifact_name

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def split_train_val_test(df, test_size, seed):

    # The 'state' column shows the outcome of the project. We only keep
    # "failed" or "successful" projects
    df = df[df["state"].isin(["failed", "successful"])]

    # Stratify based on the 'state' column (target variable)
    target = 'state'
    df_train, df_test = train_test_split(df,
                                          test_size = test_size,
                                          stratify = df[target],
                                          random_state = seed)
    size_adj = test_size/(1.-test_size)
    df_train, df_val = train_test_split(df_train,
                                         test_size = size_adj,
                                         stratify = df_train[target],
                                         random_state = seed)
    df_split = {'train': df_train,
                'val': df_val,
                'test': df_test}
    return df_split


def create_cleaning_pipeline(columns_to_drop, id_column, date_columns):
    """Combine all the custom transformers into a single pipeline"""
    pipeline = Pipeline([
        ('column_dropper', ColumnDropperTransformer(columns_to_drop)),
        ('row_dropper', DropRowsWithSameIDTransformer(id_column)),
        ('date_transformer', FunctionTransformer(to_datetime_transformer,
                                                 kw_args={'columns': date_columns})),
        ('category_transformer', ColumnTransformer(
            transformers=[
                ('category_info', FunctionTransformer(category_transformation, validate=False), ['category']),
                ('drop_category', 'drop', ['category'])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
            ).set_output(transform="pandas")
        ),
        ('usd_conversion', ColumnTransformer(
            transformers=[
                ('usd_goal', FunctionTransformer(calculate_usd_goal, validate=False), ['goal', 'static_usd_rate']),
                ('usd_pledged', FunctionTransformer(calculate_usd_pledged, validate=False),
                 ['pledged', 'static_usd_rate']),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
            ).set_output(transform="pandas")
        )
    ])
    return pipeline


def apply_cleaning_pipeline(pipe, df_split):
    df_split_clean = dict()
    df_split_clean['train'] = pipe.fit_transform(df_split['train'])
    df_split_clean['val'] = pipe.transform(df_split['val'])
    df_split_clean['test'] = pipe.transform(df_split['test'])
    return df_split_clean, pipe


def main(params):
    """ Downloads the raw data from the S3 bucket and applies a 1st
        preprocessing pipeline to clean the data. The cleaned data
        is saved locally (~/data/interim) and in S3 bucket.
    """

    logger = logging.getLogger(__name__)

    # --------------------------------------------- #
    # Download raw data from S3 Bucket (LocalStack) #
    # --------------------------------------------- #
    logger.info(f'Downloading raw data from S3 Bucket (LocalStack)...')
    info_data = dict(params["info_data"])
    info_pipe = dict(params["info_pipe"])
    load_from_s3_bucket(params["s3_bucket_name"],
                        info_data=info_data,
                        info_pipe=None)

    # ------------------------- #
    # Load raw data into Pandas #
    # ------------------------- #
    logger.info(f'Loading raw data to Pandas...')
    kicks, year, month = load_data(info_data, is_split=False)

    # ------------- #
    # Split dataset #
    # ------------- #
    logger.info(f'Splitting raw data into train/val/test...')
    kicks_split = split_train_val_test(kicks['full'],
                                       params['test_size'],
                                       params['seed'])
    # ------------------------------- #
    # Clean raw data using a pipeline #
    # ------------------------------- #
    logger.info(f'Cleaning raw data...')

    # (1) columns with missing values
    cols_to_drop_missing = ['converted_pledged_amount', 'friends', 'is_backing',
                            'is_starred', 'location', 'permissions',
                            'usd_exchange_rate', 'usd_pledged', 'usd_type']
    # (2) irrelevant columns
    cols_to_drop_irr = ['country_displayable_name', 'creator', 'currency',
                        'currency_symbol', 'currency_trailing_code',
                        'current_currency', 'fx_rate', 'is_starrable',
                        'photo', 'profile', 'slug', 'source_url',  'spotlight',
                        'state_changed_at', 'urls', 'disable_communication']
    # (3) rows with same ID
    id_column = 'id'
    # (4) columns to convert to datetime
    date_columns = ['created_at', 'deadline', 'launched_at']

    # Create pipeline:
    #   - column_dropper (1) (2)
    #   - row_dropper (3)
    #   - date_transformer (4)
    #   - category_transformer
    #   - usd_conversion
    full_pipeline = create_cleaning_pipeline(cols_to_drop_missing + \
                                             cols_to_drop_irr, id_column,
                                             date_columns)
    # Apply pipeline to train/val/test data
    kicks_split_clean, full_pipeline = apply_cleaning_pipeline(full_pipeline,
                                                               kicks_split)

    # ---------------------------------------------- #
    # Save clean train/val/test and pipeline locally #
    # ---------------------------------------------- #
    logger.info(f'Saving clean data and pipeline locally...')
    info_data = save_data(kicks_split_clean,
                          info_data,
                          year, month,
                          is_split=True)
    info_pipe = save_pipe(full_pipeline, info_pipe)

    # ------------------------------------------------------ #
    # Save clean data and pipeline in S3 Bucket (LocalStack) #
    # ------------------------------------------------------ #
    logger.info(f'Saving clean data and pipeline in a S3 Bucket (LocalStack)...')
    save_to_s3_bucket(params["s3_bucket_name"],
                      info_data=info_data,
                      info_pipe=info_pipe)

    # ------------------------------------------------ #
    # Log the cleaned data & pipeline as W&B artifacts #
    # ------------------------------------------------ #
    logger.info(f'Logging the cleaned data & pipeline to Weights & Biases...')
    run = init_wandb_run(name_script='cleaner',
                         job_type='preprocessing')
    # Data
    log_wandb_artifact(run,
                       name_artifact=get_artifact_name(info_data['path_local_out']),
                       type_artifact='dataset',
                       bucket_name=params["s3_bucket_name"],
                       path_to_log=info_data["path_s3_out"])
    # Pipeline
    log_wandb_artifact(run,
                       name_artifact=get_artifact_name(info_pipe['path_local_out']),
                       type_artifact='model',
                       bucket_name=params["s3_bucket_name"],
                       path_to_log=info_pipe["path_s3_out"])

    wandb.finish()


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
    params = gather_cleaner(standalone_mode=False)

    # ---------------------------------------------- #
    # Clean and save data (locally and S3 bucket)    #
    # ---------------------------------------------- #
    warnings.filterwarnings('ignore',category=UserWarning)
    main(params)


if __name__ == '__main__':
    wrapper_poetry()
