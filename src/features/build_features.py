import wandb
import logging
import warnings
import dotenv

from cli import gather_build_features
from utils.aws_s3 import save_to_s3_bucket, load_from_s3_bucket
from utils.io import load_data, save_data, save_pipe
from utils.pipelines import calculate_name_length, \
                            calculate_description_length, \
                            calculate_creation_to_launch_hours, \
                            calculate_campaign_hours, \
                            MedianDiffCalculatorTransformer, \
                            ColumnDropperTransformer, \
                            turn_to_log
from utils.wandb import init_wandb_run, log_wandb_artifact, \
                         get_artifact_name

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, \
                                  StandardScaler, OneHotEncoder, \
                                  OrdinalEncoder


def create_feat_eng_pipeline(cols_to_drop, cols_to_log,
                             cols_to_scale_encode):
    """Combine all the custom transformers into a single pipeline"""

    preprocessor_sentence_length = ColumnTransformer(
            transformers=[
                ('name_length', FunctionTransformer(calculate_name_length,
                                                    validate=False), ['name']),
                ('description_length', FunctionTransformer(calculate_description_length,
                                                           validate=False), ['blurb']),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
    ).set_output(transform="pandas")

    preprocessor_time_duration =  ColumnTransformer(
            transformers=[
                ('creation_to_launch_hours', FunctionTransformer(calculate_creation_to_launch_hours,
                                                                 validate=False), ['created_at', 'launched_at']),
                ('campaign_hours', FunctionTransformer(calculate_campaign_hours,
                                                       validate=False), ['launched_at', 'deadline']),
                ('passthrough', 'passthrough', ['created_at', 'launched_at', 'deadline'])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
    ).set_output(transform="pandas")

    preprocessor_log_converter = ColumnTransformer(
            transformers=[
                ('log_scaled', FunctionTransformer(func=turn_to_log), cols_to_log['yes']),
                ('remaining', 'passthrough', cols_to_log['no'])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
    ).set_output(transform="pandas")

    preprocessor_scale_and_encode = ColumnTransformer(
            transformers=[
                ("categorical_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                                      cols_to_scale_encode['cat']),
                ("numerical_scaler", StandardScaler(), cols_to_scale_encode['num']),
                ("target_encoder", OrdinalEncoder(categories=cols_to_scale_encode['target_mapping']),
                                                  cols_to_scale_encode['target'])
            ],
            # All cols are processed, so no need for 'passthrough'
            verbose_feature_names_out=False
    ).set_output(transform="pandas")

    pipeline = Pipeline([
        ('sentence_length', preprocessor_sentence_length),
        ('time_duration', preprocessor_time_duration),
        ('median_diff', MedianDiffCalculatorTransformer()),
        ('passthrough', 'passthrough'),
        ('column_dropper', ColumnDropperTransformer(cols_to_drop)),
        ('log_converter', preprocessor_log_converter),
        ('scale_and_encode', preprocessor_scale_and_encode),
    ])

    return pipeline


def apply_feat_eng_pipeline(pipe, df_split_clean):
    df_split_processed = dict()
    df_split_processed['train'] = pipe.fit_transform(df_split_clean['train'])
    df_split_processed['val'] = pipe.transform(df_split_clean['val'])
    df_split_processed['test'] = pipe.transform(df_split_clean['test'])
    return df_split_processed, pipe


def main(params):
    """ Downloads the cleaned data from the S3 bucket and applies a 2nd
        preprocessing pipeline to augment the data (feature engineering).
        The augmented data is saved locally (~/data/processed) and in S3
        bucket
    """

    logger = logging.getLogger(__name__)

    # ------------------------------------------------- #
    # Download cleaned data from S3 Bucket (LocalStack) #
    # ------------------------------------------------- #
    logger.info(f'Downloading cleaned data from S3 Bucket (LocalStack)...')
    info_data = dict(params["info_data"])
    info_pipe = dict(params["info_pipe"])
    load_from_s3_bucket(params["s3_bucket_name"],
                        info_data=info_data,
                        info_pipe=None)

    # ----------------------------- #
    # Load cleaned data into Pandas #
    # ----------------------------- #
    logger.info(f'Loading cleaned data into Pandas...')
    kicks_split_clean, year, month = load_data(info_data, is_split=True)

    # ------------------------------------ #
    # Feat. Eng. the data using a pipeline #
    # ------------------------------------ #
    logger.info(f'Performing feature engineering on the data...')

    # (1) These columns are no longer necessary after feat. eng.
    cols_to_drop = ['backers_count', 'created_at', 'deadline',
                    'launched_at', 'usd_pledged', 'id']
    # (2) Columns to apply log transformation
    cols_to_log = {'yes': ['usd_goal', 'creation_to_launch_hours',
                           'diff_main_category_goal', 'diff_sub_category_goal'],
                   'no': ['sub_category', 'staff_pick', 'name_length',
                          'description_length', 'campaign_hours',
                          'country', 'main_category']
                   }
    # (3) Columns to scale and encode
    cols_to_scale_encode = {'cat': ['staff_pick', 'sub_category', 'country', 'main_category'],
                            'num': ['creation_to_launch_hours', 'campaign_hours', 'name_length',
                                    'description_length', 'usd_goal', 'diff_main_category_goal',
                                    'diff_sub_category_goal'],
                            'target': ['state'],
                            'target_mapping': [['failed','successful']]
                            }

    # Create pipeline:
    #   - Name and description length
    #   - Time related features
    #   - Difference between the median and the current category's goal
    #   - Drop columns (1)
    #   - Convert numerical to normal distribution (2)
    #   - Scale numerical feat. and Encode categorical and target feat. (3)

    full_pipeline = create_feat_eng_pipeline(cols_to_drop,
                                             cols_to_log,
                                             cols_to_scale_encode)

    # Apply pipeline to train/val/test data
    kicks_split_processed, full_pipeline = apply_feat_eng_pipeline(full_pipeline,
                                                                   kicks_split_clean)

    # -------------------------------------------------- #
    # Save processed train/val/test and pipeline locally #
    # -------------------------------------------------- #
    logger.info(f'Saving processed (augmented) data and pipeline locally...')
    info_data = save_data(kicks_split_processed,
                          info_data,
                          year, month,
                          is_split=True)
    info_pipe = save_pipe(full_pipeline, info_pipe)

    # ---------------------------------------------------------- #
    # Save processed data and pipeline in S3 Bucket (LocalStack) #
    # ---------------------------------------------------------- #
    logger.info(f'Saving processed (augmented) data and pipeline in a S3 Bucket (LocalStack)...')
    save_to_s3_bucket(params["s3_bucket_name"],
                      info_data=info_data,
                      info_pipe=info_pipe)

    # ---------------------------------------------------- #
    # Log the processed data and pipeline as W&B artifacts #
    # ---------------------------------------------------- #
    logger.info(f'Logging processed data and pipeline to Weights & Biases...')
    run = init_wandb_run(name_script='build_features',
                         job_type='preprocessing')
    # Data
    log_wandb_artifact(run,
                       name_artifact=get_artifact_name(info_data['path_local_out']),
                       type_artifact='dataset',
                       bucket_name=params["s3_bucket_name"],
                       path_to_log=info_data["path_s3_out"])
    # Pipeline
    name_artifact = f"{get_artifact_name(info_pipe['path_local_out'])}_{run.id}"
    log_wandb_artifact(run,
                       name_artifact=name_artifact,
                       type_artifact='model',
                       bucket_name=params["s3_bucket_name"],
                       path_to_log=info_pipe["path_s3_out"])
    # Store name_artifact in .ENV
    dotenv.set_key(dotenv.find_dotenv(), "WANDB_PROCESSED_MODELS", name_artifact)

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
    params = gather_build_features(standalone_mode=False)

    # ------------------------------------------------ #
    # Feat. Eng. and save data (locally and S3 bucket) #
    # ------------------------------------------------ #
    warnings.filterwarnings('ignore',category=UserWarning)
    main(params)


if __name__ == '__main__':
    wrapper_poetry()
