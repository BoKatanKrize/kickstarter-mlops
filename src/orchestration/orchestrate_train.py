# ** To be launched as flow deployment in Prefect Cloud **

import os
from dotenv import find_dotenv, load_dotenv

from data.downloader import wrapper_poetry as downloader
from data.cleaner import wrapper_poetry as cleaner
from features.build_features import wrapper_poetry as build_features
from models.train import wrapper_poetry as train
from models.register_model import wrapper_poetry as register_model

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

load_dotenv(find_dotenv())
FLOW_NAME = os.environ["FLOW_NAME"]

# ----- #
# Tasks #
# ----- #
task_downloader = task(downloader, name = "data-downloading")
task_cleaner = task(cleaner, name = "data-cleaning")
task_build_features = task(build_features, name = "feature-engineering")
task_train = task(train, name = "model-training")
task_register_model = task(register_model, name = "model-registry")

# ----- #
# Flow  #
# ----- #
@flow(
      name=f"{FLOW_NAME}",
      task_runner=SequentialTaskRunner()
)
def train_flow():
    """Prefect flow for orchestrating the experiment tracking and model registry,
       using the functions and methods defined in data/, features/ and models/
    """
    logger = get_run_logger()
    logger.info("Downloading raw data")
    task_downloader()
    logger.info("Cleaning raw data")
    task_cleaner()
    logger.info("Feature engineering of cleaned data")
    task_build_features()
    logger.info("Model training by HPO (W&B Sweep)")
    task_train()
    logger.info("Promoting best model to Model Registry")
    task_register_model()