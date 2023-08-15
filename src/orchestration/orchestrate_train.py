from config import gather_orchestrate_train
from data.downloader import wrapper_poetry as downloader
from data.cleaner import wrapper_poetry as cleaner
from features.build_features import wrapper_poetry as build_features
from models.train import wrapper_poetry as train
from models.register_model import wrapper_poetry as register_model

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner


# ----- #
# Tasks #
# ----- #
task_downloader = task(downloader, name = "Data Downloading",log_prints=True)
task_cleaner = task(cleaner, name = "Data Cleaning",log_prints=True)
task_build_features = task(build_features, name = "Feature Engineering",log_prints=True)
task_train = task(train, name = "Model Training",log_prints=True)
task_register_model = task(register_model, name = "Promoting Model to Registry",log_prints=True)

# ----- #
# Flow  #
# ----- #
@flow(
      name="Data Preprocessing, Model Training and ",
#      task_runner=SequentialTaskRunner()
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
    #logger.info("Promoting best model to Model Registry")
    #task_register_model()

# def main(params):
# """Creates a flow deployment in Prefect Cloud for 'train_flow'"""
#

def wrapper_poetry():

    params = gather_orchestrate_train(standalone_mode=False)

    #load_dotenv(find_dotenv())

    train_flow()

    #main(params)


if __name__ == '__main__':
    wrapper_poetry()