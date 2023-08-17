import os
import io
import re
import zipfile
import logging
import requests

import pandas as pd
import wandb
from bs4 import BeautifulSoup

from cli import gather_downloader
from utils.aws_s3 import save_to_s3_bucket
from utils.io import save_data
from utils.wandb import init_wandb_run, log_wandb_artifact, \
                        get_artifact_name


def extract_year_month(url):
    pattern = r"Kickstarter_(\d{4})-(\d{2})-\d{2}T\d{2}_\d{2}_\d{2}_\d{3}Z.zip"
    match = re.search(pattern, url)
    year = match.group(1)
    month = match.group(2)
    return year, month


def get_data_url(base_url, extension, data_format):
    data = requests.get(f"{base_url}/{extension}/")
    parsed = BeautifulSoup(data.text, "html.parser")
    link = parsed.find("a", href=re.compile(data_format))
    zip_file_url = link["href"]
    year, month = extract_year_month(zip_file_url)
    return zip_file_url, year, month


def download_raw_data(zip_file_url):
    r = requests.get(zip_file_url)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    dfs = [pd.read_csv(zf.open(f)) for f in zf.namelist()]
    df = pd.concat(dfs, ignore_index=True)
    return df


def main(params):
    """ Downloads the latest data available from the given URL and saves it
        in .parquet format locally (~/data/raw) and in S3 Bucket
    """

    logger = logging.getLogger(__name__)

    # -------------------------- #
    # Obtain the URL of the data #
    # -------------------------- #
    info_url = dict(params["info_url"])
    logger.info(f'Accessing {info_url["base_url"]}...')
    zip_file_url, year, month = get_data_url(info_url['base_url'],
                                             info_url['extension'],
                                             info_url['data_format'])

    # -------------------- #
    # Downloading the data #
    # -------------------- #
    logger.info(f'Downloading raw data ({month}/{year}) from {info_url["base_url"]}/...')
    df = download_raw_data(zip_file_url)

    # ---------------------------------------------------------- #
    # Save the data (multiple .csv) as unique .parquet (locally) #
    # ---------------------------------------------------------- #
    logger.info(f'Saving raw data locally...')
    info_data = dict(params['info_data'])

    info_data = save_data(df, info_data,
                          year, month,
                          is_split=False)

    # --------------------------------------- #
    # Save the data im S3 Bucket (LocalStack) #
    # --------------------------------------- #
    logger.info(f'Saving raw data in a S3 Bucket (LocalStack)...')
    save_to_s3_bucket(params["s3_bucket_name"],
                      info_data=info_data,
                      info_pipe=None)

    # ----------------------------------- #
    # Log the raw data as an W&B artifact #
    # ----------------------------------- #
    logger.info(f'Logging the raw data to Weights & Biases...')
    run = init_wandb_run(name_script='downloader',
                         job_type='preprocessing')
    log_wandb_artifact(run,
                       name_artifact=get_artifact_name(info_data['path_local_out']),
                       type_artifact='dataset',
                       bucket_name=params["s3_bucket_name"],
                       path_to_log=info_data["path_s3_out"])
    wandb.finish()


def wrapper_poetry():
    """ So that we can call this script using Poetry"""

    # -------------- #
    # Logging config #
    # -------------- #
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # only log messages with a severity level of INFO or higher will be displayed
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # any subsequent log messages will look like:
    #   timestamp + logger name + severity level + the actual log message

    # ------------------------------ #
    # Load parameters from cli.py #
    # ------------------------------ #
    params = gather_downloader(standalone_mode=False)

    # ---------------------------------------------- #
    # Download and save data (locally and S3 bucket) #
    # ---------------------------------------------- #
    main(params)


if __name__ == '__main__':
    wrapper_poetry()