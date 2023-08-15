import os
import boto3
import botocore
from dotenv import find_dotenv, load_dotenv


def bucket_exists(bucket_name):
    s3 = boto3.resource('s3')
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
        return True
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return False


def upload_to_bucket(client, bucket_name, info):
    client.put_object(Bucket=bucket_name, Key=f'{info["path_s3_out"]}/')
    for fname in info['fnames']:
        client.upload_file(f'{info["path_local_out"]}/{fname}',
                           bucket_name,
                           f'{info["path_s3_out"]}/{fname}')


def download_from_bucket(client, bucket_name, info):
    # List objects with the specified prefix (folder)
    objects = client.list_objects(Bucket=bucket_name,
                                  Prefix=f'{info["path_s3_in"]}/{info["prefix_name"]}')
    # Iterate through objects and download each file
    for obj in objects.get('Contents', []):
        file_key = obj['Key']
        local_file_path = os.path.join(info["path_local_in"], os.path.basename(file_key))
        client.download_file(bucket_name, file_key, local_file_path)


def save_to_s3_bucket(bucket_name,
                      info_data = None,
                      info_pipe = None):
    """
    Access an existing S3 Bucket (LocalStack), add to
    its folder structure, and upload the data (optional)
    & pipeline (optional)
    """

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    AWS_ENDPOINT_URL = os.environ.get('AWS_ENDPOINT_URL')
    AWS_DEFAULT_REGION = os.environ.get('AWS_DEFAULT_REGION')

    # Define client
    client = boto3.client('s3', endpoint_url=AWS_ENDPOINT_URL)

    # Check if bucket exists
    if not bucket_exists(bucket_name):
        client.create_bucket(Bucket=bucket_name,
                             CreateBucketConfiguration={'LocationConstraint':
                                                            AWS_DEFAULT_REGION})

    # Data
    if info_data is not None:
        upload_to_bucket(client, bucket_name, info_data)
    # Pipeline
    if info_pipe is not None:
        upload_to_bucket(client, bucket_name, info_pipe)


def load_from_s3_bucket(bucket_name,
                        info_data = None,
                        info_pipe = None):
    """
    Access an existing S3 Bucket (LocalStack) to download
    data (optional) & pipeline (optional)
    """

    load_dotenv(find_dotenv())
    AWS_ENDPOINT_URL = os.environ.get('AWS_ENDPOINT_URL')

    client = boto3.client('s3', endpoint_url=AWS_ENDPOINT_URL)

    # Data
    if info_data is not None:
        download_from_bucket(client, bucket_name, info_data)
    # Pipeline
    if info_pipe is not None:
        download_from_bucket(client, bucket_name, info_pipe)