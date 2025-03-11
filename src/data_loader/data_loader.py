import boto3
import pandas as pd
import os

# cache
from functools import lru_cache


@lru_cache(maxsize=100)
def download_data_from_s3(bucket_name, file_name):
    """Download a file from S3 and load it into a DataFrame."""
    s3_client = boto3.client("s3")
    try:
        s3_client.download_file(bucket_name, file_name, file_name)
        print(f"Successfully downloaded {file_name} from S3")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
    data = pd.read_csv(file_name)
    os.remove(file_name)
    return data


def prepare_data(data):
    """Prepare the dataset by splitting features and labels."""
    X = data.drop("label", axis=1)
    y = data["label"]
    return X, y
