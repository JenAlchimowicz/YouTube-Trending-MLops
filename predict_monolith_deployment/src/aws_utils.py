from io import BytesIO
from typing import List

import boto3
import pandas as pd


def download_parquet_from_s3_to_df(bucket_name: str, key: str) -> pd.DataFrame:
    s3_client = boto3.client("s3")
    parquet_buffer = BytesIO()
    s3_client.download_fileobj(Bucket=bucket_name, Key=key, Fileobj=parquet_buffer)
    parquet_buffer.seek(0)
    dataframe = pd.read_parquet(parquet_buffer, engine="pyarrow")
    return dataframe


def retrieve_item_list(bucket_name: str, key: str) -> List[str]:
    item_list = download_parquet_from_s3_to_df(bucket_name, key)
    return item_list.iloc[:, 0].to_list()


def get_param(
        name: str = "/yt-trending/secrets/NEPTUNE_API_KEY",
        region: str = "eu-west-1",
        with_decryption: bool = True,
) ->  str:
    # Function works only with String and SecureString parameter types.
    ssm = boto3.client("ssm", region_name=region)
    parameter = ssm.get_parameter(Name=name, WithDecryption=with_decryption)
    return parameter["Parameter"]["Value"]
