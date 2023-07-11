from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd


def download_file_from_s3(bucket_name: str, key: str, local_file_path: str) -> None:
    s3_client = boto3.client("s3")

    local_file_path_ = Path(local_file_path)
    if not local_file_path_.parent.exists():
        local_file_path_.parent.mkdir(parents=True)

    s3_client.download_file(bucket_name, key, local_file_path)


def upload_df_to_s3_parquet(df: pd.DataFrame, bucket_name: str, key: str) -> None:
    s3_client = boto3.client("s3")

    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine="pyarrow", compression="snappy")
    parquet_buffer.seek(0)
    s3_client.put_object(Body=parquet_buffer.getvalue(), Bucket=bucket_name, Key=key)


def get_param(
        name: str = "/yt-trending/secrets/NEPTUNE_API_KEY",
        region: str = "eu-west-1",
        with_decryption: bool = True,
) ->  str:
    # Function works only with String and SecureString parameter types.
    ssm = boto3.client("ssm", region_name=region)
    parameter = ssm.get_parameter(Name=name, WithDecryption=with_decryption)
    return parameter["Parameter"]["Value"]
