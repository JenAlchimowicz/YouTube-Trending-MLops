import os
from io import BytesIO

import boto3
import pandas as pd


def upload_df_to_s3_parquet(df: pd.DataFrame, bucket_name: str, key: str) -> None:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine="pyarrow", compression="snappy")
    parquet_buffer.seek(0)
    s3_client.put_object(Body=parquet_buffer.getvalue(), Bucket=bucket_name, Key=key)


def download_parquet_from_s3_to_df(bucket_name: str, key: str) -> pd.DataFrame:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    parquet_buffer = BytesIO()
    s3_client.download_fileobj(Bucket=bucket_name, Key=key, Fileobj=parquet_buffer)
    parquet_buffer.seek(0)
    dataframe = pd.read_parquet(parquet_buffer, engine="pyarrow")
    return dataframe


def list_files_in_s3_directory(bucket_name, directory_path):
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=directory_path)

    if "Contents" in response:
        files = response["Contents"]
        return [file["Key"] for file in files]
    else:
        return []

# your_dataframe = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
# upload_df_to_s3_parquet(your_dataframe, "yt-trending-mlops", "data/tmp.parquet")

# downloaded_dataframe = download_parquet_from_s3_to_df("yt-trending-mlops", "data/tmp.parquet")
# print(downloaded_dataframe)

print(list_files_in_s3_directory("yt-trending-mlops", "data/"))
