import pandas as pd
from configs.config import config
from tqdm import tqdm


def create_historical_segmented_data():
    """
    Script that loads raw data, selects only useful columns and saves the data
    in separate parquet files segmented on trending date. e.g. all data
    that was trending on 25/03/2023 will be saved in file
    "gb_2023-03-25.parquet".
    """
    df = pd.read_csv(config.raw_data_path)
    categories = pd.read_json(config.categories_path)

    df["trending_date"] = pd.to_datetime(df["trending_date"])
    df = df[["channelTitle", "trending_date", "categoryId", "likes", "comment_count", "view_count"]]
    df["trending_date"] = pd.to_datetime(df["trending_date"])

    categories = pd.concat([
        categories.drop(["items"], axis=1),
        categories["items"].apply(lambda x: pd.Series(x)),
        ], axis=1)
    categories = pd.concat([
        categories.drop(["snippet"], axis=1),
        categories["snippet"].apply(lambda x: pd.Series(x)),
        ], axis=1)

    categories = (
        categories
        [["id", "title"]]
        .assign(categoryId=categories["id"].astype("int64"))
        .rename(columns={"title": "category"})
        .drop(columns="id")
        )

    df = pd.merge(df, categories, on="categoryId").drop(columns=["categoryId"])

    df["trending_date_only_date"] = df["trending_date"].map(lambda x: x.normalize())
    trending_date_timezone = df["trending_date_only_date"].dt.tz

    for date_to_filter in tqdm(df["trending_date_only_date"].dt.date.unique()):
        timestamp = pd.Timestamp(date_to_filter, tz=trending_date_timezone)
        tmp_df = df[df["trending_date"] == timestamp]
        tmp_df = tmp_df.drop(columns=["trending_date_only_date"])
        tmp_df.to_parquet(
            config.segmented_data_dir.joinpath(f"{config.segmented_file_names_prefix}{date_to_filter}.parquet"),
            )
