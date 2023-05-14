from pathlib import Path

import pandas as pd

from configs.config import config


class DataIngestor:
    def __init__(self) -> None:
        pass

    def load_data(self):
        """Loads raw data and categories mapping. Unpacks nested columns in categories mapping.

        Raises
            ValueError: if raw_data_path is invalid
            ValueError: if categories_path is invalid
            ValueError: if raw data doesn't have required columns
            ValueError: if categories dataframe doesn't have required columns
        """
        if not config.raw_data_path.exists() or not config.raw_data_path.is_file():
            raise ValueError("Invalid raw data file path")
        if not config.categories_path.exists() or not config.categories_path.is_file():
            raise ValueError("Invalid categories file path")

        df = pd.read_csv(config.raw_data_path)
        categories = pd.read_json(config.categories_path)

        # Unpack categories
        categories = pd.concat([
            categories.drop(["items"], axis=1),
            categories["items"].apply(lambda x: pd.Series(x)),
            ], axis=1)
        categories = pd.concat([
            categories.drop(["snippet"], axis=1),
            categories["snippet"].apply(lambda x: pd.Series(x)),
            ], axis=1)

        necessary_cols = {"channelTitle", "trending_date", "categoryId", "likes", "comment_count", "view_count"}
        if not necessary_cols.issubset(df.columns):
            raise ValueError("Raw data file does not have required columns")
        if not {"id", "title"}.issubset(categories.columns):
            raise ValueError("Categories file does not have required columns")

        # Format raw data
        df["trending_date"] = pd.to_datetime(df["trending_date"])

        return df, categories

    @staticmethod
    def add_categories_to_raw_data(df: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        """Unpacks categories dataframe and joins it to raw data.
        Raw data (df) onyl has categoryId, which is numeric. The string equivalent is stored in categories dataframe.

        Args:
            df (pd.DataFrame): Raw data
            categories (pd.DataFrame): Categories mapping

        Returns
            pd.DataFrame: same shape as 'df' input, but numeric categoires 'categoryId' are replaced
            with string equivalents in 'category' column.
        """
        categories = (
            categories
            [["id", "title"]]
            .assign(categoryId=categories["id"].astype("int64"))
            .rename(columns={"title": "category"})
            .drop(columns="id")
            )

        df = pd.merge(df, categories, on="categoryId").drop(columns=["categoryId"])
        return df

    def load_and_segment_data(self) -> None:
        """
        Main function to load, join and save raw data.
        """
        df, categories = self.load_data()
        df = self.add_categories_to_raw_data(df, categories)

        for trending_date in df["trending_date"].unique():
            df_day = df[df["trending_date"] == trending_date]
            filepath = config.segmented_data_dir.joinpath(
                f"{config.segmented_file_names_prefix}{df_day['trending_date'].dt.date.unique().item()}.parquet",
            )
            df_day.to_parquet(filepath, index=False)
