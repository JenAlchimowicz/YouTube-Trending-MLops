from pathlib import Path

import pandas as pd
from configs.config import config
from scripts.utils.s3 import upload_df_to_s3_parquet


class DataIngestor:
    def __init__(
            self,
            raw_data_path: Path,
            categories_path: Path,
            segmented_data_dir: Path,
            segmented_file_names_prefix: str,
        ) -> None:
        self.raw_data_path = raw_data_path
        self.categories_path = categories_path
        self.segmented_data_dir = segmented_data_dir
        self.segmented_file_names_prefix = segmented_file_names_prefix

    def load_data(self):
        """Loads raw data and categories mapping. Unpacks nested columns in categories mapping.

        Raises
            ValueError: if raw_data_path is invalid
            ValueError: if categories_path is invalid
            ValueError: if raw data doesn't have required columns
            ValueError: if categories dataframe doesn't have required columns

        Returns
            _type_: _description_
        """
        if not self.raw_data_path.exists() or not self.raw_data_path.is_file():
            raise ValueError("Invalid raw data file path")
        if not self.categories_path.exists() or not self.categories_path.is_file():
            raise ValueError("Invalid categories file path")

        df = pd.read_csv(self.raw_data_path)
        categories = pd.read_json(self.categories_path)

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

    def raw_data_has_new_dates(self, df: pd.DataFrame) -> bool:
        """Checks if passed dataframe containing raw data (should be freshly downloaded) has new data as compared
        to data already present in 'data/segmented_data'.

        Args:
            df (pd.DataFrame): Newly downloaded and formatted raw data

        Returns
            bool: True if new data is found, False if not
        """
        latest_date_in_new_data = df["trending_date"].max()
        all_segmented_filepaths = self.segmented_data_dir.glob("*")
        all_segmented_filenames = [filepath.stem for filepath in list(all_segmented_filepaths)]
        most_recent_filename = sorted(all_segmented_filenames)[-1]
        latest_date_in_old_data = most_recent_filename.split("_")[1]
        latest_date_in_old_data_timestamp = pd.Timestamp(latest_date_in_old_data, tz=df["trending_date"].dt.tz)

        #TODO: change to logger
        if latest_date_in_new_data <= latest_date_in_old_data_timestamp:
            print(f"No new data detected. Latest date in segmented data: {latest_date_in_old_data}, "
                  "latest date in new raw data: {latest_date_in_new_data.date()}")

        return latest_date_in_new_data > latest_date_in_old_data_timestamp

    @staticmethod
    def add_categories_to_raw_data(df: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        """Unpacks categories dataframe and joins it to raw data.
        Raw data (df) onyl has categoryId, which is numeric. The string equivalent is stored in categories dataframe.

        Args:
            df (pd.DataFrame): Raw data
            categories (pd.DataFrame): Categories mapping

        Returns
            pd.DataFrame: same shape as 'df' input, but numeric categoires is 'categoryId' are replaced
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

    def initiate_data_ingestion(self) -> None:
        """
        Main function to load, join and save raw data.
        """
        df, categories = self.load_data()

        if self.raw_data_has_new_dates(df):
            df = df[df["trending_date"] == df["trending_date"].max()]
            df = self.add_categories_to_raw_data(df, categories)

            filepath = self.segmented_data_dir.joinpath(
                f"{self.segmented_file_names_prefix}{df['trending_date'].dt.date.unique().item()}.parquet",
            )
            df.to_parquet(filepath, index=False)
            # upload_df_to_s3_parquet(df, "", "")
