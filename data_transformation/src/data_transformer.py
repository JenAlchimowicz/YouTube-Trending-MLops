from datetime import datetime, timedelta
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from aws_utils import download_file_from_s3, upload_df_to_s3_parquet, upload_file_to_s3
from config import config
from feature_engineering import FeatureEngineer
from joblib import dump
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DataTransformer:
    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()

    def initialise_data_transformation(self) -> None:
        """Main pipeline of data transformer.

        Returns
            pd.DataFrame: Data cleaned and enriched with new features, contains both train and cv sets
        """
        self.download_data_from_s3()
        df = self.load_data_in_window()
        df = self.initial_clean(df)
        df = self.add_features(df, config.stats_from_past_days)

        train, cross_val = self.train_cv_split(df)
        train, cross_val = self.fill_null_values(train, cross_val)
        train, cross_val = self.log_scale_features(train, cross_val, config.log_scale_columns)
        train, cross_val = self.min_max_scale_features(train, cross_val, config.min_max_scale_columns)
        self.save_benchmark_datasets(train, cross_val)
        train, cross_val = self.ohe_features(train, cross_val, config.ohe_columns)

        train = train.drop(columns=["trending_date", "channelTitle"])
        cross_val = cross_val.drop(columns=["trending_date", "channelTitle"])

        if not config.train_set_path.parent.exists():
            config.train_set_path.parent.mkdir(parents=True)
        if not config.cross_val_set_path.parent.exists():
            config.cross_val_set_path.parent.mkdir(parents=True)

        train.to_parquet(config.train_set_path)
        cross_val.to_parquet(config.cross_val_set_path)

        upload_df_to_s3_parquet(train, config.s3_bucket_name, config.s3_train_set_path)
        upload_df_to_s3_parquet(cross_val, config.s3_bucket_name, config.s3_cross_val_set_path)


    def download_data_from_s3(self) -> None:
        download_file_from_s3(
            config.s3_bucket_name,
            config.s3_raw_data_dir + "/GB_youtube_trending_data.csv",
            str(config.raw_data_path),
        )
        download_file_from_s3(
            config.s3_bucket_name,
            config.s3_raw_data_dir + "/GB_category_id.json",
            str(config.categories_path),
        )

    def load_data_in_window(self) -> pd.DataFrame:
        """Loads raw data and categories mapping. Validates the raw data. Unpacks nested columns in categories mapping.
        Joins categories to raw data. Filters to window.

        Raises
            ValueError: if raw_data_path is invalid
            ValueError: if categories_path is invalid
            ValueError: if raw data doesn't have required columns
            ValueError: if categories dataframe doesn't have required columns

        Returns
            pd.DataFrame: Raw data containing both train and cv sets
        """
        if not config.raw_data_path.exists() or not config.raw_data_path.is_file():
            raise ValueError("Invalid raw data file path")
        if not config.categories_path.exists() or not config.categories_path.is_file():
            raise ValueError("Invalid categories file path")

        df = pd.read_csv(config.raw_data_path)
        df["trending_date"] = pd.to_datetime(df["trending_date"])
        necessary_cols = {"channelTitle", "trending_date", "categoryId", "likes", "comment_count", "view_count"}
        if not necessary_cols.issubset(df.columns):
            raise ValueError("Raw data file does not have required columns")

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
        if not {"id", "title"}.issubset(categories.columns):
            raise ValueError("Categories file does not have required columns")
        categories = (
            categories
            [["id", "title"]]
            .assign(categoryId=categories["id"].astype("int64"))
            .rename(columns={"title": "category"})
            .drop(columns="id")
            )

        # Join categories to raw data
        df = pd.merge(df, categories, on="categoryId").drop(columns=["categoryId"])

        # Filter to widnow
        most_recent_date_in_data = df["trending_date"].max()
        date_cutoff = most_recent_date_in_data - timedelta(
            days = config.train_set_length + config.cv_set_length + max(config.stats_from_past_days),
        )
        df = df[df["trending_date"] >= date_cutoff]

        return df


    def initial_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["channelTitle", "trending_date", "category", "likes", "comment_count", "view_count"]]
        df = df.drop_duplicates()

        cant_be_none = ["view_count", "trending_date"]
        df = df.dropna(subset=cant_be_none ,how="any")

        df["channelTitle"] = df["channelTitle"].fillna("unk")
        df["category"] = df["category"].fillna("unk")

        return df


    def add_features(self, df: pd.DataFrame, n_days: List[int]) -> pd.DataFrame:
        """Adds 3 types of features: date based, category based and channel based.

        Args:
            df (pd.DataFrame): Clened data with both train and cv sets in it
            (it is neccesary for the two sets to be together)

        Returns
            pd.DataFrame: Dataframe with 11 new features
        """
        df = self.feature_engineer.intialise_feature_engineering(df, n_days)

        # Cleanup
        df = df.drop(columns=["likes", "comment_count"])

        most_recent_date_in_data = df["trending_date"].max().to_pydatetime().replace(tzinfo=None)
        date_cutoff = most_recent_date_in_data - timedelta(days = config.train_set_length + config.cv_set_length)
        df = df[df["trending_date"].map(lambda x: x.replace(tzinfo=None)) >= date_cutoff]

        return df


    def train_cv_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        latest_date = df["trending_date"].max()
        cv_set_range = latest_date - timedelta(days=config.cv_set_length)
        train_set_range = latest_date - timedelta(days=config.train_set_length + config.cv_set_length)

        cross_val = df[df["trending_date"] >= cv_set_range]
        train = df[(df["trending_date"] >= train_set_range) & (df["trending_date"] < cv_set_range)]

        train = train.reset_index(drop=True)
        cross_val = cross_val.reset_index(drop=True)

        return train, cross_val

    @staticmethod
    def fill_null_values(train: pd.DataFrame, cross_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train["category"] = train["category"].fillna("unk")
        cross_val["category"] = cross_val["category"].fillna("unk")

        numeric_cols = train.select_dtypes(include="number").columns.tolist()

        train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].mean())
        cross_val[numeric_cols] = cross_val[numeric_cols].fillna(cross_val[numeric_cols].mean())

        for col in numeric_cols:
            mean_in_train_set = train[col].replace([np.inf, -np.inf], np.nan).mean()
            train[col] = train[col].replace([np.inf, -np.inf], mean_in_train_set)
            cross_val[col] = cross_val[col].replace([np.inf, -np.inf], mean_in_train_set)

        return train, cross_val

    @staticmethod
    def log_scale_features(train: pd.DataFrame, cross_val: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for column in columns:
            train = (
                train
                .assign(**{f"log_{column}": np.log(train[column] + 1)})
                .drop(columns=[column])
            )
            cross_val = (
                cross_val
                .assign(**{f"log_{column}": np.log(cross_val[column] + 1)})
                .drop(columns=[column])
            )
        return train, cross_val

    def min_max_scale_features(
            self,
            train: pd.DataFrame,
            cross_val: pd.DataFrame,
            columns: List[str],
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for column in columns: #category_avg_n_views_7D
            scaler = MinMaxScaler()
            train[column] = scaler.fit_transform(train[[column]])
            cross_val[column] = scaler.transform(cross_val[[column]])
            self.save_artifact(scaler, f"scaler_{column}.joblib")

        return train, cross_val

    def ohe_features(
            self,
            train: pd.DataFrame,
            cross_val: pd.DataFrame,
            columns: List[str],
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for column in columns:
            train_ohe_feature = train[column]
            cross_val_ohe_feature = cross_val[column]

            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(train_ohe_feature.to_numpy().reshape(-1, 1))

            train_category_encoded = pd.DataFrame(
                encoder.transform(train_ohe_feature.to_numpy().reshape(-1, 1)).toarray(),
                columns=encoder.get_feature_names_out(),
            )
            cross_val_category_encoded = pd.DataFrame(
                encoder.transform(cross_val_ohe_feature.to_numpy().reshape(-1, 1)).toarray(),
                columns=encoder.get_feature_names_out(),
            )

            # modify the column names
            train_category_encoded.columns = [
                (f"{column}_" + col.split("_")[1]).replace(" ", "") for col in train_category_encoded.columns
            ]
            cross_val_category_encoded.columns = [
                (f"{column}_" + col.split("_")[1]).replace(" ", "") for col in cross_val_category_encoded.columns
            ]

            # concatenate the encoded category column with the original train and cross_val sets
            train = pd.concat([train.drop("category", axis=1), train_category_encoded], axis=1)
            cross_val = pd.concat([cross_val.drop("category", axis=1), cross_val_category_encoded], axis=1)

            self.save_artifact(encoder, f"ohe_encoder_{column}.joblib")

        return train, cross_val

    def save_artifact(self, artifact: Any, filename: str) -> None:
        local_file_path = config.artifacts_dir.joinpath(filename)

        if not local_file_path.parent.exists():
            local_file_path.parent.mkdir(parents=True)

        dump(artifact, str(local_file_path))
        upload_file_to_s3(str(local_file_path), config.s3_bucket_name, config.s3_artifacts_dir + "/" + filename)

    def save_benchmark_datasets(self, train: pd.DataFrame, cross_val: pd.DataFrame) -> None:
        train = train[["channelTitle", "category", "log_view_count"]]
        cross_val = cross_val[["channelTitle", "category", "log_view_count"]]

        if not config.train_set_benchmark_path.parent.exists():
            config.train_set_benchmark_path.parent.mkdir(parents=True)
        if not config.cross_val_set_benchmark_path.parent.exists():
            config.cross_val_set_benchmark_path.parent.mkdir(parents=True)

        train.to_parquet(config.train_set_benchmark_path)
        cross_val.to_parquet(config.cross_val_set_benchmark_path)
