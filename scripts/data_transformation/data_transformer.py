from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from scripts.data_transformation.feature_engineering import FeatureEngineer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DataTransformer:
    def __init__(
            self,
            segmented_data_dir: Path,
            train_set_path: Path,
            cross_val_set_path: Path,
            train_set_length: Path,
            cv_set_length: Path,
            artifacts_dir: Path,
            train_set_benchmark_path: Path,
            cross_val_set_benchmark_path: Path,
            feature_store_path: Path,
            stats_from_past_n_days: List[int],
            log_scale_columns: List[str],
            min_max_scale_columns: List[str],
            ohe_columns: List[str],
        ) -> None:
        """Init function.

        Args:
            segmented_data_dir (Path): Required naming format of files: "prefix_yyyy-mm-dd.parquet"
            train_set_length (Path): Number of days to include in train set (data always starts from most recent day)
            cv_set_length (Path): Number of days to include in cross_val set (data always starts from most recent day)
        """
        super().__init__()
        self.segmented_data_dir = segmented_data_dir
        self.train_set_path = train_set_path
        self.cross_val_set_path = cross_val_set_path
        self.train_set_length = train_set_length
        self.cv_set_length = cv_set_length
        self.artifacts_dir = artifacts_dir
        self.train_set_benchmark_path = train_set_benchmark_path
        self.cross_val_set_benchmark_path = cross_val_set_benchmark_path
        self.feature_store_path = feature_store_path
        self.stats_from_past_n_days = stats_from_past_n_days
        self.log_scale_columns = log_scale_columns
        self.min_max_scale_columns = min_max_scale_columns
        self.ohe_columns = ohe_columns

        self.feature_engineer = FeatureEngineer()

    def initialise_data_transformation(self) -> None:
        """Main pipeline of data transformer.

        Returns
            pd.DataFrame: Data cleaned and enriched with new features, contains both train and cv sets
        """
        df = self.load_data_in_window()
        df = self.initial_clean(df)
        df = self.add_features(df, self.stats_from_past_n_days)

        train, cross_val = self.train_cv_split(df)
        train, cross_val = self.fill_null_values(train, cross_val)
        train, cross_val = self.log_scale_features(train, cross_val, self.log_scale_columns)
        train, cross_val = self.min_max_scale_features(train, cross_val, self.min_max_scale_columns)
        self.save_benchmark_datasets(train, cross_val)
        self.save_to_feature_store(cross_val)
        train, cross_val = self.ohe_features(train, cross_val, self.ohe_columns)

        train = train.drop(columns=["trending_date", "channelTitle"])
        cross_val = cross_val.drop(columns=["trending_date", "channelTitle"])

        train.to_parquet(self.train_set_path)
        cross_val.to_parquet(self.cross_val_set_path)

    def load_data_in_window(self) -> pd.DataFrame:
        """Loads data in a specific time window.

        Returns
            pd.DataFrame: Raw data containing both train and cv sets
        """
        # Get path to most recent data
        all_segmented_filepaths = self.segmented_data_dir.glob("*")
        all_segmented_filenames = [filepath.name for filepath in list(all_segmented_filepaths)]
        most_recent_filename = sorted(all_segmented_filenames)[-1]

        # Check the most recent date
        df_latest = pd.read_parquet(self.segmented_data_dir.joinpath(most_recent_filename))
        most_recent_date_in_data = df_latest["trending_date"].max().to_pydatetime().replace(tzinfo=None)

        # Get date cutoff - we need extra 7 days to calculate historical features
        date_cutoff = most_recent_date_in_data - timedelta(days = self.train_set_length + self.cv_set_length + 7)

        # Get filepaths within date range
        file_paths = []
        for file_path in self.segmented_data_dir.glob("*.parquet"):
            file_date_str = file_path.stem.split("_")[1]
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
            if file_date >= date_cutoff:
                file_paths.append(file_path)

        # Load data
        df_list = []
        for file_path in file_paths:
            df = pd.read_parquet(file_path)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)

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
        date_cutoff = most_recent_date_in_data - timedelta(days = self.train_set_length + self.cv_set_length)
        df = df[df["trending_date"].map(lambda x: x.replace(tzinfo=None)) >= date_cutoff]

        return df


    def train_cv_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        latest_date = df["trending_date"].max()
        cv_set_range = latest_date - timedelta(days=self.cv_set_length)
        train_set_range = latest_date - timedelta(days=self.train_set_length + self.cv_set_length)

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
        dump(artifact, self.artifacts_dir.joinpath(filename))

    def save_benchmark_datasets(self, train: pd.DataFrame, cross_val: pd.DataFrame) -> None:
        train = train[["channelTitle", "category", "log_view_count"]]
        cross_val = cross_val[["channelTitle", "category", "log_view_count"]]

        train.to_parquet(self.train_set_benchmark_path)
        cross_val.to_parquet(self.cross_val_set_benchmark_path)

    def save_to_feature_store(self, df) -> None:
        df = df[df["trending_date"] == df["trending_date"].max()]
        df.to_parquet(self.feature_store_path)
