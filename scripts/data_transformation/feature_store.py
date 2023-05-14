from datetime import timedelta
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from configs.config import config
from scripts.data_transformation.feature_engineering import FeatureEngineer


class FeatureStoreSupervisor:
    def __init__(self) -> None:
        pass

    def initialise_feature_store_update(self):
        df = self.load_all_data()
        df = self.fill_data(df)
        df = self.transform(df)
        fs_channels, fs_category = self.separate_channels_and_categories(df)
        self.save_feature_store_data(fs_channels, fs_category)


    def load_all_data(self) -> pd.DataFrame:
        # Load all data
        all_segmented_filepaths = config.segmented_data_dir.glob("*")
        df_list = []
        for file_path in all_segmented_filepaths:
            df = pd.read_parquet(file_path)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)

        # Initial clean
        df = df[["channelTitle", "trending_date", "category", "likes", "comment_count", "view_count"]]
        df = df.drop_duplicates()

        cant_be_none = ["view_count", "trending_date"]
        df = df.dropna(subset=cant_be_none ,how="any")

        df["channelTitle"] = df["channelTitle"].fillna("unk")
        df["category"] = df["category"].fillna("unk")
        return df

    def fill_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """To compute features for each existing channel and category in the dataframe,
        we need to add a row with the latest date per each channel and category.

        Args:
            df (pd.DataFrame): Raw dataframe containing all of history

        Returns
            pd.DataFrame: Input dataframe enriched with new rows
        """
        # Extract channels and categories
        channels = (
            df
            ["channelTitle"]
            .drop_duplicates()
        )

        categories = (
            df
            ["category"]
            .drop_duplicates()
        )

        # Add line for each channel and category
        latest_date = df["trending_date"].max()
        new_rows = [{
            "channelTitle": channel,
            "trending_date": latest_date,
            "category": "unk",
            "likes": 0,
            "comment_count": 0,
            "view_count": 0,
        } for channel in channels]
        new_rows = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows], ignore_index=True)

        new_rows = [{
            "channelTitle": "unk",
            "trending_date": latest_date,
            "category": category,
            "likes": 0,
            "comment_count": 0,
            "view_count": 0,
        } for category in categories]
        new_rows = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows], ignore_index=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Completes feature engineering as in feature_engineering.py.
        """
        # Get last x days
        latest_date = df["trending_date"].max()
        date_cutoff = latest_date - timedelta(days = max(config.stats_from_past_days) + 1)
        df = df[df["trending_date"] >= date_cutoff]

        # Get features
        feature_engineer = FeatureEngineer()
        df = feature_engineer.intialise_feature_engineering(df, config.stats_from_past_days)
        df = df.drop(columns=["likes", "comment_count"])

        # Leave only latest date
        df = df[df["trending_date"] == latest_date]

        # fill_null_values
        df["category"] = df["category"].fillna("unk")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        for col in numeric_cols:
            mean_in_train_set = df[col].replace([np.inf, -np.inf], np.nan).mean()
            df[col] = df[col].replace([np.inf, -np.inf], mean_in_train_set)

        # log_scale_features
        for column in config.log_scale_columns:
            df = (
                df
                .assign(**{f"log_{column}": np.log(df[column] + 1)})
                .drop(columns=[column])
            )

        # min_max_scale_features
        for column in config.min_max_scale_columns:
            scaler = joblib.load(config.artifacts_dir / f"scaler_{column}.joblib")
            df[column] = scaler.transform(df[[column]])

        # Ohe
        df = df.reset_index(drop=True)
        for ohe_col in config.ohe_columns:
            encoder = joblib.load(config.artifacts_dir / f"ohe_encoder_{ohe_col}.joblib")
            encoded = pd.DataFrame(
                encoder.transform(df[ohe_col].to_numpy().reshape(-1, 1)).toarray(),
                columns=encoder.get_feature_names_out(),
            )
            encoded.columns = [
                (f"{ohe_col}_" + col.split("_")[1]).replace(" ", "") for col in encoded.columns
            ]
            if ohe_col not in ["category", "channelTitle"]:
                df = df.drop(ohe_col, axis=1)
            df = pd.concat([df, encoded], axis=1)

        return df

    def separate_channels_and_categories(self, df :pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fs_channels = (
            df
            [[col for col in df.columns if "channel" in col]]
            [df["channelTitle"] != "unk"]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        avg_row = fs_channels.drop(columns=["channelTitle"]).mean()
        avg_df = pd.DataFrame(avg_row).transpose()
        avg_df["channelTitle"] = "unk"
        fs_channels = (
            pd.concat([fs_channels, avg_df], axis=0)
            .reset_index(drop=True)
        )

        fs_category = (
            df
            [[col for col in df.columns if "category" in col]]
            [df["category"] != "unk"]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        avg_row = fs_category.drop(columns=["category"]).mean()
        avg_df = pd.DataFrame(avg_row).transpose()
        avg_df["category"] = "unk"
        fs_category = (
            pd.concat([fs_category, avg_df], axis=0)
            .reset_index(drop=True)
        )

        return fs_channels, fs_category

    def save_feature_store_data(self, fs_channels: pd.DataFrame, fs_category: pd.DataFrame) -> None:
        fs_channels.to_parquet(config.feature_store_dir / "fs_channels.parquet")
        fs_category.to_parquet(config.feature_store_dir / "fs_category.parquet")
