from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Benchmarks:
    def __init__(
            self,
            train_set_benchmark_path: Path,
            cross_val_set_benchmark_path: Path,
        ) -> None:
        self.train_set_benchmark_path = train_set_benchmark_path
        self.cross_val_set_benchmark_path = cross_val_set_benchmark_path

    def initialise_benchmarking(self):
        train, cross_val = self.load_benchmark_data()
        mea1, mse1 = self.benchmark_mean_across_entire_dataset(train, cross_val)
        mea2, mse2 = self.benchmark_mean_across_categories(train, cross_val)
        mea3, mse3 = self.benchmark_mean_across_channels(train, cross_val)
        print("Benchmarks:")
        print(mea1, mea2, mea3)

    def load_benchmark_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_benchmark = pd.read_parquet(self.train_set_benchmark_path)
        cross_val_benchmark = pd.read_parquet(self.cross_val_set_benchmark_path)
        return train_benchmark, cross_val_benchmark

    @staticmethod
    def benchmark_mean_across_entire_dataset(train: pd.DataFrame, cross_val: pd.DataFrame) -> Tuple[float, float]:
        y_train = train["log_view_count"].to_numpy()
        y_cross_val = cross_val["log_view_count"].to_numpy()
        mean_entire_df = y_train.mean()
        mae_mean_across_entire_dataset = mean_absolute_error(y_cross_val, np.full_like(y_cross_val, mean_entire_df))
        mse_mean_across_entire_dataset = mean_squared_error(
            y_cross_val, np.full_like(y_cross_val, mean_entire_df), squared=False,
            )
        return mae_mean_across_entire_dataset, mse_mean_across_entire_dataset

    @staticmethod
    def benchmark_mean_across_categories(train: pd.DataFrame, cross_val: pd.DataFrame) -> Tuple[float, float]:
        avg_views_per_category = (
            train
            .groupby("category")
            .mean("log_view_count")
            .reset_index(drop=False)
            .rename(columns={"log_view_count": "log_view_count_preds"})
        )
        views_per_category_preds = pd.merge(cross_val, avg_views_per_category, on=["category"], how="left")

        mae_mean_across_categories = mean_absolute_error(
            views_per_category_preds["log_view_count"], views_per_category_preds["log_view_count_preds"],
            )
        mse_mean_across_categories = mean_squared_error(
            views_per_category_preds["log_view_count"], views_per_category_preds["log_view_count_preds"], squared=False,
            )

        return mae_mean_across_categories, mse_mean_across_categories

    @staticmethod
    def benchmark_mean_across_channels(train: pd.DataFrame, cross_val: pd.DataFrame) -> Tuple[float, float]:
        avg_views_per_category = (
            train
            .groupby("channelTitle")
            .mean("log_view_count")
            .reset_index(drop=False)
            .rename(columns={"log_view_count": "log_view_count_preds"})
        )
        views_per_category_preds = pd.merge(cross_val, avg_views_per_category, on=["channelTitle"], how="left")

        mean_entire_df = train["log_view_count"].to_numpy().mean()
        views_per_category_preds["log_view_count_preds"] = (
            views_per_category_preds["log_view_count_preds"].fillna(mean_entire_df)
            )
        mae_mean_across_channels = mean_absolute_error(
            views_per_category_preds["log_view_count"], views_per_category_preds["log_view_count_preds"],
            )
        mse_mean_across_channels = mean_squared_error(
            views_per_category_preds["log_view_count"], views_per_category_preds["log_view_count_preds"], squared=False,
            )

        return mae_mean_across_channels, mse_mean_across_channels

