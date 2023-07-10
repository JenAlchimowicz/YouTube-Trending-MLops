from typing import List

import pandas as pd


class FeatureEngineer:
    def __init__(self) -> None:
        pass

    def intialise_feature_engineering(self, df: pd.DataFrame, n_days: List[int]) -> pd.DataFrame:
        df = self.get_date_features(df)
        for days in n_days:
            # Category stats
            df = (
                df
                .pipe(self.get_category_stats_sum_over_window, col_to_sum="likes", n_days=days)
                .pipe(self.get_category_stats_sum_over_window, col_to_sum="view_count", n_days=days)
                .pipe(self.get_category_stats_sum_over_window, col_to_sum="comment_count", n_days=days)
                .pipe(self.get_category_occurances_over_window, n_days=days)
                .assign(**{f"category_avg_n_views_{days}D": lambda x: x[f"category_view_count_sum_{days}D"] / x[f"category_trending_videos_last_{days}D"]})
                .assign(**{f"category_avg_n_comments_{days}D": lambda x: x[f"category_comment_count_sum_{days}D"] / x[f"category_trending_videos_last_{days}D"]})
                .assign(**{f"category_view_like_ratio_{days}D": lambda x: x[f"category_view_count_sum_{days}D"] / x[f"category_likes_sum_{days}D"]})
                .drop(columns=[f"category_likes_sum_{days}D", f"category_view_count_sum_{days}D", f"category_comment_count_sum_{days}D"])
            )

            # Channel stats
            df = (
                df
                .pipe(self.get_channel_stats_sum_over_window, col_to_sum="likes", n_days=days)
                .pipe(self.get_channel_stats_sum_over_window, col_to_sum="view_count", n_days=days)
                .pipe(self.get_channel_stats_sum_over_window, col_to_sum="comment_count", n_days=days)
                .pipe(self.get_channel_occurances_over_window, n_days=days)
                .assign(**{f"channel_avg_n_views_{days}D": lambda x: x[f"channel_view_count_sum_{days}D"] / x[f"channel_trending_videos_last_{days}D"]})
                .assign(**{f"channel_avg_n_comments_{days}D": lambda x: x[f"channel_comment_count_sum_{days}D"] / x[f"channel_trending_videos_last_{days}D"]})
                .assign(**{f"channel_view_like_ratio_{days}D": lambda x: x[f"channel_view_count_sum_{days}D"] / x[f"channel_likes_sum_{days}D"]})
                .drop(columns=[f"channel_likes_sum_{days}D", f"channel_view_count_sum_{days}D", f"channel_comment_count_sum_{days}D"])
            )
        return df

    def get_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df
            .assign(is_weekday=(df["trending_date"].dt.dayofweek <= 4).astype(int))
            .assign(year=df["trending_date"].dt.year)
            .assign(month=df["trending_date"].dt.month)
        )
        return df

    def get_category_stats_sum_over_window(self, df: pd.DataFrame, col_to_sum: str, n_days: int) -> pd.DataFrame:
        df_agg = (
            df
            .assign(trending_date_date=df["trending_date"].map(lambda x: x.normalize()))
            .groupby(["category", "trending_date_date"])
            .sum(numeric_only=True)
            [col_to_sum]
            .to_frame()
            .reset_index()
            .sort_values(by="trending_date_date")
            .set_index("trending_date_date")
            .groupby("category")
            .rolling(window=f"{n_days}D")
            .sum()
            .rename(columns={col_to_sum: f"category_{col_to_sum}_sum_{n_days}D"})
        )

        df["trending_date_date"] = df["trending_date"].map(lambda x: x.normalize())

        dff = (
            pd
            .merge(df, df_agg, left_on=["category", "trending_date_date"], right_on=["category", "trending_date_date"], how="left")
            .drop(columns=["trending_date_date"])
            )
        return dff


    def get_category_occurances_over_window(self, df: pd.DataFrame, n_days: int) -> pd.DataFrame:
        col_to_sum = "channelTitle"
        df_agg =(
            df
            .assign(trending_date_date=df["trending_date"].map(lambda x: x.normalize()))
            .groupby(["category", "trending_date_date"])
            .count()
            [col_to_sum]
            .to_frame()
            .reset_index()
            .sort_values(by="trending_date_date")
            .set_index("trending_date_date")
            .groupby("category")
            .rolling(window=f"{n_days}D")
            .sum()
            .rename(columns={col_to_sum: f"category_trending_videos_last_{n_days}D"})
        )

        df["trending_date_date"] = df["trending_date"].map(lambda x: x.normalize())

        dff = (
            pd
            .merge(df, df_agg, left_on=["category", "trending_date_date"], right_on=["category", "trending_date_date"], how="left")
            .drop(columns=["trending_date_date"])
            )
        return dff


    def get_channel_stats_sum_over_window(self, df: pd.DataFrame, col_to_sum: str, n_days: int) -> pd.DataFrame:
        df_agg = (
            df
            .assign(trending_date_date=df["trending_date"].map(lambda x: x.normalize()))
            .groupby(["channelTitle", "trending_date_date"])
            .sum(numeric_only=True)
            [col_to_sum]
            .to_frame()
            .reset_index()
            .sort_values(by="trending_date_date")
            .set_index("trending_date_date")
            .groupby("channelTitle")
            .rolling(window=f"{n_days}D")
            .sum()
            .rename(columns={col_to_sum: f"channel_{col_to_sum}_sum_{n_days}D"})
        )

        df["trending_date_date"] = df["trending_date"].map(lambda x: x.normalize())

        dff = (
            pd
            .merge(df, df_agg, left_on=["channelTitle", "trending_date_date"], right_on=["channelTitle", "trending_date_date"], how="left")
            .drop(columns=["trending_date_date"])
            )
        return dff


    def get_channel_occurances_over_window(self, df: pd.DataFrame, n_days: int) -> pd.DataFrame:
        col_to_sum = "category"
        df_agg =(
            df
            .assign(trending_date_date=df["trending_date"].map(lambda x: x.normalize()))
            .groupby(["channelTitle", "trending_date_date"])
            .count()
            [col_to_sum]
            .to_frame()
            .reset_index()
            .sort_values(by="trending_date_date")
            .set_index("trending_date_date")
            .groupby("channelTitle")
            .rolling(window=f"{n_days}D")
            .sum()
            .rename(columns={col_to_sum: f"channel_trending_videos_last_{n_days}D"})
        )

        df["trending_date_date"] = df["trending_date"].map(lambda x: x.normalize())

        dff = (
            pd
            .merge(df, df_agg, left_on=["channelTitle", "trending_date_date"], right_on=["channelTitle", "trending_date_date"], how="left")
            .drop(columns=["trending_date_date"])
            )
        return dff
