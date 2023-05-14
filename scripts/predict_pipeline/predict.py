from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class PredictPipeline:
    def __init__(
            self,
            checkpoint_path: Path,
            train_df_path: Path,
            feature_store_path: Path,
            ohe_category: Path,
        ) -> None:
        self.model = XGBRegressor()
        self.model.load_model(checkpoint_path)
        self.columns = pd.read_parquet(train_df_path).drop(columns=["log_view_count"]).columns.to_list()
        self.fs = pd.read_parquet(feature_store_path)
        self.ohe_category = joblib.load(ohe_category)

    def predict(self, request: Dict[str, str]) -> float:
        # Get input
        channel = request["channel"]
        category = request["category"]
        today = pd.Series(pd.to_datetime(datetime.now().date()))

        # Get features
        fs_channel = (
            self.fs
            .drop_duplicates(subset=["channelTitle"])
            [["channelTitle", "channel_trending_videos_last_7D", "log_channel_avg_n_views_7D",
               "log_channel_avg_n_comments_7D", "log_channel_view_like_ratio_7D"]]
        )
        fs_category = (
            self.fs
            .drop_duplicates(subset=["category"])
            [["category", "category_trending_videos_last_7D", "category_avg_n_views_7D",
              "category_avg_n_comments_7D", "category_view_like_ratio_7D"]]
        )
        category_encoded = pd.DataFrame(
            self.ohe_category.transform(np.array(category).reshape(-1, 1)).toarray(),
            columns=self.ohe_category.get_feature_names_out(),
            )
        category_encoded.columns = [
            ("category_" + col.split("_")[1]).replace(" ", "") for col in category_encoded.columns
            ]

        # Format request
        request = pd.concat([
            fs_channel[fs_channel["channelTitle"] == channel].drop(columns=["channelTitle"]).reset_index(drop=True),
            fs_category[fs_category["category"] == category].drop(columns=["category"]).reset_index(drop=True),
            pd.DataFrame(today.dt.year, columns=["year"]),
            pd.DataFrame(today.dt.month, columns=["month"]),
            pd.DataFrame((today.dt.day_of_week <= 4).astype(int), columns=["is_weekday"]),
            category_encoded,
            ], axis=1,
        )
        request = request[self.columns]

        # Make prediction
        prediction = self.model.predict(request).item()
        prediction = np.exp(prediction)

        return prediction
