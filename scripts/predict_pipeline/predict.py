from datetime import datetime
from pathlib import Path
from typing import Dict

import boto3
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from configs.config import config


class PredictPipeline:
    def __init__(self) -> None:
        self.model = XGBRegressor()
        self.model.load_model(config.model_path)
        self.columns = pd.read_parquet(config.train_set_path).drop(columns=["log_view_count"]).columns.to_list()

        session = boto3.Session(region_name="eu-west-1")
        resource = session.resource("dynamodb")
        self.fs_categories = resource.Table("yt-trending-categories")
        self.fs_channels = resource.Table("yt-trending-channels")

    def predict(self, request: Dict[str, str]) -> float:
        # Get input
        channel = request["channel"]
        category = request["category"]
        today = pd.Series(pd.to_datetime(datetime.now().date()))

        # Get features
        channel_features = self.fs_channels.get_item(Key={"channelTitle": channel})["Item"]
        category_features = self.fs_categories.get_item(Key={"category": category})["Item"]

        # Format Decimals to float
        del channel_features["channelTitle"]
        del category_features["category"]
        channel_features = {k: float(v) for k, v in channel_features.items()}
        category_features = {k: float(v) for k, v in category_features.items()}

        # Format request
        model_input = pd.concat([
            pd.DataFrame([channel_features]),
            pd.DataFrame([category_features]),
            pd.DataFrame(today.dt.year, columns=["year"]),
            pd.DataFrame(today.dt.month, columns=["month"]),
            pd.DataFrame((today.dt.day_of_week <= 4).astype(int), columns=["is_weekday"]),
        ], axis = 1,
        )
        model_input = model_input[self.columns]

        # Make prediction
        prediction = self.model.predict(model_input).item()
        prediction = np.exp(prediction).item()

        return prediction
