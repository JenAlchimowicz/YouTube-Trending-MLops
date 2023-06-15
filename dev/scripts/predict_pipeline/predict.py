import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

import boto3
import neptune
import numpy as np
import pandas as pd
import xgboost as xgb

from configs.config import config


class PredictPipeline:
    def __init__(self) -> None:
        self.model = self.retrieve_model()
        self.columns = self.model.feature_names

        session = boto3.Session(region_name="eu-west-1")
        resource = session.resource("dynamodb")
        self.fs_categories = resource.Table("yt-trending-categories")
        self.fs_channels = resource.Table("yt-trending-channels")

    def retrieve_model(self):
        models = neptune.init_model(
            with_id=config.neptune_model_registry_name,
            project=config.neptune_project_name,
        )
        model_versions_df = models.fetch_model_versions_table().to_pandas()
        production_model_run_id = (
            model_versions_df
            [model_versions_df["sys/stage"] == "production"]
            .reset_index(drop=True)
            .loc[0, "model/run_id"]
        )

        run = neptune.init_run(
            project=config.neptune_project_name,
            with_id=production_model_run_id,
            mode="read-only",
        )
        run["training/pickled_model"].download(destination=str(config.model_path))
        model = pickle.load(Path.open(config.model_path, "rb"))  # noqa: S301
        return model

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
        model_input = xgb.DMatrix(model_input)

        # Make prediction
        prediction = self.model.predict(model_input).item()
        prediction = np.exp(prediction).item()

        return prediction
