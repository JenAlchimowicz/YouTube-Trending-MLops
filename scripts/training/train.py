from pathlib import Path
from typing import Dict, Tuple

import neptune
import neptune.integrations.optuna as npt_utils
import optuna
import pandas as pd
import xgboost as xgb
from neptune.integrations.xgboost import NeptuneCallback
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from configs.config import config
from scripts.training.benchmarks import Benchmarks


class Trainer:
    def __init__(self) -> None:
        self.benchmark_supervisor = Benchmarks()
        self.benchmarks = {}

    def initialise_training_pipeline(self) -> None:
        self.benchmarks = self.benchmark_supervisor.initialise_benchmarking()
        train, cross_val = self.load_data()
        params, run_id = self.hyperparam_tuning(train, cross_val)
        self.train_final_model(train, cross_val, params, run_id)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(config.train_set_path)
        cross_val = pd.read_parquet(config.cross_val_set_path)
        return train, cross_val

    def hyperparam_tuning(self, train: pd.DataFrame, cross_val: pd.DataFrame) -> Dict[str, float]:
        run = neptune.init_run(
            project=config.neptune_project_name,
        )  # API token passed as env variable
        neptune_callback = npt_utils.NeptuneCallback(run)

        # Log dataset metadata
        run["data_versioning/train"].track_files("data/processed_data/train.parquet")
        run["data_versioning/cross_val"].track_files("data/processed_data/cross_val.parquet")
        run["data_versioning/dimensions/tain_n_cols"] = train.shape[1]
        run["data_versioning/dimensions/tain_n_rows"] = train.shape[0]
        run["data_versioning/dimensions/cross_val_n_cols"] = cross_val.shape[1]
        run["data_versioning/dimensions/cross_val_n_rows"] = cross_val.shape[0]
        run["data_versioning/columns/tain_cols"] = ", ".join(train.columns.tolist())
        run["data_versioning/columns/cross_val_cols"] = ", ".join(cross_val.columns.tolist())

        # Log benchmarks
        run["benchmarks"] = self.benchmarks

        x_train = train.drop(columns=["log_view_count"])
        y_train = train["log_view_count"].to_numpy()
        x_cross_val = cross_val.drop(columns=["log_view_count"])
        y_cross_val = cross_val["log_view_count"].to_numpy()

        def objective(trial) -> Tuple[float, str]:
            param = {
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 10.0),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
                "max_depth": trial.suggest_int("max_depth", 1, 15),
                "random_state": trial.suggest_int("random_state", 1, 2000),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            }

            model = XGBRegressor(**param)
            model.fit(x_train, y_train)

            preds = model.predict(x_cross_val)
            return  mean_squared_error(y_cross_val, preds, squared=False)

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=config.hyperparam_tuning_n_trials,
            timeout=7200,
            callbacks=[neptune_callback],
        )
        best_params = run["best/params"].fetch()
        run_id = run["sys/id"].fetch()
        run.stop()
        return best_params, run_id


    def train_final_model(
            self,
            train: pd.DataFrame,
            cross_val: pd.DataFrame,
            params: Dict[str, float],
            hp_tuning_run_id: str,
        ) -> None:
        run = neptune.init_run(
            project=config.neptune_project_name,
        )  # API token passed as env variable
        neptune_callback = NeptuneCallback(run=run, log_tree=[0, 1, 2, 3])

        # Log dataset meatadata
        train = pd.concat([train, cross_val], axis=0)
        run["data_versioning/train"].track_files("data/processed_data/train.parquet")
        run["data_versioning/dimensions/tain_n_cols"] = train.shape[1]
        run["data_versioning/dimensions/tain_n_rows"] = train.shape[0]
        run["data_versioning/columns/tain_cols"] = ", ".join(train.columns.tolist())
        run["training/hyperparam_tuning_run_id"] = hp_tuning_run_id
        run["benchmarks"] = self.benchmarks

        # Train
        x_train = train.drop(columns=["log_view_count"])
        y_train = train["log_view_count"].to_numpy()
        dtrain = xgb.DMatrix(x_train, label=y_train)

        params_reg = params.copy()
        params_reg["objective"] = "reg:squarederror"
        xgb.train(
            params=params_reg,
            dtrain=dtrain,
            evals=[(dtrain, "train")],
            callbacks=[neptune_callback],
        )
        run_id = run["sys/id"].fetch()
        run.stop()

        # Update model registry
        model_version = neptune.init_model_version(model="YTTREN-XGB", project=config.neptune_project_name)
        model_version["model/parameters"] = params_reg
        model_version["model/run_id"] = run_id
        model_id = model_version["sys/id"]
        model_version.change_stage("production")
        model_version.stop()

        model_registry = neptune.init_model(with_id="YTTREN-XGB", project=config.neptune_project_name)
        model_versions_df = model_registry.fetch_model_versions_table().to_pandas()
        model_registry.stop()

        old_versions_ids = (
            model_versions_df
            [model_versions_df["sys/id"] != model_id]
            [model_versions_df["sys/stage"] == "production"]
            ["sys/id"]
            .tolist()
        )
        for version_id in old_versions_ids:
            model_version = neptune.init_model_version(with_id=version_id, project=config.neptune_project_name)
            model_version.change_stage("archived")
            model_version.stop()
