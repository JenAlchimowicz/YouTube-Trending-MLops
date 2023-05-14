from pathlib import Path
from typing import List


class config:  # noqa: N801
    # Data paths
    raw_data_path: Path = Path("data/raw_data/GB_youtube_trending_data.csv")
    categories_path: Path = Path("data/raw_data/GB_category_id.json")
    segmented_data_dir: Path = Path("data/segmented_data")
    segmented_file_names_prefix: str = "gb_"
    artifacts_dir: Path = Path("artifacts")
    train_set_path: Path = Path("data/processed_data/train.parquet")
    cross_val_set_path: Path = Path("data/processed_data/cross_val.parquet")
    train_set_benchmark_path: Path = Path("data/benchmarks/train_benchmark.parquet")
    cross_val_set_benchmark_path: Path = Path("data/benchmarks/cross_val_benchmark.parquet")
    feature_store_dir: Path = Path("data/feature_store")
    # Add columns train order

    # S3 paths
    s3_bucket_name: str = "yt-trending-mlops"
    s3_feature_store_dir: str = "data/feature_store"
    s3_train_set_path: str = "data/processed_data/train.parquet"
    s3_cross_val_set_path: str = "data/processed_data/cross_val.parquet"
    s3_artifacts_dir: str = "artifacts"

    # Data transformation
    stats_from_past_days: List[int] = [7]
    log_scale_columns: List[str] = [
        "channel_avg_n_views_7D",
        "channel_avg_n_comments_7D",
        "channel_view_like_ratio_7D",
        "view_count",
    ]
    min_max_scale_columns: List[str] = ["category_avg_n_views_7D"]
    ohe_columns: List[str] = ["category"]

    # Train params
    model_path: Path = Path("checkpoints/xgb_model.json")
    train_set_length: int = 150  # in days
    cv_set_length: int = 21  # in days

    neptune_project_name: str = "jen-alchimowicz/yt-trending-mlops"
    hyperparam_tuning_n_trials: int = 10
