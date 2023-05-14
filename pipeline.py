
"""
Data ingestion
1. Load new raw data
2. Extract only last day -> save to segmented data.

Transform
1. Load specific time range
2. Train test split + save benchmark data
3. fillna
4. Add features
5. Save train test datasets

Train
1. Load train / test data
2. Load benchmark data
3. Create benchmarks
4. Train
"""

from configs.config import config
from scripts.data_ingestion.data_ingestor import DataIngestor
from scripts.data_transformation.data_transformer import DataTransformer
from scripts.data_transformation.feature_store import FeatureStoreSupervisor
from scripts.training.benchmarks import Benchmarks
from scripts.training.train import Trainer

print("imports ok")

# inestor = DataIngestor(
#     raw_data_path=config.raw_data_path,
#     categories_path=config.categories_path,
#     segmented_data_dir=config.segmented_data_dir,
#     segmented_file_names_prefix=config.segmented_file_names_prefix,
# )
# inestor.initiate_data_ingestion()

# data_transformer = DataTransformer(
#     segmented_data_dir=config.segmented_data_dir,
#     train_set_path=config.train_set_path,
#     cross_val_set_path=config.cross_val_set_path,
#     train_set_length=config.train_set_length,
#     cv_set_length=config.cv_set_length,
#     artifacts_dir=config.artifacts_dir,
#     train_set_benchmark_path=config.train_set_benchmark_path,
#     cross_val_set_benchmark_path=config.cross_val_set_benchmark_path,
#     feature_store_path=config.feature_store_path,
#     stats_from_past_n_days=config.stats_from_past_days,
#     log_scale_columns=config.log_scale_columns,
#     min_max_scale_columns=config.min_max_scale_columns,
#     ohe_columns=config.ohe_columns,
# )
# data_transformer.initialise_data_transformation()

# benchmarks = Benchmarks(
#     train_set_benchmark_path=config.train_set_benchmark_path,
#     cross_val_set_benchmark_path=config.cross_val_set_benchmark_path,
# )
# benchmarks.initialise_benchmarking()

# trainer = Trainer(
#     train_set_path=config.train_set_path,
#     cross_val_set_path=config.cross_val_set_path,
#     model_path=config.model_path,
#     neptune_project_name=config.neptune_project_name,
#     hyperparam_tuning_n_trials=config.hyperparam_tuning_n_trials,
# )
# trainer.initialise_training_pipeline()

fs = FeatureStoreSupervisor()
fs.initialise_feature_store_update()

print("all good")
