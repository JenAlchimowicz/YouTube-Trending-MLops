
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

# ingestor = DataIngestor()
# ingestor.load_and_segment_data()

# data_transformer = DataTransformer()
# data_transformer.initialise_data_transformation()

# trainer = Trainer()
# trainer.initialise_training_pipeline()

fs = FeatureStoreSupervisor()
fs.initialise_feature_store_update()

print("all good")
