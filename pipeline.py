from scripts.data_ingestion.data_ingestor import DataIngestor
from scripts.data_transformation.data_transformer import DataTransformer
from scripts.data_transformation.feature_store import FeatureStoreSupervisor
from scripts.training.benchmarks import Benchmarks
from scripts.training.train import Trainer

print("imports ok")

ingestor = DataIngestor()
ingestor.load_and_segment_data()
print("Ingestion done")

data_transformer = DataTransformer()
data_transformer.initialise_data_transformation()
print("Transform done")

fs = FeatureStoreSupervisor()
fs.initialise_feature_store_update()
print("FS done")

# trainer = Trainer()
# trainer.initialise_training_pipeline()

print("all good")
