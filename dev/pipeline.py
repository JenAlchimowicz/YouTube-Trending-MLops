from scripts.data_transformation.data_transformer import DataTransformer
from scripts.data_transformation.feature_store import FeatureStoreSupervisor
from scripts.training.train import Trainer

print("imports ok")

# ./scripts/data_ingestion/data_download.sh
# print("Ingestion done")

# data_transformer = DataTransformer()
# data_transformer.initialise_data_transformation()
# print("Transform done")

# fs = FeatureStoreSupervisor()
# fs.initialise_feature_store_update()
# print("FS done")

trainer = Trainer()
trainer.initialise_training_pipeline()
print("Training done")

print("all good")
