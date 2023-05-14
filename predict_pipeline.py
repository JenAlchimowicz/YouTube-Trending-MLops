from configs.config import config
from scripts.predict_pipeline.predict import PredictPipeline

predict_pipeline = PredictPipeline(
    checkpoint_path=config.model_path,
    train_df_path=config.train_set_path,
    feature_store_path=config.feature_store_path,
    ohe_category=config.artifacts_dir.joinpath("ohe_encoder_category.joblib"),
)

r1 = {"channel": "BT Sport", "category": "Sports"}
r2 = {"channel": "BT Sport", "category": "Education"}
r3 = {"channel": "CNN", "category": "News & Politics"}
r4 = {"channel": "CGP Grey", "category": "Education"}

print(predict_pipeline.predict(r1))
print(predict_pipeline.predict(r2))
print(predict_pipeline.predict(r3))
print(predict_pipeline.predict(r4))
