from configs.config import config
from scripts.predict_pipeline.predict import PredictPipeline

predict_pipeline = PredictPipeline()

r1 = {"channel": "BT Sport", "category": "Sports"}
r2 = {"channel": "BT Sport", "category": "Education"}
r3 = {"channel": "CNN", "category": "News & Politics"}
r4 = {"channel": "CGP Grey", "category": "Education"}

print(predict_pipeline.predict(r1))
print(predict_pipeline.predict(r2))
print(predict_pipeline.predict(r3))
print(predict_pipeline.predict(r4))
