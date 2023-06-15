from fastapi import FastAPI
from predict import PredictPipeline
from pydantic import BaseModel


class Item(BaseModel):
    channel: str
    category: str

app = FastAPI(title="yt-trending-api")
predict_pipeline = PredictPipeline()

@app.post("/")
async def predict(payload: Item):
    return predict_pipeline.predict(payload.dict())
