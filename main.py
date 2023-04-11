from fastapi import FastAPI
from models import Roberta
from pydantic import BaseModel

app = FastAPI()


class Tweet(BaseModel):
    text: str


@app.post("/roberta/all")
async def comprehensive_roberta_prediction(tweet: Tweet):
    robertaModels = Roberta()
    labels = robertaModels.predictions(tweet.text)
    return {"labels": labels}


@app.post("/roberta/{task}")
async def specific_roberta_prediction(task: str, tweet: Tweet):
    """
    ["emotion", "hate", "irony", "offensive", "sentiment"]
    """
    robertaModels = Roberta()
    label = robertaModels.predictions(tweet.text, task)
    return {"labels": label}
