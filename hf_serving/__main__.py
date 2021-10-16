from fastapi import FastAPI
from transformers import pipeline

from hf_serving.config import TASK, MODEL, TOKENIZER


pipe = pipeline(task=TASK, model=MODEL, tokenizer=TOKENIZER)
app = FastAPI(title=f"{TASK} Service", description="A web service serving up prediction requests for the {TASK} task.")


@app.get("/")
def root():
    return {"detail": "ok", "healthy": True}


@app.post("/predict")
def predict(pred):
    return pipe(pred)
