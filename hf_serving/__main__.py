from fastapi import FastAPI

from hf_serving.config import TASK, MODEL, TOKENIZER
from hf_serving.pipeline.main import TypedPipeline

pipe = TypedPipeline(task=TASK, model=MODEL, tokenizer=TOKENIZER)
app = FastAPI(title=f"{TASK} Service", description="A web service serving up prediction requests for the {TASK} task.")


@app.get("/")
def root():
    return {"detail": "ok", "healthy": True}


@app.post("/predict", response_model=pipe.spec.output_model)
def predict(inputs: pipe.spec.input_model):
    return pipe(inputs)
