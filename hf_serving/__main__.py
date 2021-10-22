from fastapi import FastAPI
from starlette.responses import RedirectResponse

from hf_serving.config import TASK, MODEL, TOKENIZER
from hf_serving.pipeline.main import TypedPipeline

pipe = TypedPipeline(task=TASK, model=MODEL, tokenizer=TOKENIZER)
app = FastAPI(
    title=f"{TASK.capitalize()} Service",
    description=(
        f"A web service serving up prediction requests for the {TASK} task. See the [docs]({pipe.spec.docs_link}) "
        "for more information about how the pipeline works."
    )
)


@app.post("/predict", response_model=pipe.spec.output_model)
def predict(inputs: pipe.spec.input_model):
    return pipe(inputs)


@app.get("/")
def root():
    return RedirectResponse("/redoc")


@app.get("/health")
def health_check():
    return {"detail": "ok", "healthy": True}