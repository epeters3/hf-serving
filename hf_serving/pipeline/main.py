import typing as t

from transformers import pipeline

from hf_serving.pipeline.types import pipeline_types
from hf_serving.pipeline.types.base import InputModel


class TypedPipeline:
    """
    Thin wrapper over `transformers.pipeline`, which has pydantic models expressing the types of the pipeline's inputs
    and outputs.
    """

    def __init__(self, task: str, model: t.Optional[str] = None, tokenizer: t.Optional[str] = None):
        self.spec = pipeline_types[task]
        self._pipe = pipeline(task=task, model=model, tokenizer=tokenizer)

    def __call__(self, inputs: InputModel):
        """
        Same as `transformers.pipeline.__call__`, but takes a pydantic model as input, and gives a pydantic model as
        output.
        """
        args, kwargs = inputs.preprocess()
        preds = self._pipe(*args, **kwargs)
        return self.spec.output_model.postprocess(preds)
