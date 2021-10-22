import typing as t

from transformers import pipeline

from hf_serving.pipeline.specs import get_pipeline_spec
from hf_serving.pipeline.specs.base import InputModel


class TypedPipeline:
    """
    Thin wrapper over `transformers.pipeline`, which has pydantic models expressing the specs of the pipeline's inputs
    and outputs.
    """

    def __init__(self, task: str, model: t.Optional[str] = None, tokenizer: t.Optional[str] = None):
        spec = get_pipeline_spec(task)
        if spec is None:
            raise ValueError(f"The {task} pipeline is currently unsupported.")
        self.spec = spec
        self._pipe = pipeline(task=task, model=model, tokenizer=tokenizer)

    def __call__(self, inputs: InputModel):
        """
        Same as `transformers.pipeline.__call__`, but takes a pydantic model as input, and gives a pydantic model as
        output.
        """
        args, kwargs = inputs.preprocess()
        preds = self._pipe(*args, **kwargs)
        return self.spec.output_model.postprocess(preds)
