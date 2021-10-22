import typing as t

from hf_serving.pipeline.specs.base import PipelineSpec
from hf_serving.pipeline.specs.summarization import summarization
from hf_serving.pipeline.specs.translation import translation

# TODO: Add more specs for more pipeline tasks.
_pipeline_specs = [summarization, translation]


def get_pipeline_spec(task: str) -> t.Optional[PipelineSpec]:
    """
    Given the name of a HuggingFace pipeline task, get the spec for it, meaning, the object
    that can be used to convert the pipelines inputs and outputs to FastAPI-friendly data structures.
    """
    for spec in _pipeline_specs:
        if spec.task_name_matches(task):
            return spec
    return None
