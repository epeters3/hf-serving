import typing as t

from hf_serving.pipeline.types.base import PipelineTypes
from hf_serving.pipeline.types.summarization import summarization

# TODO: Add more specs for more pipeline tasks.
all_specs = [summarization]

pipeline_types: t.Dict[str, PipelineTypes] = {spec.task: spec for spec in all_specs}
