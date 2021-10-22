import typing as t

from pydantic import BaseModel

from hf_serving.pipeline.specs.base import InputModel, OutputModel, PipelineSpec


class SummarizationInputs(InputModel):
    documents: t.Union[str, t.List[str]]

    def preprocess(self):
        return (self.dict()["documents"],), {}


class Summary(BaseModel):
    summaryText: str


class SummarizationOutputs(OutputModel):
    predictions: t.List[Summary]

    @classmethod
    def postprocess(cls, output):
        return cls(predictions=[Summary(summaryText=item["summary_text"]) for item in output])


summarization = PipelineSpec(
    docs_link="https://huggingface.co/transformers/main_classes/pipelines.html#summarizationpipeline",
    task_name_matches=lambda name: name == "summarization",
    input_model=SummarizationInputs,
    output_model=SummarizationOutputs
)
