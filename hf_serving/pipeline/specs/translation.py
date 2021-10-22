import typing as t

from pydantic import BaseModel

from hf_serving.pipeline.specs.base import InputModel, OutputModel, PipelineSpec


class TranslationInputs(InputModel):
    texts: t.Union[str, t.List[str]]
    """The texts to be translated."""

    def preprocess(self):
        return (self.dict()["texts"],), {}


class Translation(BaseModel):
    translationText: str


class TranslationOutputs(OutputModel):
    predictions: t.List[Translation]

    @classmethod
    def postprocess(cls, output):
        return cls(predictions=[Translation(translationText=item["translation_text"]) for item in output])


translation = PipelineSpec(
    docs_link="https://huggingface.co/transformers/main_classes/pipelines.html#translationpipeline",
    task_name_matches=lambda name: name.startswith("translation"),
    input_model=TranslationInputs,
    output_model=TranslationOutputs
)
