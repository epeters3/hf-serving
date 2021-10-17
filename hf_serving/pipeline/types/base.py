from abc import ABC, abstractmethod
import typing as t

from pydantic import BaseModel


class InputModel(BaseModel, ABC):
    @abstractmethod
    def preprocess(self) -> t.Tuple[tuple, dict]:
        """
        Converts this model to the input arguments for the `transformers.pipeline`'s `__call__` method. Should return a
        two tuple containg the ``args` tuple and `kwargs` dict for that method.
        """
        pass


class OutputModel(BaseModel, ABC):
    @classmethod
    @abstractmethod
    def postprocess(cls, output) -> "OutputModel":
        """Converts the pipeline's `output` to this model."""
        pass


class PipelineTypes(BaseModel):
    task: str
    input_model: t.Type[InputModel]
    output_model: t.Type[OutputModel]
