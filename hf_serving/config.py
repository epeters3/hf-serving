import os


TASK = os.getenv("TASK")
"""The HuggingFace pipeline task that will be used to predict with."""

if not TASK:
    raise AssertionError(
        "Please supply a `TASK` environment variable. Any task passed to `transformers.pipeline` is acceptable. "
        "Docs: https://huggingface.co/transformers/main_classes/pipelines.html#transformers.pipeline"
    )

MODEL = os.getenv("MODEL")
"""Optional ID or file path of a custom HF model to use."""

TOKENIZER = os.getenv("TOKENIZER")
"""Optional ID or file path of a custom tokenizer to use."""
