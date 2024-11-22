"""LiteLLM metrics.

https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

import asyncio
import json
from typing import Any, Literal, Self

import httpx
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from evals.modelmap import ModelMapper
from evals.scoring import BenchmarkResult, BenchmarkType, ModelScore
from evals.settings import get_settings

from ._benchmark import _Benchmark

# The relative weight to give to input tokens
# for overall scored price. Needs to be determined
# based on real data
INPUT_WEIGHT = 0.75
DATA_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

FRAME_PATH = get_settings().get_frames_dir() / "litellm.parquet"


class LLMCaps(BaseModel):
    """LiteLLM capabilities.

    Get pydantic to check the data for us.
    """

    # We add this field.
    model: str = ""

    # These are the required fields (we shorten the names).
    input: float = Field(alias="input_cost_per_token")
    output: float = Field(alias="output_cost_per_token")
    provider: str = Field(alias="litellm_provider")

    # This is the only mode we accept for now.
    mode: Literal[
        "chat",
        # "embedding",
        # "completion",
        # "image_generation",
        # "audio_transcription",
        # "audio_speech",
        # These are the data...
        # "moderations",
        # "rerank",
    ]

    @classmethod
    def maybe_create(cls, model: str, dct: Any) -> Self | None:
        """Extract values (and filter out unwanted models)."""

        # Put the model-name into data.
        model = model.split("/")[-1].lower()
        dct["model"] = model

        # Pydantic will throw out anything that doesn't meet our demands above.
        try:
            slf = cls.model_validate(dct)
        except ValidationError:
            return None

        # Filter out unwanted models and rename providers.
        match slf.provider:
            case "aleph_alpha":
                # Aleph-Alpha is very costly (which skews costs) and not
                # publicly available (?) so ignore
                return None
            case "openai" if model.startswith("ft:"):
                # Ignore fine-tuning prices
                return None
            case "azure_ai":
                slf.provider = "azure"
            case "cohere_chat":
                slf.provider = "cohere"
            case "fireworks_ai":
                slf.provider = "fireworks"
            case provider if provider.startswith("vertex_ai"):
                slf.provider = "vertex"

        return slf


def get_file_path():
    downloads_dir = get_settings().get_downloads_dir("litellm")
    file_name = DATA_URL.split("/")[-1]
    return downloads_dir / file_name


async def async_download():
    """Async download current version of data file."""
    file_path = get_file_path()

    async with httpx.AsyncClient() as client:
        response = await client.get(DATA_URL, follow_redirects=True)
        response.raise_for_status()

        logger.info(f"Downloading {DATA_URL}")
        with file_path.open("wb") as file:
            async for chunk in response.aiter_bytes(chunk_size=8192):
                file.write(chunk)


def download():
    asyncio.run(async_download())


def assemble_frame() -> pl.DataFrame:
    """Extract input and output costs and derive a score for each model."""
    file_path = get_file_path()

    # Read it in with JSON and filter out unwanted models
    with file_path.open("r") as fd:
        data = json.load(fd)

    # Create models and filter out None.
    models = [
        created_model
        for model, details in data.items()
        if (created_model := LLMCaps.maybe_create(model, details)) is not None
    ]

    # Create a DataFrame with the models now we are sure the datatypes are correct.
    df = pl.DataFrame([item.model_dump() for item in models])

    c_input, c_output = [pl.col(col) for col in "input output".split()]

    # Column normalization function.
    # This makes the smallest value => 1 and the largest value => 0.
    def normalize(c: pl.Expr) -> pl.Expr:
        return 1.0 - (c - c.min()) / (c.max() - c.min())

    # Normalize columns first
    df = df.with_columns(
        [
            normalize(c_input).alias("input_n"),
            normalize(c_output).alias("output_n"),
        ]
    )

    # Calculate the weighted score
    df = df.with_columns(
        (
            INPUT_WEIGHT * pl.col("input_n") + (1 - INPUT_WEIGHT) * pl.col("output_n")
        ).alias("score")
    )

    return df


def assemble():
    df = assemble_frame()
    df.write_parquet(FRAME_PATH)


def all():
    download()
    assemble()


class LiteLLM(_Benchmark):
    def get_benchmarks(self, mm: ModelMapper) -> BenchmarkResult:
        df = self.load_frame()

        # Filter for our models
        df = self.map_and_filter_column(df, "model", mm)

        # Rename the columns and add the kind we are.
        scores = [ModelScore.model_validate(dct) for dct in df.iter_rows(named=True)]
        return BenchmarkResult(bm_type=BenchmarkType.COST, scores=scores)


# def get_scores(models: list[str]) -> pl.DataFrame:
#     """Get scores for a list of models."""
#     df = assemble_frame()
#     return df.filter(pl.col("model").isin(models))
