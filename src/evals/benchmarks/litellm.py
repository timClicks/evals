"""LiteLLM metrics.

https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

import asyncio
import json
from decimal import Decimal
from typing import Any, Literal, Self

import httpx
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, computed_field

from evals.modelmap import ModelMapper
from evals.orm import Benchmark, BenchmarkResult, BenchmarkType
from evals.settings import get_settings

from ._benchmark import _Benchmark

# The relative weight to give to input tokens for overall scored price.
# TODO: Needs to be determined based on real data
INPUT_WEIGHT = Decimal.from_float(0.75)
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
    input_cpt: Decimal = Field(alias="input_cost_per_token")
    output_cpt: Decimal = Field(alias="output_cost_per_token")

    @computed_field
    @property
    def weighted_cpt(self) -> Decimal:
        return INPUT_WEIGHT * self.input_cpt + (Decimal(1) - INPUT_WEIGHT) * self.output_cpt

    # Provider or host
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
        # Anything that is not "chat" mode, for example.
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

def assemble_frame(extract_model_names=True) -> pl.DataFrame:
    """
    Extract input and output costs and derive a score for each model.

    When extract_model_names is True, also ensure that all model names are present in "data/working/model-names.json" and the corresponding csv file.
    """
    file_path = get_file_path()

    # Read it in with JSON and filter out unwanted models
    with file_path.open("r") as fd:
        data = json.load(fd)

    # Create models and filter out None.
    models = []
    for model, details in data.items():
        validated = LLMCaps.maybe_create(model, details)
        if validated is not None:
            models.append(validated)

    # Create a DataFrame with the models now we are sure the datatypes are correct.
    df = pl.DataFrame([item.model_dump() for item in models])

    return df


def assemble():
    logger.info(f"Writing LiteLLM data to parquet at {FRAME_PATH}")
    df = assemble_frame()
    df.write_parquet(FRAME_PATH)


def extract_model_names():
    import json
    import pandas as pd

    model_names_path = get_settings().get_base_dir() / "working" / "model-id-mapping.json"
    df = pd.read_parquet(FRAME_PATH)

    try:
        with open(model_names_path, "r") as fd:
            all_model_names = json.load(fd)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        all_model_names = dict()

    for model_id in df["model"].unique():
        model_id = model_id + " [litellm] "
        if model_id in all_model_names:
            continue
        logger.info(f"new model id to map - {model_id}")
        all_model_names[model_id] = None

    # TODO: fix race condition- leaving in for now because we process in serial
    with model_names_path.open("w", newline="") as fd:
        json.dump(all_model_names, fd, indent=2, sort_keys=True)

def all():
    # download()
    # assemble()
    extract_model_names()


class LiteLLM(_Benchmark):
    def get_benchmarks(self, mm: ModelMapper) -> BenchmarkResult:
        df = self.load_frame()

        # Filter for our models
        df = self.map_and_filter_column(df, "model", mm)

        df = df.rename(
            {
                "weighted_cpt": "score",
            }
        )
        # Rename the columns and add the kind we are.
        scores = [Benchmark.model_validate(dct) for dct in df.iter_rows(named=True)]
        return BenchmarkResult(bm_type=BenchmarkType.COST, scores=scores, unit="cpt")


# def get_scores(models: list[str]) -> pl.DataFrame:
#     """Get scores for a list of models."""
#     df = assemble_frame()
#     return df.filter(pl.col("model").isin(models))
