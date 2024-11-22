"""Speed metrics based on data collected by https://thefastest.ai."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final

import httpx
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from evals.benchmarks._benchmark import _Benchmark
from evals.modelmap import ModelMapper
from evals.orm import BenchmarkResult, BenchmarkType, ModelScore
from evals.settings import get_settings

ORIGIN = "thefastestai"
REGIONS = ["cdg", "iad", "sea"]
BASE_URL = "https://storage.googleapis.com/thefastest-data"


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model_name: str
    location: str = ""


# Provider-specific parsing functions
def _parse_azure(model: str) -> tuple[str, str]:
    """Parse Azure-specific model format."""
    if "/" in model:
        location, model_name = model.split("/")
    else:
        model_name, location, _ = model.split(".")

    location = location.replace("fixie-", "").replace("azure", "").replace(".", "")
    model_name = model_name.replace("fixie-", "")
    return location, model_name


PROVIDER_PATTERNS: Final[dict[str, str]] = {
    "anthropic/": "anthropic",
    "claude-": "anthropic",
    "anyscale.com/": "anyscale",
    "cloudflare/": "cloudflare",
    "cloudflare.com/": "cloudflare",
    "@cf/": "cloudflare",
    "cohere/": "cohere",
    "command-": "cohere",
    "databricks.com/": "databricks",
    "fireworks.ai/": "fireworks",
    "google/": "google",
    "groq.com/": "groq",
    "Neets-": "neets",
    "octo.ai/": "octoai",
    "octoai.run/": "octoai",
    "openai/": "openai",
    "gpt-": "openai",
    "ovh.net/": "ovh",
    "perplexity.ai/": "perplexity",
    "together.ai/": "together",
    "together.xyz": "together",
    "togethercomputer/": "together",
}


def parse_model(model: str) -> ModelInfo:
    # Handle Azure special case first
    if "azure" in model:
        location, model_name = _parse_azure(model)
        return ModelInfo("azure", model_name, location)

    # Check against known patterns
    for pattern, provider in PROVIDER_PATTERNS.items():
        if pattern in model.lower():  # Case-insensitive matching
            model_name = model.split("/")[-1].lower()
            return ModelInfo(provider, model_name)

    # Default case for unknown providers
    return ModelInfo("?", model)


# We use pydantic to validate the data.
class ResultItem(BaseModel):
    model: str
    # Yes, sometimes this is null.
    ttr: float | None
    ttft: float
    tps: float
    total_time: float
    # We don't need these, so we can comment them out.
    # num_tokens: int | None = None
    # output: str | None = None

    # We will fill these out ourselves in the post init.
    provider: str = Field(init=False, default="")
    location: str = Field(init=False, default="")

    def model_post_init(self, _: Any):
        model_info = parse_model(self.model)
        self.provider = model_info.provider
        self.model = model_info.model_name
        self.location = model_info.location


class ResultDateRegion(BaseModel):
    time: datetime
    region: str
    results: list[ResultItem]  # Using built-in list type annotation
    # NOT NEEDED.
    # cmd: str

    def to_frame(self) -> pl.DataFrame:
        """Convert the data to a Polars DataFrame."""
        df = pl.DataFrame(self.results)

        return df.with_columns(
            [
                pl.lit(self.time.date()).alias("date"),  # Drop time?
                pl.lit(self.region).alias("region"),
            ]
        )


async def download_date(date: datetime):
    """Download files for a date (if not already present)."""
    downloads_dir = get_settings().get_downloads_dir(ORIGIN)

    async with httpx.AsyncClient() as client:
        for region in REGIONS:
            file_name = f"{region}-{date:%Y-%m-%d}.json"
            file_path = downloads_dir / file_name
            if file_path.exists():
                continue

            url = f"{BASE_URL}/{region}/text/{date:%Y-%m-%d}.json"
            response = await client.get(url, follow_redirects=True)

            if response.status_code == 404:
                continue

            response.raise_for_status()

            logger.info(f"Downloading {file_name}")

            with file_path.open("wb") as file:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    file.write(chunk)


async def async_download():
    """Download data for all dates."""

    start_date = datetime(2024, 4, 13)
    end_date = datetime.now()

    date = start_date
    # Start TaskGroup for downloading files concurrently
    async with asyncio.TaskGroup() as tg:
        # Add specific date downloads to TaskGroup
        while date <= end_date:
            tg.create_task(download_date(date))
            date += timedelta(days=1)


def download():
    asyncio.run(async_download())


def assemble_frame() -> pl.DataFrame:
    """Score models for all dates since start of dataset to the current date."""
    downloads_dir = get_settings().get_downloads_dir(ORIGIN)
    frames: list[pl.DataFrame] = []
    for file in downloads_dir.glob("*.json"):
        try:
            result = ResultDateRegion.model_validate_json(file.read_bytes())
        except ValidationError as e:
            logger.error(f"Error parsing {file.name}: {e}")
            continue
        logger.info(f"Processing {file.name}, {result.time:%Y-%m-%d}, {result.region}")
        frames.append(result.to_frame())

    df = pl.concat(frames)
    return df


def assemble():
    df = assemble_frame()
    df.write_parquet(get_settings().get_frames_dir() / f"{ORIGIN}.parquet")


def all():
    download()
    assemble()


class TheFastestAI(_Benchmark):
    def get_benchmarks(self, mm: ModelMapper) -> BenchmarkResult:
        df = self.load_frame()

        # Get the most recent data
        max_date = df["date"].max()
        df = df.filter(pl.col("date") == max_date)

        # Filter for our models
        df = self.map_and_filter_column(df, "model", mm)
        df = df.group_by("model", "provider").agg(pl.col("tps").mean())
        df = df.with_columns(
            [
                (pl.col("tps") - pl.col("tps").min())
                / (pl.col("tps").max() - pl.col("tps").min())
            ]
        )
        df = df.rename({"tps": "score", "provider": "context"})
        scores = [ModelScore.model_validate(dct) for dct in df.iter_rows(named=True)]
        return BenchmarkResult(bm_type=BenchmarkType.SPEED, scores=scores)
