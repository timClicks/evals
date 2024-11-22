import argparse
import asyncio
from enum import StrEnum
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from .modelmap import ModelMapper
from .orm import PROMPT_CSV, Database, ModelRecord, PromptRecord


class BenchmarkType(StrEnum):
    """Kind of score."""

    COST = "cost"
    SPEED = "speed"
    QUALITY = "quality"


class ModelScore(BaseModel):
    """Model score."""

    # Model name. Must match the name in the model table.
    model: str

    # This could be:
    # - a category if it is quality score.
    # - a host if it is cost or speed score.
    # - "*" is a catchall that indicates everything (all hosts or any category).
    context: str = "*"

    # Normalized to 0-1
    score: float = Field(ge=0.0, le=1.0)


class BenchmarkResult(BaseModel):
    """Benchmark result."""

    bm_type: BenchmarkType
    scores: list[ModelScore]


# Generate schemas
async def generate_scores(path: Path):
    prompt_df = pl.read_csv(PROMPT_CSV.resolve())
    mm = ModelMapper.load()

    async with Database(path):
        for row_dict in prompt_df.iter_rows(named=True):
            prompt = PromptRecord(**row_dict)
            await prompt.save()
        for model in mm.models:
            rec = ModelRecord(id=model)
            await rec.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", help="sqlite file to output")
    args = parser.parse_args()
    pth = Path(args.database)
    if pth.exists():
        pth.unlink()
    asyncio.run(generate_scores(pth))


if __name__ == "__main__":
    main()
