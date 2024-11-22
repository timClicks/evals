from enum import StrEnum

from pydantic import BaseModel, Field


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
