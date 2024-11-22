"""Build the scores."""

import polars as pl
from loguru import logger
from pydantic import BaseModel

from .benchmarks import all_benchmarks
from .modelmap import ModelMapper
from .orm import BenchmarkResult, BenchmarkType, ModelScore
from .settings import get_settings


def get_scores_like(
    results: list[BenchmarkResult], bm_type: BenchmarkType
) -> list[ModelScore]:
    chosen = [r for r in results if r.bm_type == bm_type]
    # There should only be one quality score.
    match len(chosen):
        case 1:
            return chosen[0].scores

        case 0:
            msg = "No quality score found."
            raise ValueError(msg)

        case _:
            msg = "Multiple quality scores found."
            raise ValueError(msg)


class ScoreRecord(BaseModel):
    category: str
    quality: float
    cost: float
    speed: float
    model: str


def generate_scoring_frame() -> pl.DataFrame:
    mm = ModelMapper.load()
    results = [bm.get_benchmarks(mm) for bm in all_benchmarks().values()]
    qscores = get_scores_like(results, BenchmarkType.QUALITY)
    cscores = get_scores_like(results, BenchmarkType.COST)
    sscores = get_scores_like(results, BenchmarkType.SPEED)

    score_records = []

    # We lead with the quality scores, adding the cost and speed scores.
    for qscore in qscores:
        model = qscore.model

        # Attach the speed and cost.
        cost = next((s.score for s in cscores if s.model == model), None)
        speed = next((s.score for s in sscores if s.model == model), None)
        if cost is None or speed is None:
            logger.warning(f"Skipping {model} due to missing cost or speed score.")
            continue

        score_records.append(
            ScoreRecord(
                category=qscore.context,
                quality=qscore.score,
                cost=cost,
                speed=speed,
                model=model,
            )
        )

    df = pl.DataFrame(score_records)
    return df


def main():
    st = get_settings()
    df = generate_scoring_frame()
    logger.info(f"Writing scoring frame to {st.get_routing_dir()}.")
    df.write_parquet(st.get_routing_dir() / "scoring.parquet")


if __name__ == "__main__":
    main()
