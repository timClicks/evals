"""Build the routing table."""

from itertools import product

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel

from .benchmarks import all_benchmarks
from .modelmap import ModelMapper
from .scoring import BenchmarkResult, BenchmarkType, ModelScore
from .settings import get_settings


async def load_stats(snapshot_id: int) -> pl.DataFrame:
    """Load the stats for a snapshot."""
    records = await LLMStatsRecord.filter(snapshot_id=snapshot_id).all()
    cols = "name cost speed quality".split()
    data = [[getattr(x, c) for c in cols] for x in records]
    df = pl.DataFrame(data, schema=cols)
    return df


async def build_grid(snapshot_id: int):
    df = await load_stats(snapshot_id)
    model_names = df.select("name").to_numpy().flatten()
    arr = df.select("cost speed quality".split()).to_numpy()

    # Define the range of weights (0 to 1 in steps of 0.1)
    weights = np.arange(0, 1.1, 0.1)

    # Function to calculate the weighted score for a model
    def calculate_score(model, weight):
        return np.dot(model, weight)

    # Store the results
    results = []

    # Iterate over all possible weight combinations that add to 1.0
    for w_cost, w_speed, w_quality in product(weights, repeat=3):
        if np.isclose(
            w_cost + w_speed + w_quality, 1.0
        ):  # Ensure the sum of weights is 1
            current_weights = np.array([w_cost, w_speed, w_quality])
            scores = np.apply_along_axis(calculate_score, 1, arr, current_weights)
            best = np.argmax(scores)
            results.append(
                (w_cost, w_speed, w_quality, model_names[best], scores[best])
            )

    return pl.DataFrame(results, schema="cost speed quality model score".split())


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

    # We lead with the quality scores.
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
    df.write_parquet(st.get_routing_dir() / "scoring.parquet")
    print(df)


if __name__ == "__main__":
    main()
