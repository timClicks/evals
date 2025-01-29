"""Build the scores.

For each model and category, we have a quality, cost, and speed score.
Together these can be used to generate the ideal models, given a weighting.

"""

# TODO: for now, we ignore *hosts*, that a model can be hosted by one or more
# platforms. We can add this later.

from collections import defaultdict
import csv
import os
from datetime import date
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from pydantic import BaseModel, RootModel

from .benchmarks import all_benchmarks
from .modelmap import ModelMapper
from .orm import Benchmark, BenchmarkResult, BenchmarkType
from .settings import get_settings


def get_scores_like(
    results: list[BenchmarkResult], bm_type: BenchmarkType
) -> tuple[str, list[Benchmark]]:
    chosen = [r for r in results if r.bm_type == bm_type]
    # There should only be one quality score.
    match len(chosen):
        case 1:
            return chosen[0].unit, chosen[0].scores

        case 0:
            msg = "No quality score found."
            raise ValueError(msg)

        case _:
            msg = "Multiple quality scores found."
            raise ValueError(msg)


class ScoreRecord(BaseModel):
    model: str

    # TODO: We should use an enum here later.
    category: str

    quality: float
    quality_rank: None | int = None
    quality_pct: None | float = None
    quality_unit: str

    cost: float
    cost_rank: None | int = None
    cost_pct: None | float = None
    cost_unit: str

    speed: float
    speed_rank: None | int = None
    speed_pct: None | float = None
    speed_unit: str


class ScoreRecordList(RootModel):
    root: list[ScoreRecord]


def calculate_geometric_means(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    group_by = "category"
    cols = list(cols)

    def geom_mean(row: tuple[float]):
        row = np.array(row, dtype=np.float64)  # type: ignore
        return np.exp(np.mean(np.log(row)))

    df_overall = df.with_columns(
        [
            pl.struct(cols)
            .map_elements(
                lambda x: geom_mean(tuple(x.values())), return_dtype=pl.Float64
            )
            .alias("geometric_avg_rank")
        ]
    )

    df_by_category = df.group_by(group_by).map_groups(
        lambda group: group.with_columns(
            [
                pl.struct(cols)
                .map_elements(
                    lambda x: geom_mean(tuple(x.values())), return_dtype=pl.Float64
                )
                .alias("geometric_avg_rank_by_category")
            ]
        )
    )

    overall = df_overall.get_column("geometric_avg_rank")
    by_category = df_by_category.get_column("geometric_avg_rank_by_category")

    df = df.with_columns([overall, by_category])

    return df


def rank_by_category(df: pl.DataFrame) -> pl.DataFrame:
    rank_qual = (
        pl.col("quality")
        .rank(method="min", descending=True)
        .over("category")
        .alias("quality_rank")
    )
    rank_cost = (
        pl.col("cost")
        .rank(method="min", descending=False)
        .over("category")
        .alias("cost_rank")
    )
    rank_speed = (
        pl.col("speed")
        .rank(method="min", descending=True)
        .over("category")
        .alias("speed_rank")
    )

    df = df.with_columns([rank_qual, rank_cost, rank_speed])

    return df


def generate_scores() -> ScoreRecordList:
    mm = ModelMapper.load()
    results = [bm.get_benchmarks(mm) for bm in all_benchmarks().values()]
    q_unit, quality_scores = get_scores_like(results, BenchmarkType.QUALITY)
    c_unit, cost_scores = get_scores_like(results, BenchmarkType.COST)
    s_unit, speed_scores = get_scores_like(results, BenchmarkType.SPEED)

    metrics = ["quality", "cost", "speed"]
    # contexts = set() # ignore for now
    model_names = set()
    model_percentiles = defaultdict(dict)

    metric_observations = {
        "quality": { "model": [], "score": [] },
        "cost": { "model": [], "score": [] },
        "speed": { "model": [], "score": [] },
    }

    quality_of_model = {}
    for score in quality_scores:
        quality_of_model[score.model] = score.score
        model_names.add(score.model)
        metric_observations["quality"]["model"].append(score.model)
        metric_observations["quality"]["score"].append(score.score)

    cost_of_model = {}
    for score in cost_scores:
        cost_of_model[score.model] = score.score
        model_names.add(score.model)
        metric_observations["cost"]["model"].append(score.model)
        metric_observations["cost"]["score"].append(score.score)

    speed_of_model = {}
    for score in speed_scores:
        speed_of_model[score.model] = score.score
        model_names.add(score.model)
        metric_observations["speed"]["model"].append(score.model)
        metric_observations["speed"]["score"].append(score.score)

    for metric, scores in metric_observations.items():
        model_names = scores["model"]
        obs = np.array(scores["score"])

        # lower is better for cost
        # TODO: check whether this is already done earlier
        if metric == "cost":
            obs = 1.0 - obs
        
        sorted_obs = np.sort(obs)
        positions = np.searchsorted(sorted_obs, obs)
        percentiles = positions / len(obs)

        stars = percentiles * 5.0 # Convert to 0-1 to 0-5 scale
        stars = np.round(stars * 2) / 2 # Round to nearest 0.5

        for (model, score, pct) in zip(model_names, obs, percentiles):
            pct = pct * 100.0
            model_percentiles[metric].update(**{model: pct})

    score_records = []
    
    for model in model_names:
        quality = quality_of_model.get(model)
        cost = cost_of_model.get(model)
        speed = speed_of_model.get(model)

    # We lead with the quality scores, adding the cost and speed scores.
    for quality_score in quality_scores:
        model = quality_score.model

        try:
            quality = quality_score.score
            cost = cost_of_model[model]
            speed = speed_of_model[model]
        except LookupError:
            fields = ["quality", "cost", "speed"]
            present = [ model in quality_of_model, model in cost_of_model, model in speed_of_model ]
            missing = [field for (field, is_present) in zip(fields, present) if not is_present]

            match len(missing):
                case 1: logger.warning(f"Skipping {model}, missing: {missing[0]}.")
                case 2: logger.warning(f"Skipping {model}, missing: {missing[0]} and {missing[1]}.")
                case _: logger.warning(f"Skipping {model}, missing: {(', ').join(missing[0:-1])} and {missing[-1]}.")
            
            continue

        score_records.append(
            ScoreRecord(
                category=quality_score.context,
                quality=quality,
                quality_unit=q_unit,
                quality_pct=model_percentiles["quality"][model],
                cost=cost,
                cost_unit=c_unit,
                cost_pct=model_percentiles["cost"][model],
                speed=speed,
                speed_unit=s_unit,
                speed_pct=model_percentiles["speed"][model],
                model=model,
            )
        )

    return ScoreRecordList(root=score_records)


# utility functions


def today():
    return date.today().strftime("%Y-%m-%d")


# disk-related


def symlink(original: Path):
    parent_dir = original.parent
    symlink_path = parent_dir / f"scoring{original.suffix}"

    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    os.symlink(original.name, symlink_path)


def main():
    """Write parquet and JSON files with scoring data."""
    st = get_settings()
    home = st.get_routing_dir()

    score_card = generate_scores()
    df = pl.DataFrame(score_card.root)
    df = rank_by_category(df)
    df = calculate_geometric_means(df, ["quality_rank", "cost_rank", "speed_rank"])

    df.write_csv(home / f"scoring-{today()}.csv")
    df.write_json(home / f"scoring-{today()}.json")
    df.write_parquet(home / f"scoring-{today()}.parquet")

    # update "scoring.(csv|json|parquet)" to always point to the most recent version
    symlink(home / f"scoring-{today()}.csv")
    symlink(home / f"scoring-{today()}.json")
    symlink(home / f"scoring-{today()}.parquet")


    print(df)

if __name__ == "__main__":
    main()
