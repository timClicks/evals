"""Build the scores.

For each model and category, we have a quality, cost, and speed score.
Together these can be used to generate the ideal models, given a weighting.

"""
# TODO: for now, we ignore *hosts*, that a model can be hosted by one or more
# platforms. We can add this later.

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
    quality_unit: str
    cost: float
    cost_unit: str
    speed: float
    speed_unit: str


class ScoreRecordList(RootModel):
    root: list[ScoreRecord]


def generate_scores() -> ScoreRecordList:
    mm = ModelMapper.load()
    results = [bm.get_benchmarks(mm) for bm in all_benchmarks().values()]
    q_unit, quality_scores = get_scores_like(results, BenchmarkType.QUALITY)
    c_unit, cost_scores = get_scores_like(results, BenchmarkType.COST)
    s_unit, speed_scores = get_scores_like(results, BenchmarkType.SPEED)

    cost_of_model = {}
    for score in cost_scores:
        cost_of_model[score.model] = score.score

    speed_of_model = {}
    for score in speed_scores:
        speed_of_model[score.model] = score.score

    score_records = []

    # We lead with the quality scores, adding the cost and speed scores.
    for quality_score in quality_scores:
        model = quality_score.model
        try:
            cost = cost_of_model[model]
            speed = speed_of_model[model]
        except LookupError:
            logger.warning(f"Skipping {model} due to missing cost or speed score.")
            continue

        score_records.append(
            ScoreRecord(
                category=quality_score.context,
                quality=quality_score.score,
                quality_unit=q_unit,
                cost=cost,
                cost_unit=c_unit,
                speed=speed,
                speed_unit=s_unit,
                model=model,
            )
        )

    return ScoreRecordList(root=score_records)


def main():
    """Write parquet and JSON files with scoring data."""
    st = get_settings()
    sc = generate_scores()
    logger.info(f"Writing scoring frame to {st.get_routing_dir()}.")
    fpath = st.get_routing_dir() / "scoring.json"
    fpath.write_text(sc.model_dump_json(indent=2))
    pl.DataFrame(sc.root).write_parquet(st.get_routing_dir() / "scoring.parquet")


if __name__ == "__main__":
    main()
