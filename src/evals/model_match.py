"""Generate candidate matches between Stencila model names and other sources."""

from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path

import polars as pl

from .settings import get_settings

TABLES_DIR = Path(__file__).parent.parent.parent / "tables"
CUTOFF = 0.6


def load_model_names() -> dict[str, set[str]]:
    frames_dir = get_settings().get_frames_dir()
    names = {}
    for sources in ("lmsys", "litellm", "thefastestai"):
        df = pl.read_parquet(frames_dir / f"{sources}.parquet")
        names[sources] = set(df["model"].to_list())

    return names


def load_stencila_names() -> dict[str, str]:
    df = pl.read_csv(TABLES_DIR / "model.csv")
    return dict(zip(df["stencila"].to_list(), df["use"].to_list(), strict=True))


def match_model(model: str, candidates: set[str]) -> str:
    # * means identical
    if model in candidates:
        return "="

    closest = get_close_matches(model, candidates, n=1, cutoff=CUTOFF)
    if not closest:
        return ""
    return closest[0]


def main():
    """This keeps the first two columns and overwrites with best guesses."""

    model_names = load_model_names()
    stencila_names = load_stencila_names()

    dct: defaultdict[str, list[str]] = defaultdict(list)
    for model, use in stencila_names.items():
        dct["use"].append(use)
        dct["stencila"].append(model)
        for source, candidates in model_names.items():
            best = match_model(model, candidates)
            dct[source].append(best)

    df = pl.DataFrame(dct)
    df.write_csv(TABLES_DIR / "model.csv")
