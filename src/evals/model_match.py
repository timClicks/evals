"""Generate candidate matches between Stencila model names and other sources."""

from difflib import get_close_matches
from pathlib import Path

import polars as pl

from .settings import get_settings

TABLES_DIR = Path(__file__).parent.parent.parent / "tables"
MATCHES = 1
CUTOFF = 0.6


def load_model_names() -> dict[str, set[str]]:
    frames_dir = get_settings().get_frames_dir()
    names = {}
    for sources in ("lmsys", "litellm", "thefastestai"):
        df = pl.read_parquet(frames_dir / f"{sources}.parquet")
        names[sources] = set(df["model"].to_list())

    return names


def load_stencila_names() -> set[str]:
    df = pl.read_csv(TABLES_DIR / "model.csv")
    return set(df["identifier"].to_list()) - {"router"}


def match_models(from_set: set[str], to_set: set[str]) -> pl.DataFrame:
    entries: list[tuple[str, ...]] = []

    for item in from_set:
        closest = get_close_matches(item, to_set, n=MATCHES, cutoff=CUTOFF)
        while len(closest) < MATCHES:
            closest.append("")
        entries.append((item, *closest))

    return pl.DataFrame(
        entries,
        schema=["stencila", "match"] + [f"maybe_{n + 1}" for n in range(MATCHES - 1)],
        orient="row",
    )


def main():
    model_names = load_model_names()
    stencila_names = load_stencila_names()

    for source, names in model_names.items():
        matches = match_models(stencila_names, names)
        matches.write_csv(TABLES_DIR / f"{source}_candidate.csv")
