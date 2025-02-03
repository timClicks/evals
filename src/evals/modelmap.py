"""Generate candidate matches between Stencila model names and other sources."""

import json
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Self

import polars as pl
from pydantic import BaseModel

from .settings import get_settings

TABLES_DIR = Path(__file__).parent.parent.parent / "tables"
MODELS_PATH = TABLES_DIR / "model.csv"
KEY_COLUMN = "id"


class ModelMapper(BaseModel):
    """Map Stencila model names to other sources.

    This turns the CSV into something more pythonic and easier to use.
    """

    models: set[str]
    mappings: dict[str, dict[str, str]]

    def mapping_for(self, benchmark: str):
        return self.mappings[benchmark]

    @classmethod
    def load(cls) -> Self:
        df = pl.read_csv(MODELS_PATH)
        # Grab those we'll actually use and drop the use column.
        df = df.filter(pl.col("use") == "*")
        df.drop_in_place("use")

        # We get an entry for
        columns: dict[str, list[str]] = {
            nm: series.to_list() for (nm, series) in df.to_dict().items()
        }

        mappings = {}
        for nm in columns:
            if nm == KEY_COLUMN:
                continue
            bm_to_stencila = [
                # We need to translate the '=' to the key.
                (a if a != "=" else b, b)
                for (a, b) in zip(columns[nm], columns[KEY_COLUMN], strict=True)
            ]
            mapping = dict(bm_to_stencila)
            mappings[nm] = mapping

        # supplement CSV w/ 1:1 file
        model_mapping_path = get_settings().get_base_dir() / ".." / "tables" / "model-id-mapping.json"

        try:
            with open(model_mapping_path) as fd:
                all_model_names = json.load(fd)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            all_model_names = dict()

        for model_id_with_source, stencila_model_id in all_model_names.items():
            if stencila_model_id is None:
                continue
            try:
                model_id, source = model_id_with_source.rsplit(' ', 1)
            except ValueError as err:
                print(model_id_with_source)
                raise err
            source = source.strip('[]')
            if source == "lmarena":
                source = "lmsys"
            mappings[source][model_id] = stencila_model_id
            mappings[source][model_id.lower()] = stencila_model_id

            if source == "litellm":
                model_id = model_id.split("/")[-1]
                mappings[source][model_id] = stencila_model_id
                mappings[source][model_id.lower()] = stencila_model_id

        return cls(models=set(columns[KEY_COLUMN]), mappings=mappings)


# BELOW is some code for trying to match names -----------------------------------
# It can be used to kickstart some manual matching.
def load_model_names() -> dict[str, set[str]]:
    frames_dir = get_settings().get_frames_dir()
    names = {}
    for sources in ("lmsys", "litellm", "thefastestai"):
        df = pl.read_parquet(frames_dir / f"{sources}.parquet")
        names[sources] = set(df["model"].to_list())

    return names


def load_stencila_names() -> dict[str, str]:
    df = pl.read_csv(MODELS_PATH)
    return dict(zip(df[KEY_COLUMN].to_list(), df["use"].to_list(), strict=True))


def match_model(model: str, candidates: set[str], cutoff: float) -> str:
    # * means identical
    if model in candidates:
        return "="

    closest = get_close_matches(model, candidates, n=1, cutoff=cutoff)
    if not closest:
        return ""
    return closest[0]


CUTOFF = 0.6


def main():
    """This keeps the first two columns and overwrites with best guesses."""

    model_names = load_model_names()
    stencila_names = load_stencila_names()

    dct: defaultdict[str, list[str]] = defaultdict(list)
    for model, use in stencila_names.items():
        dct["use"].append(use)
        dct["stencila"].append(model)
        for source, candidates in model_names.items():
            best = match_model(model, candidates, CUTOFF)
            dct[source].append(best)

    df = pl.DataFrame(dct)
    df.write_csv(MODELS_PATH)
