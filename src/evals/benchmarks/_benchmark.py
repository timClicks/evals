import logging
from abc import ABC, abstractmethod
from functools import cache

import polars as pl
from pydantic import BaseModel

from evals.modelmap import ModelMapper
from evals.scoring import BenchmarkResult
from evals.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type["_Benchmark"]] = {}


class _Benchmark(BaseModel, ABC):
    settings: Settings

    @classmethod
    def registry_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Add ourselves to the registry."""
        super().__pydantic_init_subclass__(**kwargs)
        _REGISTRY[cls.registry_name()] = cls

    @property
    def frame_path(self):
        return get_settings().get_frames_dir() / f"{self.registry_name()}.parquet"

    def map_and_filter_column(
        self,
        df: pl.DataFrame,
        column: str,
        mm: ModelMapper,
    ) -> pl.DataFrame:
        mapping = mm.mapping_for(self.registry_name())
        return df.filter(pl.col(column).is_in(mapping.keys())).with_columns(
            pl.col(column).replace_strict(mapping)
        )

    def load_frame(self) -> pl.DataFrame:
        return pl.read_parquet(self.frame_path)

    @abstractmethod
    def get_benchmarks(self, mm: ModelMapper) -> BenchmarkResult: ...


@cache
def all_benchmarks() -> dict[str, _Benchmark]:
    st = get_settings()
    return {nm: cls(settings=st) for nm, cls in _REGISTRY.items()}


# @beartype
# def load_module(module_name: str) -> BenchmarkProvider:
#     """We can load a module by name or by path."""
#     module = import_module(f"evals.benchmarks.{module_name}")
#     return cast(BenchmarkProvider, module)
#
#
# PROVIDERS: dict[str, BenchmarkProvider] = {
#     nm: load_module(nm) for nm in "lmsys litellm thefastestai".split()
# }
