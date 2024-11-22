"""Build the routing table."""

from loguru import logger

from .benchmarks import all_benchmarks
from .modelmap import ModelMapper


def build_routing():
    bm = all_benchmarks()
    mm = ModelMapper.load()

    for nm, benchmark in bm.items():
        logger.info(f"Processing benchmark {nm}")
        benchmark.get_benchmarks(mm)


def main():
    build_routing()
