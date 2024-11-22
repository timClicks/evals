from ._benchmark import all_benchmarks

# We import here as it forces the registry to be populated.
from .litellm import LiteLLM
from .lmsys import LmSys
from .thefastestai import TheFastestAI

__all__ = ["LiteLLM", "LmSys", "TheFastestAI", "all_benchmarks"]
