import logging
from importlib import import_module
from typing import Protocol, cast, runtime_checkable

from beartype import beartype

logger = logging.getLogger(__name__)


# The simplest "plugin" system ever.
# Each module must conform to this protocol.
# We use beartype to check it.
@runtime_checkable
class BenchmarkProvider(Protocol):
    """Each of the modules in this folder should have the following..."""

    def download(self): ...
    def assemble_frame(self): ...


@beartype
def load_module(module_name: str) -> BenchmarkProvider:
    """We can load a module by name or by path."""
    module = import_module(f"evals.benchmarks.{module_name}")
    return cast(BenchmarkProvider, module)


PROVIDERS: dict[str, BenchmarkProvider] = {
    nm: load_module(nm) for nm in "lmsys litellm thefastestai".split()
}
