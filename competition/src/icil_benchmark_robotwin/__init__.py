"""RoboTwin ICIL, as a benchmark the ICIL competition can score on.

A separate distribution for the same reason `policies/` is one: the benchmark core stays a
benchmark and learns nothing about duels, kings or stores, and this moves at the competition's
pace rather than the benchmark's.

It never imports `icilval`. A benchmark that stands on its own cannot depend on the competition
that scores it, so the contract between them is duck-typed and carries a version.

Everything that needs a simulator is behind a function-local import or, better, behind an argv
this package hands back: the orchestrator host has no SAPIEN, no assets and no GPU, and must be
able to import this to read a catalogue or verify a prompt.
"""

from .plugin import BENCHMARK, BENCHMARK_API_VERSION, Benchmark
from .units import BenchmarkUnit

__all__ = ["BENCHMARK", "BENCHMARK_API_VERSION", "Benchmark", "BenchmarkUnit"]
