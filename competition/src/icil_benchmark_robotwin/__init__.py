"""The RoboTwin ICIL benchmark as a plugin of the RoboTensor ICIL competition orchestrator.

The orchestrator finds it through the `icil.benchmarks` entry point `robotwin`, which loads
`BENCHMARK`. It never imports `icil_orchestrator`: the orchestrator checks it structurally.
"""

from .plugin import BENCHMARK, VERSION, RoboTwinBenchmark

__version__ = VERSION

__all__ = ["BENCHMARK", "RoboTwinBenchmark", "__version__"]
