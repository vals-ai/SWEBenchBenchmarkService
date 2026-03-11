"""SWE-bench Benchmark Service."""

from benchmark_service import BenchmarkServiceApp
from swebench_service.benchmark_service import SWEBenchService

app = BenchmarkServiceApp(SWEBenchService)
