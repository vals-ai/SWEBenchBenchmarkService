"""SWE-bench Benchmark Service."""

from benchmark_service.vals import ValsBenchmarkServiceApp as BenchmarkServiceApp

from swebench_service.benchmark_service import SWEBenchService

app = BenchmarkServiceApp(SWEBenchService)
