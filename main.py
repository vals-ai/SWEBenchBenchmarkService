"""SWE-bench Benchmark Service."""

from benchmark_service import create_app
from swebench_service.benchmark_service import SWEBenchService

app = create_app(SWEBenchService())
