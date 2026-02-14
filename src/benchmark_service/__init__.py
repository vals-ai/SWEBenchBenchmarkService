"""Benchmark service framework for creating evaluation APIs."""

from benchmark_service.app import create_app
from benchmark_service.base import BenchmarkService

__all__ = [
    "BenchmarkService",
    "create_app",
]
