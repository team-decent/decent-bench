"""Message compression schemes."""

from decent_bench.schemes.compression._compression import (
    CompressionScheme,
    NoCompression,
    Quantization,
    RandK,
    StochasticQuantization,
    TopK,
)

__all__ = [
    "CompressionScheme",
    "NoCompression",
    "Quantization",
    "RandK",
    "StochasticQuantization",
    "TopK",
]
