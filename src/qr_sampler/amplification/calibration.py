"""Amplifier calibration utilities for hardware QRNG devices.

Provides functions to measure the actual statistical properties of an
entropy source and estimate its entropy rate. Use these to calibrate the
z-score amplifier's ``population_mean`` and ``population_std`` for a
specific QRNG device whose byte distribution may differ from the
theoretical uniform [0, 255].
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qr_sampler.entropy.base import EntropySource


def calibrate_population_stats(
    source: EntropySource,
    n_samples: int = 10000,
    bytes_per_sample: int = 1024,
) -> dict[str, float]:
    """Measure actual population mean and std from an entropy source.

    Collects ``n_samples`` batches of ``bytes_per_sample`` bytes each,
    computes the per-byte mean and standard deviation across all collected
    data, and returns the results.

    Use this to calibrate the z-score amplifier for a specific QRNG device
    whose byte distribution may differ from the theoretical uniform [0, 255].

    Args:
        source: The entropy source to calibrate.
        n_samples: Number of sample batches to collect.
        bytes_per_sample: Bytes per batch.

    Returns:
        Dictionary with ``'mean'``, ``'std'``, ``'min'``, ``'max'``, and
        ``'n_bytes_total'``.
    """
    all_bytes: list[bytes] = []
    for _ in range(n_samples):
        all_bytes.append(source.get_random_bytes(bytes_per_sample))

    combined = b"".join(all_bytes)
    arr = np.frombuffer(combined, dtype=np.uint8).astype(np.float64)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_bytes_total": float(len(arr)),
    }


def measure_entropy_rate(
    source: EntropySource,
    n_bytes: int = 100_000,
) -> dict[str, float]:
    """Estimate entropy rate of an entropy source via compression.

    Fetches ``n_bytes`` from the source, compresses them with zlib at
    maximum compression level, and uses the compression ratio to estimate
    the bits of entropy per byte. A perfectly random source will have
    ~8.0 bits per byte (incompressible).

    The min-entropy estimate uses a conservative heuristic based on the
    most frequent byte value: ``-log2(max_frequency)``.

    Args:
        source: The entropy source to measure.
        n_bytes: Number of bytes to fetch for measurement.

    Returns:
        Dictionary with ``'bits_per_byte'`` (8.0 = maximum),
        ``'compression_ratio'``, and ``'estimated_min_entropy'``.
    """
    raw = source.get_random_bytes(n_bytes)
    compressed = zlib.compress(raw, level=9)
    compression_ratio = len(compressed) / len(raw)

    # Bits per byte: if compression ratio >= 1.0 (incompressible), the
    # source has ~8 bits/byte. Otherwise scale proportionally.
    bits_per_byte = min(8.0, compression_ratio * 8.0)

    # Min-entropy estimate from byte frequency distribution.
    arr = np.frombuffer(raw, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    max_freq = float(np.max(counts)) / len(arr)
    # -log2(max_freq) is the min-entropy per byte.
    estimated_min_entropy = float(-np.log2(max_freq)) if max_freq > 0 else 0.0

    return {
        "bits_per_byte": bits_per_byte,
        "compression_ratio": compression_ratio,
        "estimated_min_entropy": estimated_min_entropy,
    }
