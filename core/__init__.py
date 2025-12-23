"""Core algorithms for entanglement optimization and fast rod packing.

This package contains:
- fast_packing: Non-PBC random rod placement with Numba acceleration.
- transforms, utils, visualizations: Helper utilities.

You can run the fast packing benchmark CLI with:
    python -m core.fast_packing --help
or, if installed as a package, via the console script:
    entangle-pack-npbc --help
"""

__all__ = [
    "fast_packing",
]
