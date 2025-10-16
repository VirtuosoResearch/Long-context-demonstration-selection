"""
Data utilities mirrored from rllm.

The Dataset and DatasetRegistry classes are re-exported so that downstream
code can keep using the same abstractions while living under the mta namespace.
"""

from .dataset import Dataset, DatasetRegistry

__all__ = ["Dataset", "DatasetRegistry"]
