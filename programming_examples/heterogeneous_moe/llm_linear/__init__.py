# SPDX-License-Identifier: MIT

"""LLM-style linear-layer benchmark for Ryzen heterogeneous execution studies."""

from .manifest import SCHEMA_VERSION
from .reference import LinearConfig, LinearWeights

__all__ = ["SCHEMA_VERSION", "LinearConfig", "LinearWeights"]
