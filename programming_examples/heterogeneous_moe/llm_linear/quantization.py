# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

QuantKind = Literal["int4", "uint4"]


@dataclass(frozen=True)
class PackedWeightMetadata:
    quant_kind: QuantKind
    shape: tuple[int, int]
    block_size: int
    quant_axis: int
    signed: bool
    packing: str = "two_values_per_byte_low_nibble_first"


def pack_4bit(values: np.ndarray, *, signed: bool = False) -> np.ndarray:
    ints = np.asarray(values, dtype=np.int16).reshape(-1)
    if signed:
        if np.any(ints < -8) or np.any(ints > 7):
            raise ValueError("signed int4 values must be in [-8, 7]")
        nibbles = (ints & 0xF).astype(np.uint8)
    else:
        if np.any(ints < 0) or np.any(ints > 15):
            raise ValueError("uint4 values must be in [0, 15]")
        nibbles = ints.astype(np.uint8)
    if nibbles.size % 2:
        nibbles = np.concatenate((nibbles, np.zeros(1, dtype=np.uint8)))
    low = nibbles[0::2]
    high = nibbles[1::2] << np.uint8(4)
    return np.ascontiguousarray(low | high)


def unpack_4bit(packed: np.ndarray, *, count: int, signed: bool = False) -> np.ndarray:
    data = np.asarray(packed, dtype=np.uint8).reshape(-1)
    low = data & np.uint8(0xF)
    high = (data >> np.uint8(4)) & np.uint8(0xF)
    unpacked = np.empty(data.size * 2, dtype=np.int8 if signed else np.uint8)
    if signed:
        low_signed = np.where(low >= 8, low.astype(np.int8) - 16, low.astype(np.int8))
        high_signed = np.where(
            high >= 8, high.astype(np.int8) - 16, high.astype(np.int8)
        )
        unpacked[0::2] = low_signed
        unpacked[1::2] = high_signed
    else:
        unpacked[0::2] = low
        unpacked[1::2] = high
    return np.ascontiguousarray(unpacked[:count])


def metadata_for_packed_weights(
    *,
    quant_kind: QuantKind,
    shape: tuple[int, int],
    block_size: int,
    quant_axis: int,
) -> PackedWeightMetadata:
    if quant_kind not in {"int4", "uint4"}:
        raise ValueError(f"Unsupported quant_kind: {quant_kind}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if quant_axis not in {0, 1}:
        raise ValueError("quant_axis must be 0 or 1")
    return PackedWeightMetadata(
        quant_kind=quant_kind,
        shape=shape,
        block_size=block_size,
        quant_axis=quant_axis,
        signed=quant_kind == "int4",
    )
