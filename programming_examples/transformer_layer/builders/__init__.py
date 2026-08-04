# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Transformer-block operator builders for the execution studies port.

One ``build_<name>_module(...)`` function per operator, each returning an
``air.ir.Module``, with its FP32 reference as a module-level function in the
same file (porting convention 4). There is no operator class and no
``op.py``/``design.py`` pair -- see
``docs/plans/transformer-layer-execution-studies/02-porting-conventions.md``.

WHAT LIVES HERE AND WHAT DOES NOT
    Builders that are specific to the transformer block being studied. The
    architecture-orthogonal builders stay in ``llms/shared/builders/`` and are
    *called* from here, never modified -- editing ``shared/`` triggers the
    ten-model ``make verify`` regression rule.

NUMERICS
    Every reference here computes in float32 from bf16-rounded inputs, and
    every check runs through ``opcheck.py`` at ``rtol=1.6e-2`` with an ``atol``
    sized to the kernel's measured worst-case absolute error. That is the
    ``kernel_registry`` methodology; it is deliberately NOT iron's bf16
    reference with a 0.5% mismatch budget.
"""
