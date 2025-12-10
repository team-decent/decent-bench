"""
Utilities for operating on arrays from different deep learning and linear algebra frameworks.

Mirrors NumPy's functionality for interoperability across frameworks.
"""

from __future__ import annotations

from . import _ext as ext
from ._decorators import autodecorate_cost_method
from ._functions import (
    argmax,
    argmin,
    astype,
    copy,
    eye,
    eye_like,
    get_item,
    max,  # noqa: A004
    mean,
    min,  # noqa: A004
    norm,
    ones_like,
    rand_like,
    randn_like,
    reshape,
    set_item,
    shape,
    stack,
    sum,  # noqa: A004
    to_array,
    to_array_like,
    to_jax,
    to_numpy,
    to_tensorflow,
    to_torch,
    transpose,
    zeros,
    zeros_like,
)
from ._helpers import framework_device_of_array
from ._operators import (
    absolute,
    add,
    div,
    dot,
    matmul,
    mul,
    negative,
    power,
    sqrt,
    sub,
)

__all__ = [  # noqa: RUF022
    # From _functions
    "argmax",
    "argmin",
    "astype",
    "copy",
    "eye",
    "eye_like",
    "get_item",
    "max",
    "mean",
    "min",
    "norm",
    "ones_like",
    "rand_like",
    "randn_like",
    "reshape",
    "set_item",
    "shape",
    "stack",
    "sum",
    "to_array",
    "to_array_like",
    "to_numpy",
    "to_torch",
    "to_tensorflow",
    "to_jax",
    "transpose",
    "zeros",
    "zeros_like",
    # From _operators
    "absolute",
    "add",
    "div",
    "dot",
    "matmul",
    "mul",
    "negative",
    "power",
    "sqrt",
    "sub",
    # From _helpers
    "framework_device_of_array",
    # From _decorators
    "autodecorate_cost_method",
    # Extensions
    "ext",
]
