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
    diag,
    eye,
    eye_like,
    get_item,
    max,  # noqa: A004
    mean,
    min,  # noqa: A004
    norm,
    ones_like,
    reshape,
    set_item,
    shape,
    squeeze,
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
from ._helpers import device_to_framework_device, framework_device_of_array
from ._operators import (
    absolute,
    add,
    div,
    dot,
    matmul,
    maximum,
    mul,
    negative,
    power,
    sign,
    sqrt,
    sub,
)
from ._rng import (
    get_numpy_generator,
    get_rng_state,
    get_seed,
    get_tensorflow_generator,
    get_torch_generator,
    get_next_jax_key,
    rand,
    rand_like,
    randn,
    randn_like,
    set_rng_state,
    set_seed,
)

__all__ = [  # noqa: RUF022
    # From _functions
    "argmax",
    "argmin",
    "astype",
    "copy",
    "diag",
    "eye",
    "eye_like",
    "get_item",
    "max",
    "mean",
    "min",
    "norm",
    "ones_like",
    "reshape",
    "set_item",
    "shape",
    "squeeze",
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
    "maximum",
    "mul",
    "negative",
    "power",
    "sign",
    "sqrt",
    "sub",
    # From _helpers
    "device_to_framework_device",
    "framework_device_of_array",
    # From _decorators
    "autodecorate_cost_method",
    # Extensions
    "ext",
    # RNG manager
    "get_numpy_generator",
    "get_rng_state",
    "get_seed",
    "get_tensorflow_generator",
    "get_torch_generator",
    "set_rng_state",
    "set_seed",
    "rand_like",
    "rand",
    "randn",
    "randn_like",
    "get_next_jax_key",
]
