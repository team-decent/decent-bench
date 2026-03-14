from __future__ import annotations

import contextlib
import random
from functools import cache
from types import ModuleType

import numpy as np

jax: ModuleType | None = None
jnp: ModuleType | None = None
tf: ModuleType | None = None
torch: ModuleType | None = None

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch as _torch

    torch = _torch

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import tensorflow as _tf

    tf = _tf

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax.numpy as _jnp

    jnp = _jnp

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax as _jax

    jax = _jax


_np_types = (np.ndarray, np.generic, float, int)
_torch_types = (torch.Tensor, float, int) if torch else (float,)
_tf_types = (tf.Tensor, float, int) if tf else (float,)
if jax and jnp:
    _jnp_array_types = tuple(
        dict.fromkeys(
            (
                jnp.ndarray,
                jnp.generic,
                type(jnp.array(0)),
                *( (jax.Array,) if hasattr(jax, "Array") else () ),
            )
        )
    )
else:
    _jnp_array_types = ()
_jnp_types = (*_jnp_array_types, float, int) if jnp else (float,)

_jax_key = jax.random.key(random.randint(0, 2**32 - 1)) if jax else None


@cache
def _numpy_generator() -> np.random.Generator:
    """Get a NumPy random number generator instance."""
    return np.random.default_rng()
