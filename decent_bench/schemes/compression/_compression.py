from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from decent_array import Array
from decent_array import interoperability as iop


class CompressionScheme(ABC):
    """Scheme defining how messages are compressed when sent over the network."""

    @abstractmethod
    def compress(self, msg: Array) -> Array:
        """Apply compression and return a new, compressed message."""

    def compressed_msg_size(self, msg: Array) -> int:
        """Compute the size of the compressed version of *msg*."""
        return int(np.prod(iop.shape(msg)))  # replace with msg.size once available


class NoCompression(CompressionScheme):
    """Scheme that leaves messages uncompressed."""

    def compress(self, msg: Array) -> Array:
        return msg


class Quantization(CompressionScheme):
    r"""
    Scheme applying uniform quantization to the message.

    Given a message :math:`x` and quantization step :math:`\Delta`, the scheme returns

        .. math:: q(x) = \Delta \operatorname{round}(x / \Delta)

    where :math:`\operatorname{round}(\cdot)` represents rounding to the nearest integer.

    Raises:
        ValueError: if ``quantization_step`` is not positive.

    """

    def __init__(self, quantization_step: float):
        if quantization_step <= 0:
            raise ValueError("`quantization_step` must be a positive float")
        self.quantization_step = quantization_step

    def compress(self, msg: Array) -> Array:
        msg_np = np.asarray(iop.to_numpy(msg), dtype=np.float64)
        compressed_msg = self.quantization_step * np.rint(msg_np / self.quantization_step)
        return iop.from_numpy_like(compressed_msg, msg)


class StochasticQuantization(CompressionScheme):
    r"""
    Stochastic quantization used in QSGD :footcite:p:`Scheme_QSGD`.

    The scheme quantizes each coordinate using ``n_levels`` stochastic levels scaled by the message norm. This keeps the
    compressed message unbiased in expectation while preserving the original message shape. Given a message
    :math:`x` and :math:`s=\texttt{n\_levels}`, the quantizer computes

    .. math::

        a_i = \frac{s |x_i|}{\lVert x \rVert_2}, \qquad
        \ell_i = \lfloor a_i \rfloor, \qquad
        p_i = a_i - \ell_i.

    The quantization level is sampled as

    .. math::

        \xi_i =
        \begin{cases}
            \ell_i + 1, & \text{with probability } p_i, \\
            \ell_i, & \text{with probability } 1 - p_i,
        \end{cases}

    and the compressed coordinate is

    .. math::

        Q_s(x_i) = \lVert x \rVert_2 \operatorname{sign}(x_i) \frac{\xi_i}{s}.

    Args:
        n_levels: number of stochastic quantization levels. Larger values give a finer quantization grid and usually
            lower quantization error. Smaller values give coarser quantization and stronger compression noise.

    Raises:
        ValueError: if ``n_levels`` is not positive.

    Warning:
        This scheme computes the :math:`\ell_2` norm of each message. This can be computationally expensive for large
        messages or when messages live on accelerator devices.

    .. footbibliography::

    """

    def __init__(self, n_levels: int):
        if n_levels <= 0:
            raise ValueError("`n_levels` must be a positive integer")
        self.n_levels = n_levels

    def compress(self, msg: Array) -> Array:
        msg_norm = float(iop.norm(msg))
        if msg_norm == 0:
            return iop.zeros_like(msg)

        msg_np = np.asarray(iop.to_numpy(msg), dtype=np.float64)
        magnitudes = np.abs(msg_np)
        signs = np.sign(msg_np)
        scaled_magnitudes = self.n_levels * magnitudes / msg_norm
        lower_levels = np.floor(scaled_magnitudes)
        probabilities = scaled_magnitudes - lower_levels
        quantized_levels = lower_levels + (iop.get_numpy_rng().random(size=magnitudes.shape) < probabilities)
        compressed_msg = msg_norm * signs * quantized_levels / self.n_levels
        return iop.from_numpy_like(compressed_msg, msg)


class TopK(CompressionScheme):
    """
    Top-k compression which transmits only a subset of elements with largest magnitude.

    The parameter ``k`` can be either:

    - an ``int``: transmit exactly ``k`` elements, or
    - a ``float`` in :math:`(0, 1]`: transmit a fraction ``k`` of elements.

    Message size is preserved by transmitting zeros in place of non-transmitted elements.

    Raises:
        ValueError: if ``k`` is a float and not in :math:`(0, 1]`
        ValueError: if ``k`` is an int and less than 1

    Note:
        If ``k * n_elements < 1``, at least one element is still transmitted.

    """

    def __init__(self, k: float):
        if isinstance(k, int):
            if k < 1:
                raise ValueError(f"If `k` is an integer, it must be at least 1, got {k}")
        elif k <= 0 or k > 1:
            raise ValueError(f"If `k` is a float, it must be in (0, 1], got {k}")
        self.k = k
        self.is_integer_k = isinstance(self.k, int)

    def compress(self, msg: Array) -> Array:
        msg_np = iop.to_numpy(msg)
        n_elements = msg_np.size
        k_count = min(int(self.k), n_elements) if self.is_integer_k else max(1, int(np.ceil(self.k * n_elements)))

        flat_msg = msg_np.reshape(-1)
        idx = np.argpartition(np.abs(flat_msg), -k_count)[-k_count:]
        compressed_flat = np.zeros_like(flat_msg)
        compressed_flat[idx] = flat_msg[idx]

        return iop.from_numpy_like(compressed_flat.reshape(msg_np.shape), msg)

    def compressed_msg_size(self, msg: Array) -> int:
        """Compute the size of the compressed version of *msg*."""
        return int(self.k if self.is_integer_k else np.ceil(self.k * np.prod(iop.shape(msg))))  # replace with msg.size


class RandK(CompressionScheme):
    """
    Rand-k compression which transmits only a random subset of elements.

    The parameter ``k`` can be either:

    - an ``int``: transmit exactly ``k`` elements chosen uniformly at random (without replacement), or
    - a ``float`` in :math:`(0, 1]`: transmit a fraction ``k`` of elements.

    Message size is preserved by transmitting zeros in place of non-transmitted elements.

    Raises:
        ValueError: if ``k`` is a float and not in :math:`(0, 1]`
        ValueError: if ``k`` is an int and less than 1

    Note:
        If ``k * n_elements < 1``, at least one element is still transmitted.

    """

    def __init__(self, k: float):
        if isinstance(k, int):
            if k < 1:
                raise ValueError(f"`k` must be at least 1 if an integer, got {k}")
        elif k <= 0 or k > 1:
            raise ValueError(f"`k` must be in (0, 1], got {k}")
        self.k = k
        self.is_integer_k = isinstance(self.k, int)

    def compress(self, msg: Array) -> Array:
        msg_np = iop.to_numpy(msg)
        n_elements = msg_np.size
        k_count = min(int(self.k), n_elements) if self.is_integer_k else max(1, int(np.ceil(self.k * n_elements)))

        flat_msg = msg_np.reshape(-1)
        idx = iop.get_numpy_rng().choice(n_elements, size=k_count, replace=False)
        compressed_flat = np.zeros_like(flat_msg)
        compressed_flat[idx] = flat_msg[idx]

        return iop.from_numpy_like(compressed_flat.reshape(msg_np.shape), msg)

    def compressed_msg_size(self, msg: Array) -> int:
        """Compute the size of the compressed version of *msg*."""
        return int(self.k if self.is_integer_k else np.ceil(self.k * np.prod(iop.shape(msg))))  # replace with msg.size
