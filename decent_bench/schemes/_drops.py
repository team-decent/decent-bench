from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np
from decent_array import interoperability as iop


class DropScheme(ABC):
    """Scheme defining how message drops occur over the network."""

    @abstractmethod
    def should_drop(self) -> bool:
        """Whether or not to drop."""


class NoDrops(DropScheme):
    """Scheme that never drops messages."""

    def should_drop(self) -> bool:
        return False


class UniformDropRate(DropScheme):
    """
    Scheme that drops messages with uniform probability.

    Each call samples an independent Bernoulli event with probability ``drop_rate``.

    Args:
        drop_rate: probability that a message is dropped.

    Raises:
        ValueError: if ``drop_rate`` is not in :math:`[0, 1]`.

    """

    def __init__(self, drop_rate: float):
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("Drop rate must be in [0, 1]")
        self.drop_rate = drop_rate

    def should_drop(self) -> bool:
        return random.random() < self.drop_rate


class GilbertElliott(DropScheme):
    """
    Drop scheme based on the Gilbert-Elliott model :footcite:p:`Scheme_GilbertElliott`.

    The Gilbert-Elliott model is characterized by a Markov chain with two states (good and bad), which
    can stay the same or transition into each other. In the bad state message drops occur with probability
    `drop_rate`, while in the good state no message drops occur.

    Args:
        drop_rate: message drop rate while in the bad state
        bad_to_good: transition probability from bad to good state
        good_to_bad: transition probability from good to bad state

    Raises:
        ValueError: if `drop_rate`, `bad_to_good` or `good_to_bad` are not in :math:`[0, 1]`

    .. footbibliography::

    """

    def __init__(self, drop_rate: float, bad_to_good: float = 0.5, good_to_bad: float = 0.5):
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("Drop rate must be in [0, 1]")
        if (bad_to_good < 0 or bad_to_good > 1) or (good_to_bad < 0 or good_to_bad > 1):
            raise ValueError("Transition probabilities `bad_to_good` and `good_to_bad` must be in [0, 1]")
        self.drop_rate = drop_rate
        self.bad_to_good = bad_to_good
        self.good_to_bad = good_to_bad
        self._states = np.array([0, 1])  # good = 0, bad = 1
        self._P = np.array([[1 - good_to_bad, good_to_bad], [bad_to_good, 1 - bad_to_good]])  # transition matrix
        self._current_state = iop.get_numpy_rng().choice(self._states)  # initialize uniformly at random

    def should_drop(self) -> bool:
        self._current_state = iop.get_numpy_rng().choice(
            self._states, p=self._P[self._current_state]
        )  # evolve the Markov chain

        return iop.get_numpy_rng().random() < self.drop_rate if self._current_state else False
