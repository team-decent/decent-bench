from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np
from decent_array import interoperability as iop


class AgentActivationScheme(ABC):
    """
    Scheme defining how agents go active/inactive over the course of the algorithm execution.

    Activation schemes are attached to agents by networks and are queried during algorithm execution.
    """

    @abstractmethod
    def is_active(self, iteration: int) -> bool:
        """
        Whether or not the agent is active.

        Args:
            iteration: current iteration of algorithm execution

        """


class AlwaysActive(AgentActivationScheme):
    """Scheme that makes the agent always active."""

    def is_active(self, iteration: int) -> bool:  # noqa: ARG002
        return True


class UniformActivationRate(AgentActivationScheme):
    """
    Scheme where the agent is active with fixed probability.

    Each call samples an independent Bernoulli event with probability ``activation_probability``.

    Args:
        activation_probability: probability that the agent is active at a queried iteration.

    Raises:
        ValueError: if ``activation_probability`` is not in :math:`[0, 1]`.

    """

    def __init__(self, activation_probability: float):
        if activation_probability < 0 or activation_probability > 1:
            raise ValueError("activation_probability must be in [0, 1]")
        self.activation_probability = activation_probability

    def is_active(self, iteration: int) -> bool:  # noqa: ARG002
        return random.random() < self.activation_probability


class MarkovChainActivation(AgentActivationScheme):
    """
    Scheme modeling activation with a 2-state Markov chain.

    The scheme models activation with a 2-state (active and inactive) Markov chain. The agent transitions
    between the two states with the given probabilities.

    Args:
        inactive_to_active: transition probability from inactive to active
        active_to_inactive: transition probability from active to inactive

    Raises:
        ValueError: if `inactive_to_active` or `active_to_inactive` are not in :math:`[0, 1]`

    """

    def __init__(self, inactive_to_active: float = 0.5, active_to_inactive: float = 0.5):
        if (inactive_to_active < 0 or inactive_to_active > 1) or (active_to_inactive < 0 or active_to_inactive > 1):
            raise ValueError("Transition probabilities must be in [0, 1]")
        self.inactive_to_active = inactive_to_active
        self.active_to_inactive = active_to_inactive
        self._states = np.array([0, 1])  # inactive = 0, active = 1
        self._P = np.array(
            [
                [1 - inactive_to_active, inactive_to_active],
                [active_to_inactive, 1 - active_to_inactive],
            ]
        )  # transition matrix
        self._current_state = iop.get_numpy_rng().choice(self._states, p=[0, 1])

    def is_active(self, iteration: int) -> bool:  # noqa: ARG002
        self._current_state = iop.get_numpy_rng().choice(
            self._states,
            p=self._P[self._current_state],
        )  # evolve the Markov chain

        return bool(self._current_state)


class PoissonActivation(AgentActivationScheme):
    """
    Scheme modeling activation at random intervals determined by a Poisson distribution.

    The agent activates at random intervals of length sampled from a Poisson distribution of given mean.

    Args:
        mean_interval: mean interval of inactivity

    Raises:
        ValueError: if `mean_interval` is negative

    """

    def __init__(self, mean_interval: float = 1.0):
        if mean_interval < 0:
            raise ValueError("`mean_interval` must be non-negative")
        self.mean_interval = mean_interval
        self._countdown = int(iop.get_numpy_rng().poisson(self.mean_interval))

    def is_active(self, iteration: int) -> bool:  # noqa: ARG002
        if self._countdown == 0:
            self._countdown = int(iop.get_numpy_rng().poisson(self.mean_interval))
            return True
        self._countdown -= 1
        return False


class CyclicActivation(AgentActivationScheme):
    """
    Scheme where an agent cycles through active and inactive intervals.

    The agent is active for ``active_for`` iterations and inactive for ``inactive_for`` iterations in each cycle.
    If ``inactive_for`` is not provided, it defaults to ``active_for``. ``offset`` shifts the phase of the cycle,
    allowing agents to follow the same cycle with staggered active windows.

    Args:
        active_for: number of active iterations in each cycle.
        inactive_for: number of inactive iterations in each cycle. If ``None``, it defaults to ``active_for``.
        offset: phase offset applied to the cycle.

    Raises:
        ValueError: if ``active_for``, ``inactive_for``, or ``offset`` is negative, both intervals are zero, or
            ``iteration`` is negative.

    """

    def __init__(self, active_for: int, inactive_for: int | None = None, offset: int = 0):
        inactive_for = active_for if inactive_for is None else inactive_for
        if active_for < 0 or inactive_for < 0:
            raise ValueError("active_for and inactive_for must be non-negative")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if active_for == 0 and inactive_for == 0:
            raise ValueError("At least one of active_for or inactive_for must be positive")
        self.active_for = active_for
        self.inactive_for = inactive_for
        self.offset = offset

    def is_active(self, iteration: int) -> bool:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        period = self.active_for + self.inactive_for
        phase = (iteration + self.offset) % period
        return phase < self.active_for
