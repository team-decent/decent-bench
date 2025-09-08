import itertools
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator


class AgentActivationScheme(ABC):
    """Scheme defining how agents go active/inactive over the course of the algorithm execution."""

    @abstractmethod
    def is_active(self, iteration: int) -> bool:
        """
        Whether or not the agent is active.

        Args:
            iteration: current iteration of algorithm execution

        """


class AlwaysActive(AgentActivationScheme):
    """Scheme that makes the agent always active."""

    def is_active(self, iteration: int) -> bool:  # noqa: D102, ARG002
        return True

    def __iter__(self) -> Iterator[AgentActivationScheme]:
        """
        Return an iterator that repeats *self* when calling ``iter()`` on this object.

        This enables the class to act as an infinite :class:`~collections.abc.Iterable`, always yielding *self*.
        """
        return itertools.repeat(self)


class UniformActivationRate(AgentActivationScheme):
    """Scheme where the agent's probability of being active is uniformly distributed."""

    def __init__(self, activation_probability: float):
        self.activation_probability = activation_probability

    def is_active(self, iteration: int) -> bool:  # noqa: D102, ARG002
        return random.random() < self.activation_probability

    def __iter__(self) -> Iterator[AgentActivationScheme]:
        """
        Return an iterator that repeats *self* when calling ``iter()`` on this object.

        This enables the class to act as an infinite :class:`~collections.abc.Iterable`, always yielding *self*.
        """
        return itertools.repeat(self)
