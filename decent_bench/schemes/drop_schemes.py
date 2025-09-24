import random
from abc import ABC, abstractmethod


class DropScheme(ABC):
    """Scheme defining how message drops occur over the network."""

    @abstractmethod
    def should_drop(self) -> bool:
        """Whether or not to drop."""


class NoDrops(DropScheme):
    """Scheme that never drops messages."""

    def should_drop(self) -> bool:  # noqa: D102
        return False


class UniformDropRate(DropScheme):
    """Scheme that drops messages with uniform probability."""

    def __init__(self, drop_rate: float):
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("Drop rate must be in [0, 1]")
        self.drop_rate = drop_rate

    def should_drop(self) -> bool:  # noqa: D102
        return random.random() < self.drop_rate
