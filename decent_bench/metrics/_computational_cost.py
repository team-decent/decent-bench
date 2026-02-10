from dataclasses import dataclass


@dataclass
class ComputationalCost:
    """Computational costs associated with an algorithm for plot metrics."""

    function: float = 1.0
    gradient: float = 1.0
    hessian: float = 1.0
    proximal: float = 1.0
    communication: float = 1.0
