from decent_bench.algorithms._algorithm import Algorithm
from decent_bench.networks import P2PNetwork


class P2PAlgorithm(Algorithm[P2PNetwork]):
    """Distributed algorithm - agents collaborate using peer-to-peer communication."""
