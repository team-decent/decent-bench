from collections.abc import Iterable
from typing import TYPE_CHECKING

from decent_bench.algorithms._algorithm import Algorithm
from decent_bench.networks import P2PNetwork

if TYPE_CHECKING:
    from decent_bench.agents import Agent


class P2PAlgorithm(Algorithm[P2PNetwork]):
    """Distributed algorithm - agents collaborate using peer-to-peer communication."""

    def cleanup_agents(self, network: P2PNetwork) -> Iterable["Agent"]:
        return network.agents()
