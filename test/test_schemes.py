import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.costs import L2RegularizerCost
from decent_bench.schemes import (
    AgentActivationScheme,
    AlwaysActive,
    ClientSelectionScheme,
    CompressionScheme,
    DropScheme,
    GaussianNoise,
    GilbertElliott,
    MarkovChainActivation,
    NoCompression,
    NoDrops,
    NoiseScheme,
    NoNoise,
    PoissonActivation,
    Quantization,
    RandK,
    TopK,
    UniformActivationRate,
    UniformClientSelection,
    UniformDropRate,
)
from decent_bench.utils.array import Array
from decent_bench.utils import interoperability as iop


## AgentActivationScheme

# test when agents always activate
@pytest.mark.parametrize(
    "scheme",
    [
        AlwaysActive(),
        UniformActivationRate(activation_probability=1),
        MarkovChainActivation(inactive_to_active=1.0, active_to_inactive=0.0),
        PoissonActivation(mean_interval=0),
    ],
)
def test_always_active(
    scheme: AgentActivationScheme,
    n_iterations: int = 10
) -> None:
    activations = sum(scheme.is_active(iteration) for iteration in range(n_iterations))

    assert activations == n_iterations


# test when agents never activate
@pytest.mark.parametrize(
    "scheme",
    [
        UniformActivationRate(activation_probability=0),
        MarkovChainActivation(inactive_to_active=0, active_to_inactive=1),
        PoissonActivation(mean_interval=1e10),
    ],
)
def test_never_active(
    scheme: AgentActivationScheme,
    n_iterations: int = 10
) -> None:
    activations = sum(scheme.is_active(iteration) for iteration in range(n_iterations))

    assert activations == 0


# test when agents activate randomly
@pytest.mark.parametrize(
    "scheme",
    [
        UniformActivationRate(activation_probability=0.5),
        MarkovChainActivation(inactive_to_active=0.5, active_to_inactive=0.5),
        PoissonActivation(mean_interval=1),
    ],
)
def test_randomly_active(
    scheme: AgentActivationScheme,
    n_iterations: int = 10
) -> None:
    activations = sum(scheme.is_active(iteration) for iteration in range(n_iterations))

    assert 0 <= activations <= n_iterations


## ClientSelectionScheme

# test client selection
def make_clients(n_clients: int) -> list[Agent]:
    return [Agent(agent_id, L2RegularizerCost((2,))) for agent_id in range(n_clients)]


@pytest.mark.parametrize(
    ("scheme", "n_clients", "expected_selected"),
    [
        (UniformClientSelection(clients_per_round=3), 5, 3),
        (UniformClientSelection(client_fraction=0.4), 5, 2),
        (UniformClientSelection(clients_per_round=5), 5, 5),
    ],
)
def test_client_selection(
    scheme: ClientSelectionScheme,
    n_clients: int,
    expected_selected: int,
) -> None:
    selected_clients = scheme.select(make_clients(n_clients), iteration=0)

    assert len(selected_clients) == expected_selected


## CompressionScheme

# test no compression
@pytest.mark.parametrize(
    ("scheme", "message"),
    [
        (NoCompression(), Array(np.array([3.0, -4.0, 1.0]))),
        (Quantization(n_significant_digits=5), Array(np.array([1.2345, -2.3456]))),
        (TopK(k=3), Array(np.array([3.0, -4.0, 1.0]))),
        (RandK(k=3), Array(np.array([3.0, -4.0, 1.0]))),
    ],
)
def test_no_compression(
    scheme: CompressionScheme,
    message: Array
) -> None:
    original_message = Array(np.copy(iop.to_numpy(message)))
    compression_error = float(iop.norm(original_message - scheme.compress(message)))

    assert compression_error == pytest.approx(0)


# test with compression
@pytest.mark.parametrize(
    ("scheme", "message", "expected_norm"),
    [
        (
            Quantization(n_significant_digits=3),
            Array(np.array([1.2345, -2.3456])),
            float(np.linalg.norm(np.array([0.0045, -0.0044]))),
        ),
        (TopK(k=2), Array(np.array([3.0, -4.0, 1.0])), 1.0),
    ],
)
def test_compression(
    scheme: CompressionScheme,
    message: Array,
    expected_norm: float,
) -> None:
    original_message = Array(np.copy(iop.to_numpy(message)))
    compression_error = float(iop.norm(original_message - scheme.compress(message)))

    assert compression_error == pytest.approx(expected_norm)


# test RandK
def test_randk_compression() -> None:
    scheme = RandK(k=2)
    observed_norms = set()

    for _ in range(200):
        message = Array(np.array([3.0, -4.0, 1.0]))
        original_message = Array(np.copy(iop.to_numpy(message)))
        compression_error = float(iop.norm(original_message - scheme.compress(message)))
        observed_norms.add(compression_error)

    assert observed_norms == {1.0, 3.0, 4.0}


# test RandK and TopK
@pytest.mark.parametrize(
    "scheme",
    [
        TopK,
        RandK
    ],
)
def test_k_compression(scheme: CompressionScheme) -> None:
    for k in range(1, 15+1):
        s = scheme(k=k)
        compressed_msg = s.compress(Array(np.ones(k+5)))

        assert np.count_nonzero(compressed_msg) == k


# test RandK and TopK with mismatched k and message size
@pytest.mark.parametrize(
    "scheme",
    [
        TopK,
        RandK
    ],
)
def test_k_compression_mismatched(scheme: CompressionScheme) -> None:
    for k in range(5, 15+1):
        s = scheme(k=k)
        compressed_msg = s.compress(Array(np.ones(k-2)))

        assert np.count_nonzero(compressed_msg) == k-2


# test RandK and TopK with mismatched k and message size
@pytest.mark.parametrize(
    "scheme",
    [
        TopK,
        RandK
    ],
)
def test_k_compression_mismatched_k_msg_size(scheme: CompressionScheme) -> None:
    for k in range(5, 15+1):
        s = scheme(k=k)
        compressed_msg = s.compress(Array(np.ones(k-2)))

        assert np.count_nonzero(compressed_msg) == k-2


# test RandK and TopK to check message shape is preserved
@pytest.mark.parametrize(
    "scheme",
    [
        TopK,
        RandK
    ],
)
def test_k_compression_preserved_shape(scheme: CompressionScheme) -> None:
    message = Array(np.ones((10, 15)))
    s = scheme(k=5)
    compressed_msg = s.compress(message)

    assert iop.to_numpy(message).shape == iop.to_numpy(compressed_msg).shape


## DropScheme

# test no drops
@pytest.mark.parametrize(
    "scheme",
    [
        NoDrops(),
        UniformDropRate(drop_rate=0),
        GilbertElliott(drop_rate=0)
    ],
)
def test_no_drops(
    scheme: DropScheme,
    n_messages: int = 10,
) -> None:
    delivered_messages = sum(not scheme.should_drop() for _ in range(n_messages))

    assert delivered_messages == n_messages


# test drops
@pytest.mark.parametrize(
    "scheme",
    [
        UniformDropRate(drop_rate=0.5),
        GilbertElliott(drop_rate=0.5)
    ],
)
def test_drops(
    scheme: DropScheme,
    n_messages: int = 10,
) -> None:
    delivered_messages = sum(not scheme.should_drop() for _ in range(n_messages))

    assert 0 <= delivered_messages <= n_messages


## NoiseScheme

# test no noise
@pytest.mark.parametrize(
    "scheme",
    [
        NoNoise(),
        GaussianNoise(mean=0, std=0)
    ],
)
def test_no_noise(
    scheme: NoiseScheme,
) -> None:
    message = Array(np.ones(5))
    noise_error = float(iop.norm(message - scheme.make_noise(message)))

    assert noise_error == pytest.approx(0)


# test noise
@pytest.mark.parametrize(
    "scheme",
    [
        GaussianNoise(mean=0.5, std=0.5)
    ],
)
def test_noise(
    scheme: NoiseScheme,
) -> None:
    message = Array(np.ones(5))
    noise_error = float(iop.norm(message - scheme.make_noise(message)))

    assert noise_error > 0
