import numpy as np
import pytest

from decent_bench.agents import Agent
from decent_bench.costs import L2RegularizerCost, LinearRegressionCost
from decent_bench.schemes import (
    AgentActivationScheme,
    AlwaysActive,
    ClientSelectionScheme,
    CompressionScheme,
    CyclicActivation,
    DataSizeSelection,
    DropScheme,
    GaussianNoise,
    GilbertElliott,
    HighLossSelection,
    MarkovChainActivation,
    NoCompression,
    NoDrops,
    NoiseScheme,
    NoNoise,
    FairSelection,
    PoissonActivation,
    StochasticQuantization,
    Quantization,
    RandK,
    TopK,
    UniformActivationRate,
    UniformDropRate,
    UniformSelection,
)
from decent_bench.utils import interoperability as iop
from decent_bench.utils.array import Array

## AgentActivationScheme


# test when agents always activate
@pytest.mark.parametrize(
    "scheme",
    [
        AlwaysActive(),
        UniformActivationRate(activation_probability=1),
        MarkovChainActivation(inactive_to_active=1.0, active_to_inactive=0.0),
        PoissonActivation(mean_interval=0),
        CyclicActivation(active_for=1, inactive_for=0),
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
        CyclicActivation(active_for=0, inactive_for=1),
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


def test_uniform_activation_rate_requires_probability() -> None:
    with pytest.raises(ValueError, match="activation_probability"):
        UniformActivationRate(activation_probability=1.1)


def test_cyclic_activation_follows_active_then_inactive_cycle() -> None:
    scheme = CyclicActivation(active_for=2, inactive_for=3)

    activations = [scheme.is_active(iteration) for iteration in range(8)]

    assert activations == [True, True, False, False, False, True, True, False]


def test_cyclic_activation_defaults_inactive_period_to_active_period() -> None:
    scheme = CyclicActivation(active_for=2)

    activations = [scheme.is_active(iteration) for iteration in range(6)]

    assert activations == [True, True, False, False, True, True]


def test_cyclic_activation_applies_phase_offset() -> None:
    scheme = CyclicActivation(active_for=2, inactive_for=3, offset=4)

    activations = [scheme.is_active(iteration) for iteration in range(7)]

    assert activations == [False, True, True, False, False, False, True]


def test_cyclic_activation_requires_non_empty_cycle() -> None:
    with pytest.raises(ValueError, match="active_for"):
        CyclicActivation(active_for=0, inactive_for=0)


def test_cyclic_activation_requires_non_negative_offset() -> None:
    with pytest.raises(ValueError, match="offset"):
        CyclicActivation(active_for=1, offset=-1)


## ClientSelectionScheme


# test client selection
def make_clients(n_clients: int) -> list[Agent]:
    return [_make_linear_regression_client(agent_id, agent_id + 1) for agent_id in range(n_clients)]


def _make_linear_regression_client(agent_id: int, n_samples: int, scale: float = 1.0) -> Agent:
    dataset = [(np.array([1.0, 1.0]), np.array([0.0])) for _ in range(n_samples)]
    return Agent(agent_id, scale * LinearRegressionCost(dataset))


@pytest.mark.parametrize(
    ("scheme", "n_clients", "expected_selected"),
    [
        (UniformSelection(num_selected_clients=3), 5, 3),
        (UniformSelection(fraction_selected_clients=0.4), 5, 2),
        (UniformSelection(num_selected_clients=5), 5, 5),
        (DataSizeSelection(num_selected_clients=3), 5, 3),
        (DataSizeSelection(fraction_selected_clients=0.4), 5, 2),
        (FairSelection(num_selected_clients=3), 5, 3),
        (FairSelection(fraction_selected_clients=0.4), 5, 2),
    ],
)
def test_client_selection(
    scheme: ClientSelectionScheme,
    n_clients: int,
    expected_selected: int,
) -> None:
    selected_clients = scheme.select(make_clients(n_clients), iteration=0)

    assert len(selected_clients) == expected_selected


def test_data_size_client_selection_prefers_larger_clients() -> None:
    iop.set_seed(0)
    clients = [
        _make_linear_regression_client(0, 1),
        _make_linear_regression_client(1, 1000),
    ]
    scheme = DataSizeSelection(num_selected_clients=1)

    selected_client_ids = [scheme.select(clients, iteration=i)[0].id for i in range(200)]

    assert selected_client_ids.count(1) > 190


def test_data_size_client_selection_requires_empirical_risk_cost() -> None:
    clients = [Agent(0, L2RegularizerCost((2,))), Agent(1, L2RegularizerCost((2,)))]
    scheme = DataSizeSelection(num_selected_clients=1)

    with pytest.raises(ValueError, match="EmpiricalRiskCost"):
        scheme.select(clients, iteration=0)


def test_fair_selection_prioritizes_under_selected_clients() -> None:
    clients = make_clients(2)
    scheme = FairSelection(num_selected_clients=1)

    assert scheme.select([clients[0]], iteration=0) == [clients[0]]
    assert scheme.select(clients, iteration=1) == [clients[1]]


def test_fair_selection_updates_counts_when_all_selected() -> None:
    clients = make_clients(3)
    scheme = FairSelection(num_selected_clients=2)

    assert scheme.select(clients[:2], iteration=0) == clients[:2]
    assert clients[2] in scheme.select(clients, iteration=1)


def test_high_loss_client_selection_selects_largest_loss() -> None:
    clients = [
        Agent(0, 1.0 * L2RegularizerCost((2,)), data={"n_samples": 1}),
        Agent(1, 10.0 * L2RegularizerCost((2,)), data={"n_samples": 1}),
        Agent(2, 2.0 * L2RegularizerCost((2,)), data={"n_samples": 1}),
    ]
    for client in clients:
        client.x = Array(np.ones(2))
    scheme = HighLossSelection(num_selected_clients=1)

    selected_clients = scheme.select(clients, iteration=0)

    assert selected_clients == [clients[1]]


def test_high_loss_client_selection_selects_fraction() -> None:
    clients = [
        Agent(0, 1.0 * L2RegularizerCost((2,)), data={"n_samples": 1}),
        Agent(1, 10.0 * L2RegularizerCost((2,)), data={"n_samples": 1}),
        Agent(2, 2.0 * L2RegularizerCost((2,)), data={"n_samples": 1}),
        Agent(3, 5.0 * L2RegularizerCost((2,)), data={"n_samples": 1}),
        Agent(4, 3.0 * L2RegularizerCost((2,)), data={"n_samples": 1}),
    ]
    for client in clients:
        client.x = Array(np.ones(2))
    scheme = HighLossSelection(fraction_selected_clients=0.4)

    selected_clients = scheme.select(clients, iteration=0)

    assert selected_clients == [clients[1], clients[3]]


def test_high_loss_client_selection_does_not_consume_empirical_risk_batch() -> None:
    dataset = [
        (np.array([1.0, 0.0]), np.array([1.0])),
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([2.0, 0.0]), np.array([2.0])),
        (np.array([0.0, 2.0]), np.array([2.0])),
    ]
    x = Array(np.zeros(2))

    iop.set_seed(0)
    control_cost = LinearRegressionCost(dataset, batch_size=2)
    control_cost.gradient(x)
    expected_batch = control_cost.batch_used

    iop.set_seed(0)
    selected_cost = LinearRegressionCost(dataset, batch_size=2)
    clients = [
        Agent(0, selected_cost),
        Agent(1, LinearRegressionCost(dataset, batch_size=2)),
    ]
    for client in clients:
        client.x = x

    HighLossSelection(num_selected_clients=1).select(clients, iteration=0)
    selected_cost.gradient(x)

    assert selected_cost.batch_used == expected_batch


## CompressionScheme


# test no compression
@pytest.mark.parametrize(
    ("scheme", "message"),
    [
        (NoCompression(), Array(np.array([3.0, -4.0, 1.0]))),
        (Quantization(n_significant_digits=5), Array(np.array([1.2345, -2.3456]))),
        (StochasticQuantization(n_levels=4), Array(np.array([0.0, 0.0, 0.0]))),
        (TopK(k=1.0), Array(np.array([3.0, -4.0, 1.0]))),
        (RandK(k=1.0), Array(np.array([3.0, -4.0, 1.0]))),
        (TopK(k=3), Array(np.array([3.0, -4.0, 1.0]))),
        (RandK(k=3), Array(np.array([3.0, -4.0, 1.0]))),
    ],
)
def test_no_compression(scheme: CompressionScheme, message: Array) -> None:
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
        (TopK(k=2 / 3), Array(np.array([3.0, -4.0, 1.0])), 1.0),
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
@pytest.mark.parametrize(
    "k",
    [
        2 / 3,
        2,
    ],
)
def test_randk_compression(k) -> None:
    scheme = RandK(k=k)
    observed_norms = set()

    for _ in range(200):
        message = Array(np.array([3.0, -4.0, 1.0]))
        original_message = Array(np.copy(iop.to_numpy(message)))
        compression_error = float(iop.norm(original_message - scheme.compress(message)))
        observed_norms.add(compression_error)

    assert observed_norms == {1.0, 3.0, 4.0}


def test_stochastic_quantization_preserves_shape_and_signs() -> None:
    message = Array(np.array([[3.0, -4.0], [0.0, 1.0]]))
    compressed_message = StochasticQuantization(n_levels=4).compress(message)

    assert iop.to_numpy(compressed_message).shape == iop.to_numpy(message).shape
    assert np.all(np.sign(iop.to_numpy(compressed_message)) == np.sign(iop.to_numpy(message)))


def test_stochastic_quantization_uses_norm_scaled_levels() -> None:
    iop.set_seed(0)
    message = Array(np.array([3.0, 4.0]))
    compressed_message = StochasticQuantization(n_levels=4).compress(message)
    quantized_levels = 4 * np.abs(iop.to_numpy(compressed_message)) / float(iop.norm(message))

    assert set(quantized_levels).issubset({0.0, 1.0, 2.0, 3.0, 4.0})


def test_stochastic_quantization_is_stochastic() -> None:
    message = Array(np.array([3.0, 4.0]))
    scheme = StochasticQuantization(n_levels=4)
    observed_messages = {tuple(iop.to_numpy(scheme.compress(message))) for _ in range(100)}

    assert len(observed_messages) > 1


def test_stochastic_quantization_requires_positive_levels() -> None:
    with pytest.raises(ValueError, match="n_levels"):
        StochasticQuantization(n_levels=0)


# test RandK and TopK
@pytest.mark.parametrize(
    "scheme",
    [
        TopK,
        RandK
    ],
)
def test_k_compression(scheme: CompressionScheme) -> None:
    for n_kept in range(1, 15 + 1):
        s = scheme(k=n_kept / (n_kept + 5))
        compressed_msg = s.compress(Array(np.ones(n_kept + 5)))
        assert np.count_nonzero(compressed_msg) == n_kept
        s = scheme(k=n_kept)
        compressed_msg = s.compress(Array(np.ones(n_kept + 5)))
        assert np.count_nonzero(compressed_msg) == n_kept


# test RandK and TopK with mismatched k and message size
@pytest.mark.parametrize(
    "scheme",
    [
        TopK,
        RandK
    ],
)
def test_k_compression_mismatched(scheme: CompressionScheme) -> None:
    for k in range(5, 15 + 1):
        s = scheme(k=k)
        compressed_msg = s.compress(Array(np.ones(k - 2)))
        assert np.count_nonzero(compressed_msg) == k - 2


# test RandK and TopK with mismatched k and message size
@pytest.mark.parametrize(
    "scheme",
    [
        TopK,
        RandK
    ],
)
def test_k_compression_mismatched_k_msg_size(scheme: CompressionScheme) -> None:
    s = scheme(k=1.0)
    compressed_msg = s.compress(Array(np.ones(8)))
    assert np.count_nonzero(compressed_msg) == 8

    for k in range(5, 15 + 1):
        s = scheme(k=k)
        compressed_msg = s.compress(Array(np.ones(k - 2)))

        assert np.count_nonzero(compressed_msg) == k - 2


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
    s = scheme(k=0.1)
    compressed_msg = s.compress(message)
    assert iop.to_numpy(message).shape == iop.to_numpy(compressed_msg).shape

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


def test_topk_compression_on_multi_dimensional_message() -> None:
    message = Array(np.array([[1.0, 0.5], [0.2, 0.1]]))
    scheme = TopK(k=0.5)
    compressed_message = scheme.compress(message)

    np.testing.assert_array_equal(compressed_message, Array(np.array([[1.0, 0.5], [0.0, 0.0]])))
