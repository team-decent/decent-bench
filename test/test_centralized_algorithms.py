import numpy as np
import pytest

import decent_bench.centralized_algorithms as ca
from decent_bench.costs import Cost, QuadraticCost
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class DummyCost(Cost):
    def __init__(self, m_smooth: float, m_cvx: float):
        self._m_smooth = m_smooth
        self._m_cvx = m_cvx

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def m_smooth(self) -> float:
        return self._m_smooth

    @property
    def m_cvx(self) -> float:
        return self._m_cvx

    def function(self, x: np.ndarray) -> float:
        return float(0.5 * x[0] * x[0])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([x[0]], dtype=float)

    def hessian(self, x: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.asarray([[1.0]], dtype=float)

    def proximal(self, x: np.ndarray, rho: float) -> np.ndarray:
        return x / (1.0 + rho)


class InPlaceStepSolver(ca.Solver):
    def step(self, iteration: int) -> None:  # noqa: ARG002
        self.x += 1.0


class DecayStepSolver(ca.Solver):
    def __init__(self, cost: Cost, x0: np.ndarray | None = None):
        super().__init__(cost=cost, x0=x0)
        self.steps = 0

    def step(self, iteration: int) -> None:
        self.x += 1.0 / (iteration + 1) ** 2
        self.steps += 1


class ConstantStepSolver(ca.Solver):
    def step(self, iteration: int) -> None:  # noqa: ARG002
        self.x += 1.0


class StubLogger:
    def __init__(self, handlers: list[object] | None = None):
        self.handlers = [] if handlers is None else handlers
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_solve_quadratic_uses_closed_form(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_run(*args: object, **kwargs: object) -> np.ndarray:  # noqa: ARG001
        raise AssertionError("Iterative solver should not be used for quadratic costs")

    monkeypatch.setattr(ca.GradientDescent, "run", fail_run)
    monkeypatch.setattr(ca.AcceleratedGradientDescent, "run", fail_run)

    cost = QuadraticCost(A=np.asarray([[2.0]], dtype=float), b=np.asarray([-4.0], dtype=float))
    x = ca.solve(cost)

    np.testing.assert_allclose(np.asarray(x), np.asarray([2.0], dtype=float))


def test_solve_rejects_affine_costs() -> None:
    cost = DummyCost(m_smooth=0.0, m_cvx=0.0)

    with pytest.raises(ValueError, match="m_smooth = 0"):
        ca.solve(cost)


def test_solve_routes_smooth_convex_to_accelerated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"gd": 0, "agd": 0}
    sentinel = np.asarray([42.0], dtype=float)

    def fake_gd_run(self: ca.GradientDescent, **kwargs: object) -> np.ndarray:  # noqa: ARG001
        calls["gd"] += 1
        return np.asarray([-1.0], dtype=float)

    def fake_agd_run(self: ca.AcceleratedGradientDescent, **kwargs: object) -> np.ndarray:  # noqa: ARG001
        calls["agd"] += 1
        return sentinel

    monkeypatch.setattr(ca.GradientDescent, "run", fake_gd_run)
    monkeypatch.setattr(ca.AcceleratedGradientDescent, "run", fake_agd_run)

    cost = DummyCost(m_smooth=1.0, m_cvx=0.0)
    out = ca.solve(cost, max_iter=7, stop_tol=1e-4, max_tol=1e-3)

    np.testing.assert_allclose(np.asarray(out), sentinel)
    assert calls["agd"] == 1
    assert calls["gd"] == 0


@pytest.mark.parametrize(
    ("m_smooth", "m_cvx"),
    [
        (np.inf, 0.0),
        (np.nan, 0.0),
        (1.0, np.nan),
    ],
)
def test_solve_routes_nonsmooth_or_nonconvex_to_gradient(
    monkeypatch: pytest.MonkeyPatch,
    m_smooth: float,
    m_cvx: float,
) -> None:
    calls = {"gd": 0, "agd": 0}
    sentinel = np.asarray([7.0], dtype=float)

    def fake_gd_run(self: ca.GradientDescent, **kwargs: object) -> np.ndarray:  # noqa: ARG001
        calls["gd"] += 1
        return sentinel

    def fake_agd_run(self: ca.AcceleratedGradientDescent, **kwargs: object) -> np.ndarray:  # noqa: ARG001
        calls["agd"] += 1
        return np.asarray([-1.0], dtype=float)

    monkeypatch.setattr(ca.GradientDescent, "run", fake_gd_run)
    monkeypatch.setattr(ca.AcceleratedGradientDescent, "run", fake_agd_run)

    cost = DummyCost(m_smooth=m_smooth, m_cvx=m_cvx)
    out = ca.solve(cost, max_iter=7, stop_tol=1e-4, max_tol=1e-3)

    np.testing.assert_allclose(np.asarray(out), sentinel)
    assert calls["gd"] == 1
    assert calls["agd"] == 0


def test_solver_run_uses_snapshot_for_inplace_updates() -> None:
    solver = InPlaceStepSolver(cost=DummyCost(m_smooth=1.0, m_cvx=0.0), x0=np.asarray([0.0], dtype=float))

    out = solver.run(max_iter=5, stop_tol=1e-12, show_progress=False)

    np.testing.assert_allclose(np.asarray(out), np.asarray([5.0], dtype=float))


def test_solver_run_honors_stop_tol() -> None:
    solver = DecayStepSolver(cost=DummyCost(m_smooth=1.0, m_cvx=0.0), x0=np.asarray([0.0], dtype=float))

    out = solver.run(max_iter=50, stop_tol=0.2, show_progress=False)

    expected = 1.0 + 0.25
    np.testing.assert_allclose(np.asarray(out), np.asarray([expected], dtype=float))
    assert solver.steps == 2


def test_solver_run_raises_when_max_tol_not_met() -> None:
    solver = ConstantStepSolver(cost=DummyCost(m_smooth=1.0, m_cvx=0.0), x0=np.asarray([0.0], dtype=float))

    with pytest.raises(RuntimeError, match="failed to converge"):
        solver.run(max_iter=3, max_tol=0.5, show_progress=False)


@pytest.mark.parametrize(
    ("max_iter", "stop_tol", "max_tol", "msg"),
    [
        (0, None, None, "max_iter"),
        (1, 0.0, None, "stop_tol"),
        (1, None, 0.0, "max_tol"),
    ],
)
def test_solver_run_validates_inputs(
    max_iter: int,
    stop_tol: float | None,
    max_tol: float | None,
    msg: str,
) -> None:
    solver = ConstantStepSolver(cost=DummyCost(m_smooth=1.0, m_cvx=0.0), x0=np.asarray([0.0], dtype=float))

    with pytest.raises(ValueError, match=msg):
        solver.run(max_iter=max_iter, stop_tol=stop_tol, max_tol=max_tol, show_progress=False)


def test_solve_logger_initialization_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    cost = QuadraticCost(A=np.asarray([[1.0]], dtype=float), b=np.asarray([-1.0], dtype=float))

    started_no_handlers = {"count": 0}
    stub_no_handlers = StubLogger(handlers=[])

    def fake_start_logger_no_handlers() -> None:
        started_no_handlers["count"] += 1
        stub_no_handlers.handlers.append(object())

    monkeypatch.setattr(ca, "LOGGER", stub_no_handlers)
    monkeypatch.setattr(ca.logger, "start_logger", fake_start_logger_no_handlers)

    ca.solve(cost)

    assert started_no_handlers["count"] == 1

    started_with_handlers = {"count": 0}
    stub_with_handlers = StubLogger(handlers=[object()])

    def fake_start_logger_with_handlers() -> None:
        started_with_handlers["count"] += 1

    monkeypatch.setattr(ca, "LOGGER", stub_with_handlers)
    monkeypatch.setattr(ca.logger, "start_logger", fake_start_logger_with_handlers)

    ca.solve(cost)

    assert started_with_handlers["count"] == 0
