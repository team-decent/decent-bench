"""Backend-aware fixtures for the test suite."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from decent_array import interoperability as iop
from decent_array.interoperability._backend_manager import reset_backends
from decent_array.types import Devices, Frameworks

from decent_bench.benchmark import configure

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest

LOGGER = logging.getLogger(__name__)


def _backend_available(framework: Frameworks, device: Devices) -> tuple[bool, str | None]:
    """Check whether a backend/device pair can run in this environment."""
    if framework is Frameworks.NUMPY:
        return (device is Devices.CPU, None if device is Devices.CPU else "NumPy only supports CPU")

    try:
        if framework is Frameworks.PYTORCH:
            import torch

            if device is Devices.CPU:
                return True, None
            if device is Devices.GPU:
                return bool(torch.cuda.is_available()), "PyTorch CUDA is unavailable"
            if device is Devices.MPS:
                return bool(torch.backends.mps.is_available()), "PyTorch MPS is unavailable"
        elif framework is Frameworks.TENSORFLOW:
            import tensorflow as tf

            if device is Devices.CPU:
                return True, None
            if device is Devices.GPU:
                return bool(tf.config.list_physical_devices("GPU")), "TensorFlow GPU is unavailable"
        elif framework is Frameworks.JAX:
            import jax

            if device is Devices.MPS:
                return False, "JAX MPS is unsupported"
            jax.devices(device.value)
            return True, None
    except (ImportError, RuntimeError, ValueError) as error:
        return False, f"{framework.value}/{device.value} unavailable: {error}"

    return False, f"{framework.value}/{device.value} is unsupported"


def _backend_matrix() -> list[pytest.ParameterSet]:
    """Build backend parameters, marking unavailable combinations as skips."""
    params: list[pytest.ParameterSet] = []
    for framework in Frameworks:
        for device in Devices:
            available, reason = _backend_available(framework, device)
            pair = (framework, device)
            if available:
                params.append(pytest.param(pair, id=f"{framework.value}-{device.value}"))
            else:
                params.append(
                    pytest.param(
                        pair,
                        id=f"{framework.value}-{device.value}",
                        marks=pytest.mark.skip(reason=reason),
                    )
                )
                LOGGER.info("Skipping unavailable test backend %s/%s: %s", framework.value, device.value, reason)
    return params


BACKEND_PARAMS = _backend_matrix()


def pytest_report_header() -> str:
    """Report backend/device combinations omitted from matrix tests."""
    skipped = [
        f"{framework.value}/{device.value}"
        for framework in Frameworks
        for device in Devices
        if not _backend_available(framework, device)[0]
    ]
    return "Unavailable backend matrix entries skipped: " + ", ".join(skipped) if skipped else "All backends available"


@pytest.fixture(params=BACKEND_PARAMS)
def backend(request: FixtureRequest) -> Iterator[tuple[Frameworks, Devices]]:
    """Configure one available backend/device pair for a backend-agnostic test."""
    framework, device = request.param
    configure(framework, device)
    yield framework, device


@pytest.fixture(autouse=True)
def isolate_and_configure_backend(request: FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test a clean backend and configure its declared/default backend."""
    configure_module = importlib.import_module("decent_bench.benchmark._configure")
    reset_backends()
    monkeypatch.setattr(configure_module, "_STATE", configure_module._ConfigurationState())

    markers = request.node.keywords
    if "backend" in request.fixturenames:
        # The backend fixture configures the selected pair when requested.
        request.getfixturevalue("backend")
    else:
        framework = request.node.get_closest_marker("backend_framework")
        framework_value = framework.args[0] if framework else None
        callspec = getattr(request.node, "callspec", None)
        params: dict[str, Any] = callspec.params if callspec is not None else {}
        selected_framework = params.get("framework", framework_value)
        selected_device = params.get("device")
        if selected_framework is None:
            selected_framework = Frameworks(os.environ.get("DECENT_BENCH_BACKEND", Frameworks.NUMPY.value))
        elif not isinstance(selected_framework, Frameworks):
            selected_framework = Frameworks(selected_framework)
        if selected_device is None:
            selected_device = Devices(os.environ.get("DECENT_BENCH_DEVICE", Devices.CPU.value))
        available, reason = _backend_available(selected_framework, selected_device)
        if not available:
            pytest.skip(reason or f"{selected_framework.value}/{selected_device.value} unavailable")
        iop.set_backend(selected_framework, selected_device)
        if "no_auto_configure" not in markers:
            configure(selected_framework, selected_device)

    yield
    reset_backends()
