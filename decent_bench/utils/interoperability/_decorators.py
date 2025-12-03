from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

from decent_bench.utils.array import Array
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from ._functions import to_jax, to_numpy, to_tensorflow, to_torch
from ._helpers import _framework_device_of_array, _return_array

if TYPE_CHECKING:
    from decent_bench.costs import Cost

T = TypeVar("T", bound=Callable[..., Any])
"""A generic callable type variable."""


def _get_converter(framework: SupportedFrameworks) -> Callable[[Array | Any, SupportedDevices], Any]:
    if framework == SupportedFrameworks.NUMPY:
        return to_numpy
    if framework == SupportedFrameworks.TORCH:
        return to_torch
    if framework == SupportedFrameworks.TENSORFLOW:
        return to_tensorflow
    if framework == SupportedFrameworks.JAX:
        return to_jax

    raise ValueError(f"Unsupported framework: {framework}")


def autodecorate_cost_method[T: Callable[..., Any]](superclass_method: T) -> Callable[[Callable[..., Any]], T]:
    """
    Decorate Cost methods to automatically convert input :class:`~decent_bench.utils.array.Array` args and return types.

    It automatically converts input :class:`~decent_bench.utils.array.Array` arguments
    to the cost's framework-specific array type and wraps the output based on the
    superclass method's return type annotation.

    Args:
        superclass_method: The method from the superclass (e.g., `Cost.function`) that is being overridden.

    Note:
        Does not work on __add__ or similar special methods.

    """

    def decorator(func: Callable[..., Any]) -> T:
        # Determine the expected return type from the superclass method's annotations.
        try:
            return_type_annotation = superclass_method.__annotations__["return"]
        except (AttributeError, KeyError):
            return_type_annotation = None

        @wraps(func)
        def wrapper(self: Cost, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            converter = _get_converter(self.framework)

            new_args = []
            for arg in args:
                if isinstance(arg, Array):
                    framework, _ = _framework_device_of_array(arg)
                    if framework != self.framework:
                        LOGGER.warning(
                            f"Converting array from framework {framework} to {self.framework}"
                            f" in method {func.__name__}. This may lead to unexpected behavior or performance issues."
                        )
                    new_args.append(converter(arg, self.device))
                else:
                    new_args.append(arg)

            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, Array):
                    framework, _ = _framework_device_of_array(value)
                    if framework != self.framework:
                        LOGGER.warning(
                            f"Converting array from framework {framework} to {self.framework}"
                            f" in method {func.__name__}. This may lead to unexpected behavior or performance issues."
                        )
                    new_kwargs[key] = converter(value, self.device)
                else:
                    new_kwargs[key] = value

            result = func(self, *new_args, **new_kwargs)

            if return_type_annotation is Array:
                return _return_array(result)

            return result

        # Cast the wrapper to the type of the superclass method.
        # This tells mypy that the decorated method is compatible with the superclass.
        return cast("T", wrapper)

    return decorator
