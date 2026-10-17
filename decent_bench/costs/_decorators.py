from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast, get_type_hints

from decent_array import Array

T = TypeVar("T", bound=Callable[..., Any])


def autodecorate_cost_method[T: Callable[..., Any]](superclass_method: T) -> Callable[[Callable[..., Any]], T]:
    """Adapt native cost methods to the public cost method contract."""
    try:
        return_type_annotation = get_type_hints(superclass_method).get("return")
    except (AttributeError, KeyError, NameError, TypeError):
        return_type_annotation = None

    def decorator(func: Callable[..., Any]) -> T:
        try:
            input_annotations = get_type_hints(func)
            native_input = input_annotations.get("x") is not Array
        except (NameError, TypeError):
            native_input = True

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if args:
                x_like = args[0]
            elif "x" in kwargs:
                x_like = kwargs["x"]
            else:
                raise ValueError("First argument must be 'x' for autodecorate_cost_method to work.")

            new_args = [arg.value if native_input and isinstance(arg, Array) else arg for arg in args]
            new_kwargs = {
                key: value.value if native_input and isinstance(value, Array) else value
                for key, value in kwargs.items()
            }
            result = func(self, *new_args, **new_kwargs)

            if return_type_annotation is Array and isinstance(x_like, Array):
                if isinstance(result, Array):
                    return result
                if not isinstance(result, (list, tuple)):
                    return Array(result)
            return result

        return cast("T", wrapper)

    return decorator
