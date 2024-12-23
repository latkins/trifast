import os
from collections.abc import Callable
from typing import Any, TypeVar

from beartype import beartype
from jaxtyping import jaxtyped

F = TypeVar("F", bound=Callable[..., Any])


def disable_typechecking() -> None:
    os.environ["jaxtyping_disable"] = "1"


def typing_enabled() -> bool:
    return os.environ.get("jaxtyping_disable") == "1"


def typecheck(func: F) -> F:
    """Combines @beartype and @jaxtyped into a single decorator.

    Returns func if typing is not enabled.
    """
    if not typing_enabled():
        return func

    return jaxtyped(typechecker=beartype)(func)
