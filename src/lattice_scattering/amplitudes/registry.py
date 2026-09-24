"""Registration interface for K-matrix parameterisations.

Adding a paper's parameterisation should not mean editing the kernel, the runner and every
test.  A model is anything that supplies :meth:`k` and, optionally,
:meth:`inverse` / :meth:`s_breakpoints`; :func:`register` names it and records the reading
it implements, and :func:`build` turns a plain dict of printed parameters into a model.

The contract a model must satisfy is small on purpose:

``k(s)``
    Real symmetric ``K(s)``.  A real ``s`` must give a real array; the legacy kernels reject
    complex dtypes rather than discarding an imaginary part, so the dtype has to follow the
    input.  A complex ``s`` is allowed and gives a complex array.
``inverse(s)``
    ``K(s)^{-1}``, raising on a singular ``K`` instead of regularising it.
``s_breakpoints()``
    Real singular points of the *inverse* form -- the bare pole and any K zero.  The scan
    splits at these, because the determinant flips sign across them without passing through
    zero.  Returning ``()`` is legal but means such a point inside a scan window will look
    like a level; see ``docs/conventions.md`` for the scan limitation.

Registering a model makes it reachable by name from a transcribed row, and
:func:`registered` lists what exists so a missing one is visible rather than silently
unsupported.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = ["register", "registered", "specification", "build", "UnknownModel"]

#: name -> (factory, one-line description, required parameter names)
_REGISTRY: dict[str, tuple[Callable[..., Any], str, tuple[str, ...]]] = {}


class UnknownModel(KeyError):
    """Raised when a name is not registered, listing what is."""

    def __init__(self, name: str):
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        super().__init__(f"unknown amplitude model {name!r}; registered: {known}")
        self.name = name


def register(name: str, *, description: str = "", required: tuple[str, ...] = (),
             replace: bool = False):
    """Decorator registering ``factory`` under ``name``.

    ``required`` names the printed parameters the factory reads, so a caller can tell a
    missing column from a refused reading.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("model name must be a non-empty string")

    def decorate(factory):
        if name in _REGISTRY and not replace:
            raise ValueError(f"model {name!r} is already registered; pass replace=True to override")
        _REGISTRY[name] = (factory, description, tuple(required))
        return factory

    return decorate


def registered() -> dict[str, dict]:
    """Every registered model, with its description and required parameters."""
    return {
        name: {"description": description, "required": list(required)}
        for name, (_, description, required) in sorted(_REGISTRY.items())
    }


def specification(name: str) -> tuple[Callable[..., Any], str, tuple[str, ...]]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownModel(name) from None


def build(name: str, parameters, *, reading: str | None = None):
    """Construct the model called ``name`` from a mapping of printed parameters.

    Two factory shapes are recognised, and no signature introspection is involved because
    guessing from ``co_varnames`` silently mis-classified the classes registered by decorator:

    * a class whose ``__init__`` takes keyword parameters -- built as
      ``factory(**parameters)`` after checking the required keys are present;
    * a callable taking ``(parameters, reading=None)``, for factories that need to interpret
      the raw mapping themselves.

    A class is told apart from a function by ``isinstance(factory, type)``, which is exact.
    """
    factory, _, required = specification(name)
    if not isinstance(parameters, dict):
        raise ValueError(f"{name}: parameters must be a mapping, got {type(parameters).__name__}")
    missing = [key for key in required if key not in parameters]
    if missing:
        raise ValueError(f"{name}: missing printed parameter(s) {missing}")
    if isinstance(factory, type):
        return factory(**parameters)
    return factory(parameters, reading=reading)
