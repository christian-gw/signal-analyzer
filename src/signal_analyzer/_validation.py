"""Private validation helpers shared by the numerical modules."""

import numpy as np


def require_positive_finite_scalar(value, name):
    """Raise ValueError unless value is a positive finite scalar."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive finite scalar")
    try:
        valid = bool(np.isfinite(value) and value > 0)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a positive finite scalar")


def require_nonnegative_finite_scalar(value, name):
    """Raise ValueError unless value is a non-negative finite scalar."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    try:
        valid = bool(np.isfinite(value) and value >= 0)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a non-negative finite scalar")


def require_positive_integer(value, name):
    """Raise ValueError unless value is a positive non-boolean integer."""
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")


def require_finite_values(values, name):
    """Raise ValueError unless every array value is finite."""
    try:
        finite = np.all(np.isfinite(values))
    except TypeError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must contain only finite values")


def as_finite_1d_array(values, name="x", min_size=1, real=False):
    """Return a finite 1D array with an optional size and real-value contract."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size < min_size:
        if min_size == 1:
            raise ValueError(f"{name} must be a non-empty 1D array")
        minimum = "two" if min_size == 2 else str(min_size)
        raise ValueError(
            f"{name} must be a 1D array with at least {minimum} samples"
        )
    if real and np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    require_finite_values(array, name)
    return array
