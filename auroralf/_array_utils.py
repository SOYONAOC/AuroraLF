from __future__ import annotations

from numbers import Real

import numpy as np


def validate_real_array_members(name: str, value: object) -> None:
    message = f"{name} must contain real non-boolean values"
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_) or np.issubdtype(
            value.dtype,
            np.complexfloating,
        ):
            raise TypeError(message)
        if np.issubdtype(value.dtype, np.integer) or np.issubdtype(
            value.dtype,
            np.floating,
        ):
            return
        if value.dtype != np.dtype(object):
            raise TypeError(message)
        members = value.flat
    elif isinstance(value, (list, tuple)):
        members = value
    else:
        members = (value,)
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in members
    ):
        raise TypeError(message)


def immutable_array(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.flags.writeable = False
    return result
