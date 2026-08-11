from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

import numpy as np

from auroralf._array_utils import (
    immutable_array as _immutable_array,
    validate_real_array_members as _validate_real_array_members,
)


def _strict_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer and boolean values are not allowed")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _working_real_1d(name: str, value: object, *, allow_empty: bool) -> np.ndarray:
    _validate_real_array_members(name, value)
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _working_integer_1d(name: str, value: object, *, allow_empty: bool) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_) or not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"{name} must contain integer non-boolean values")
    elif isinstance(value, (list, tuple)):
        if any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
            for item in value
        ):
            raise TypeError(f"{name} must contain integer non-boolean values")
    else:
        raise TypeError(f"{name} must contain integer non-boolean values")
    array = np.array(value, dtype=np.int64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _working_bool_1d(name: str, value: object, *, allow_empty: bool) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype(bool):
            raise TypeError(f"{name} must have exact boolean dtype")
    elif isinstance(value, (list, tuple)):
        if any(not isinstance(item, (bool, np.bool_)) for item in value):
            raise TypeError(f"{name} must contain only boolean values")
    else:
        raise TypeError(f"{name} must be a boolean array")
    array = np.array(value, dtype=bool, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _stable_axis_geometry(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        width = np.diff(edges)
    if not np.all(np.isfinite(width)) or np.any(width <= 0.0):
        raise ValueError("edges bin width must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        centers = edges[:-1] + 0.5 * width
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers computed by the stable edge formula must be finite")
    return width, centers


@dataclass(frozen=True, slots=True)
class HistogramRejectionDiagnostics:
    total_samples: int
    accepted_samples: int
    rejected_nonfinite_value: int
    rejected_nonfinite_weight: int
    rejected_external_mask: int
    out_of_range: int

    def __post_init__(self) -> None:
        normalized: dict[str, int] = {}
        for field in fields(self):
            normalized[field.name] = _strict_nonnegative_int(
                field.name,
                getattr(self, field.name),
            )
        classified = (
            normalized["accepted_samples"]
            + normalized["rejected_nonfinite_value"]
            + normalized["rejected_nonfinite_weight"]
            + normalized["rejected_external_mask"]
            + normalized["out_of_range"]
        )
        if classified != normalized["total_samples"]:
            raise ValueError("classified sample counts must sum exactly to total_samples")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class WeightedHistogramResult:
    edges: np.ndarray
    centers: np.ndarray
    width: np.ndarray
    raw_counts: np.ndarray
    weighted_counts_per_mpc3: np.ndarray
    weight_squared_counts_per_mpc6: np.ndarray
    weighted_count_sigma_per_mpc3: np.ndarray
    effective_counts: np.ndarray
    phi_per_mpc3_per_unit: np.ndarray
    phi_sigma_per_mpc3_per_unit: np.ndarray
    diagnostics: HistogramRejectionDiagnostics

    def __post_init__(self) -> None:
        edges = _working_real_1d("edges", self.edges, allow_empty=False)
        centers = _working_real_1d("centers", self.centers, allow_empty=False)
        width = _working_real_1d("width", self.width, allow_empty=False)
        raw_counts = _working_integer_1d("raw_counts", self.raw_counts, allow_empty=False)
        weighted = _working_real_1d(
            "weighted_counts_per_mpc3",
            self.weighted_counts_per_mpc3,
            allow_empty=False,
        )
        squared = _working_real_1d(
            "weight_squared_counts_per_mpc6",
            self.weight_squared_counts_per_mpc6,
            allow_empty=False,
        )
        sigma = _working_real_1d(
            "weighted_count_sigma_per_mpc3",
            self.weighted_count_sigma_per_mpc3,
            allow_empty=False,
        )
        effective = _working_real_1d(
            "effective_counts",
            self.effective_counts,
            allow_empty=False,
        )
        phi = _working_real_1d(
            "phi_per_mpc3_per_unit",
            self.phi_per_mpc3_per_unit,
            allow_empty=False,
        )
        phi_sigma = _working_real_1d(
            "phi_sigma_per_mpc3_per_unit",
            self.phi_sigma_per_mpc3_per_unit,
            allow_empty=False,
        )
        if edges.size < 2 or not np.all(np.isfinite(edges)):
            raise ValueError("edges must contain at least two finite values")
        expected_width, expected_centers = _stable_axis_geometry(edges)
        bin_count = edges.size - 1
        bin_arrays = (
            centers,
            width,
            raw_counts,
            weighted,
            squared,
            sigma,
            effective,
            phi,
            phi_sigma,
        )
        if any(array.size != bin_count for array in bin_arrays):
            raise ValueError("all histogram result arrays must match the number of bins")
        if not np.all(np.isfinite(centers)):
            raise ValueError("centers must be finite")
        if not np.all(np.isfinite(width)) or np.any(width <= 0.0):
            raise ValueError("width must be finite and positive")
        if not np.array_equal(centers, expected_centers):
            raise ValueError("centers must equal the stable edge midpoint formula")
        if not np.array_equal(width, expected_width):
            raise ValueError("width must equal np.diff(edges)")
        if np.any(raw_counts < 0):
            raise ValueError("raw_counts must be non-negative")
        for name, array in (
            ("weighted_counts_per_mpc3", weighted),
            ("weight_squared_counts_per_mpc6", squared),
            ("weighted_count_sigma_per_mpc3", sigma),
            ("effective_counts", effective),
            ("phi_per_mpc3_per_unit", phi),
            ("phi_sigma_per_mpc3_per_unit", phi_sigma),
        ):
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.diagnostics) is not HistogramRejectionDiagnostics:
            raise TypeError("diagnostics must be exactly HistogramRejectionDiagnostics")
        if int(np.sum(raw_counts, dtype=np.int64)) != self.diagnostics.accepted_samples:
            raise ValueError("raw_counts must sum exactly to diagnostics.accepted_samples")
        for name, array in (
            ("edges", edges),
            ("centers", centers),
            ("width", width),
            ("raw_counts", raw_counts),
            ("weighted_counts_per_mpc3", weighted),
            ("weight_squared_counts_per_mpc6", squared),
            ("weighted_count_sigma_per_mpc3", sigma),
            ("effective_counts", effective),
            ("phi_per_mpc3_per_unit", phi),
            ("phi_sigma_per_mpc3_per_unit", phi_sigma),
        ):
            object.__setattr__(self, name, _immutable_array(array))


class WeightedHistogramAccumulator:
    __slots__ = (
        "_edges",
        "_require_positive_values",
        "weighted_sum",
        "weighted_square_sum",
        "sample_count",
        "_total_samples",
        "_accepted_samples",
        "_rejected_nonfinite_value",
        "_rejected_nonfinite_weight",
        "_rejected_external_mask",
        "_out_of_range",
    )

    def __init__(
        self,
        edges: np.ndarray | list[float] | tuple[float, ...],
        *,
        require_positive_values: bool = False,
    ) -> None:
        normalized_edges = _working_real_1d("edges", edges, allow_empty=False)
        if normalized_edges.size < 2:
            raise ValueError("edges must contain at least two values")
        if not np.all(np.isfinite(normalized_edges)):
            raise ValueError("edges must contain only finite values")
        _stable_axis_geometry(normalized_edges)
        if type(require_positive_values) is not bool:
            raise TypeError("require_positive_values must be exactly bool")
        self._edges = _immutable_array(normalized_edges)
        self._require_positive_values = require_positive_values
        bin_count = normalized_edges.size - 1
        self.weighted_sum = np.zeros(bin_count, dtype=float)
        self.weighted_square_sum = np.zeros(bin_count, dtype=float)
        self.sample_count = np.zeros(bin_count, dtype=np.int64)
        self._total_samples = 0
        self._accepted_samples = 0
        self._rejected_nonfinite_value = 0
        self._rejected_nonfinite_weight = 0
        self._rejected_external_mask = 0
        self._out_of_range = 0

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def require_positive_values(self) -> bool:
        return self._require_positive_values

    def update(
        self,
        values: np.ndarray | list[float] | tuple[float, ...],
        weights: np.ndarray | list[float] | tuple[float, ...],
        valid_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
    ) -> None:
        value_array = _working_real_1d("values", values, allow_empty=False)
        weight_array = _working_real_1d("weights", weights, allow_empty=False)
        if value_array.shape != weight_array.shape:
            raise ValueError("values and weights must have the same shape")
        if valid_mask is None:
            external_mask = np.ones(value_array.shape, dtype=bool)
        else:
            external_mask = _working_bool_1d(
                "valid_mask",
                valid_mask,
                allow_empty=False,
            )
            if external_mask.shape != value_array.shape:
                raise ValueError("valid_mask must have the same shape as values and weights")

        finite_negative_weight = np.isfinite(weight_array) & (weight_array < 0.0)
        if np.any(finite_negative_weight):
            raise ValueError("weights must not contain finite negative values")

        unclassified = np.ones(value_array.shape, dtype=bool)
        rejected_value = ~np.isfinite(value_array)
        unclassified &= ~rejected_value
        rejected_weight = unclassified & ~np.isfinite(weight_array)
        unclassified &= ~rejected_weight
        external_rejection_condition = ~external_mask
        if self._require_positive_values:
            external_rejection_condition |= value_array <= 0.0
        rejected_external = unclassified & external_rejection_condition
        unclassified &= ~rejected_external
        rejected_range = unclassified & (
            (value_array < self._edges[0]) | (value_array > self._edges[-1])
        )
        accepted = unclassified & ~rejected_range

        accepted_values = value_array[accepted]
        accepted_weights = weight_array[accepted]
        bin_indices = np.searchsorted(self._edges, accepted_values, side="right") - 1
        bin_indices[accepted_values == self._edges[-1]] = self.sample_count.size - 1
        np.add.at(self.sample_count, bin_indices, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(self.weighted_sum, bin_indices, accepted_weights)
            np.add.at(self.weighted_square_sum, bin_indices, np.square(accepted_weights))

        self._total_samples += int(value_array.size)
        self._accepted_samples += int(np.count_nonzero(accepted))
        self._rejected_nonfinite_value += int(np.count_nonzero(rejected_value))
        self._rejected_nonfinite_weight += int(np.count_nonzero(rejected_weight))
        self._rejected_external_mask += int(np.count_nonzero(rejected_external))
        self._out_of_range += int(np.count_nonzero(rejected_range))

    def _diagnostics(self) -> HistogramRejectionDiagnostics:
        return HistogramRejectionDiagnostics(
            total_samples=self._total_samples,
            accepted_samples=self._accepted_samples,
            rejected_nonfinite_value=self._rejected_nonfinite_value,
            rejected_nonfinite_weight=self._rejected_nonfinite_weight,
            rejected_external_mask=self._rejected_external_mask,
            out_of_range=self._out_of_range,
        )

    def _validate_state(self) -> None:
        expected_shape = (self._edges.size - 1,)
        for name, array, expected_dtype in (
            ("weighted_sum", self.weighted_sum, np.dtype(float)),
            ("weighted_square_sum", self.weighted_square_sum, np.dtype(float)),
            ("sample_count", self.sample_count, np.dtype(np.int64)),
        ):
            if not isinstance(array, np.ndarray) or array.shape != expected_shape:
                raise ValueError(f"{name} must retain shape {expected_shape}")
            if array.dtype != expected_dtype:
                raise ValueError(f"{name} must retain dtype {expected_dtype}")
        if not np.all(np.isfinite(self.weighted_sum)):
            raise ValueError("weighted_sum must remain finite")
        if np.any(self.weighted_sum < 0.0):
            raise ValueError("weighted_sum must remain non-negative")
        if not np.all(np.isfinite(self.weighted_square_sum)):
            raise ValueError("weighted_square_sum must remain finite")
        if np.any(self.weighted_square_sum < 0.0):
            raise ValueError("weighted_square_sum must remain non-negative")
        if np.any(self.sample_count < 0):
            raise ValueError("sample_count must remain non-negative")
        diagnostics = self._diagnostics()
        if int(np.sum(self.sample_count, dtype=np.int64)) != diagnostics.accepted_samples:
            raise ValueError("sample_count must sum exactly to accepted_samples")

    def merge(self, other: WeightedHistogramAccumulator) -> None:
        if type(other) is not WeightedHistogramAccumulator:
            raise TypeError("other must be exactly WeightedHistogramAccumulator")
        if other is self:
            raise ValueError("an accumulator cannot merge itself")
        self_state = (self.weighted_sum, self.weighted_square_sum, self.sample_count)
        other_state = (other.weighted_sum, other.weighted_square_sum, other.sample_count)
        if any(
            np.shares_memory(self_array, other_array)
            for self_array in self_state
            for other_array in other_state
        ):
            raise ValueError("accumulator state arrays must not share memory")
        if not np.array_equal(self._edges, other._edges):
            raise ValueError("accumulator edges must match exactly for merge")
        if self._require_positive_values != other._require_positive_values:
            raise ValueError("require_positive_values must match exactly for merge")
        self._validate_state()
        other._validate_state()
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_sum = self.weighted_sum + other.weighted_sum
            weighted_square_sum = self.weighted_square_sum + other.weighted_square_sum
            sample_count = self.sample_count + other.sample_count
        if not np.all(np.isfinite(weighted_sum)):
            raise ValueError("merged weighted_sum must be finite")
        if not np.all(np.isfinite(weighted_square_sum)):
            raise ValueError("merged weighted_square_sum must be finite")
        if np.any(sample_count < self.sample_count) or np.any(sample_count < other.sample_count):
            raise ValueError("merged sample_count overflowed int64")
        self.weighted_sum[:] = weighted_sum
        self.weighted_square_sum[:] = weighted_square_sum
        self.sample_count[:] = sample_count
        self._total_samples += other._total_samples
        self._accepted_samples += other._accepted_samples
        self._rejected_nonfinite_value += other._rejected_nonfinite_value
        self._rejected_nonfinite_weight += other._rejected_nonfinite_weight
        self._rejected_external_mask += other._rejected_external_mask
        self._out_of_range += other._out_of_range
        self._validate_state()

    def finalize(self) -> WeightedHistogramResult:
        self._validate_state()
        width, centers = _stable_axis_geometry(self._edges)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sigma = np.sqrt(self.weighted_square_sum)
            effective = np.divide(
                np.square(self.weighted_sum),
                self.weighted_square_sum,
                out=np.zeros_like(self.weighted_sum),
                where=self.weighted_square_sum > 0.0,
            )
            phi = self.weighted_sum / width
            phi_sigma = sigma / width
        for name, array in (
            ("weighted_count_sigma_per_mpc3", sigma),
            ("effective_counts", effective),
            ("phi_per_mpc3_per_unit", phi),
            ("phi_sigma_per_mpc3_per_unit", phi_sigma),
        ):
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"finalized {name} must be finite and non-negative")
        return WeightedHistogramResult(
            edges=self._edges,
            centers=centers,
            width=width,
            raw_counts=self.sample_count,
            weighted_counts_per_mpc3=self.weighted_sum,
            weight_squared_counts_per_mpc6=self.weighted_square_sum,
            weighted_count_sigma_per_mpc3=sigma,
            effective_counts=effective,
            phi_per_mpc3_per_unit=phi,
            phi_sigma_per_mpc3_per_unit=phi_sigma,
            diagnostics=self._diagnostics(),
        )


__all__ = [
    "HistogramRejectionDiagnostics",
    "WeightedHistogramAccumulator",
    "WeightedHistogramResult",
]
