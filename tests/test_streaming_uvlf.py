from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import gc
import math
import weakref

import numpy as np
import pytest

from auroralf.uvlf.streaming import (
    HistogramRejectionDiagnostics,
    WeightedHistogramAccumulator,
    WeightedHistogramResult,
)


def test_rejection_diagnostics_are_frozen_nonnegative_and_exhaustive() -> None:
    diagnostics = HistogramRejectionDiagnostics(
        total_samples=10,
        accepted_samples=4,
        rejected_nonfinite_value=1,
        rejected_nonfinite_weight=2,
        rejected_external_mask=1,
        out_of_range=2,
    )

    assert diagnostics.total_samples == 10
    with pytest.raises(FrozenInstanceError):
        diagnostics.total_samples = 9  # type: ignore[misc]
    with pytest.raises(ValueError, match="sum.*total_samples"):
        HistogramRejectionDiagnostics(10, 5, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        HistogramRejectionDiagnostics(1, -1, 1, 0, 0, 1)
    with pytest.raises(TypeError, match="total_samples"):
        HistogramRejectionDiagnostics(True, 0, 0, 0, 0, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "edges",
    [
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([False, True]),
        np.array(["0", "1"]),
        np.array([0.0 + 0.0j, 1.0 + 0.0j]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_accumulator_rejects_invalid_edges_before_cast(edges: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="edges"):
        WeightedHistogramAccumulator(edges)


def test_accumulator_edges_are_defensive_irreversibly_read_only_copy() -> None:
    edges = np.array([0.0, 1.0, 2.0])
    accumulator = WeightedHistogramAccumulator(edges)
    edges[0] = -99.0

    np.testing.assert_array_equal(accumulator.edges, [0.0, 1.0, 2.0])
    assert not accumulator.edges.flags.writeable
    with pytest.raises(ValueError):
        accumulator.edges.flags.writeable = True


@pytest.mark.parametrize("require_positive", [1, np.bool_(True), "true"])
def test_accumulator_requires_exact_bool_positive_mode(require_positive: object) -> None:
    with pytest.raises(TypeError, match="require_positive_values"):
        WeightedHistogramAccumulator(
            np.array([0.0, 1.0]),
            require_positive_values=require_positive,  # type: ignore[arg-type]
        )


def test_update_rejection_precedence_and_histogram_boundary_semantics() -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0, 2.0]))
    values = np.array(
        [np.nan, 0.5, 0.5, -1.0, 0.0, 1.0, 2.0, np.nextafter(2.0, np.inf)]
    )
    weights = np.array([np.nan, np.nan, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    external_mask = np.array([False, False, False, True, True, True, True, True])

    accumulator.update(values, weights, valid_mask=external_mask)
    result = accumulator.finalize()

    # value nonfinite -> weight nonfinite -> external mask -> out-of-range -> accepted
    assert result.diagnostics == HistogramRejectionDiagnostics(
        total_samples=8,
        accepted_samples=3,
        rejected_nonfinite_value=1,
        rejected_nonfinite_weight=1,
        rejected_external_mask=1,
        out_of_range=2,
    )
    # np.histogram: [left, right), with the final right edge included.
    np.testing.assert_array_equal(result.raw_counts, [1, 2])
    np.testing.assert_array_equal(result.weighted_counts_per_mpc3, [2.0, 7.0])
    np.testing.assert_array_equal(result.weight_squared_counts_per_mpc6, [4.0, 25.0])


def test_positive_value_mode_counts_nonpositive_as_external_rejection_once() -> None:
    accumulator = WeightedHistogramAccumulator(
        np.array([0.0, 1.0, 2.0]),
        require_positive_values=True,
    )
    accumulator.update(
        np.array([np.nan, -1.0, 0.0, 0.5, np.inf]),
        np.ones(5),
        valid_mask=np.array([True, False, True, True, True]),
    )

    result = accumulator.finalize()

    assert result.diagnostics == HistogramRejectionDiagnostics(
        total_samples=5,
        accepted_samples=1,
        rejected_nonfinite_value=2,
        rejected_nonfinite_weight=0,
        rejected_external_mask=2,
        out_of_range=0,
    )


def test_finite_negative_weight_is_immediate_scientific_error_even_if_value_rejected() -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="weights.*negative"):
        accumulator.update(np.array([np.nan]), np.array([-1.0]))

    assert accumulator.finalize().diagnostics.total_samples == 0


@pytest.mark.parametrize(
    ("values", "weights", "mask", "match"),
    [
        (np.array([], dtype=float), np.array([], dtype=float), None, "non-empty"),
        (np.array([0.5]), np.array([1.0, 2.0]), None, "same shape"),
        (np.array([[0.5]]), np.array([[1.0]]), None, "1D"),
        (np.array(["0.5"]), np.array([1.0]), None, "values"),
        (np.array([0.5 + 0.0j]), np.array([1.0]), None, "values"),
        (np.array([True]), np.array([1.0]), None, "values"),
        (np.array([0.5]), np.array(["1.0"]), None, "weights"),
        (np.array([0.5]), np.array([1.0 + 0.0j]), None, "weights"),
        (np.array([0.5]), np.array([1.0]), np.array([1]), "valid_mask"),
        (np.array([0.5]), np.array([1.0]), np.array([True, False]), "same shape"),
    ],
)
def test_update_rejects_invalid_shapes_and_types_before_cast(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None,
    match: str,
) -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0]))

    with pytest.raises((TypeError, ValueError), match=match):
        accumulator.update(values, weights, valid_mask=mask)


def test_single_and_chunked_updates_are_exact_for_same_sample_order() -> None:
    rng = np.random.default_rng(314159)
    edges = np.linspace(-3.0, 3.0, 31)
    values = rng.normal(size=2_000)
    weights = rng.uniform(0.0, 2.0, size=2_000)
    mask = rng.random(2_000) > 0.1
    values[::101] = np.nan
    weights[::137] = np.nan

    single = WeightedHistogramAccumulator(edges)
    single.update(values, weights, valid_mask=mask)
    expected = single.finalize()

    for boundaries in (
        (0, 1, 2_000),
        (0, 17, 103, 997, 2_000),
        tuple(range(0, 2_001, 100)),
    ):
        chunked = WeightedHistogramAccumulator(edges)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            chunked.update(
                values[start:stop],
                weights[start:stop],
                valid_mask=mask[start:stop],
            )
        actual = chunked.finalize()
        np.testing.assert_array_equal(actual.raw_counts, expected.raw_counts)
        np.testing.assert_array_equal(
            actual.weighted_counts_per_mpc3,
            expected.weighted_counts_per_mpc3,
        )
        np.testing.assert_array_equal(
            actual.weight_squared_counts_per_mpc6,
            expected.weight_squared_counts_per_mpc6,
        )
        assert actual.diagnostics == expected.diagnostics


def test_deterministic_random_reference_matches_legacy_numpy_histograms() -> None:
    rng = np.random.default_rng(2718)
    edges = np.linspace(-4.0, 4.0, 21)
    values = rng.normal(size=10_000)
    weights = rng.lognormal(mean=-1.0, sigma=0.5, size=10_000)
    valid_mask = rng.random(10_000) > 0.2
    values[::211] = np.nan
    weights[::307] = np.nan
    valid = np.isfinite(values) & np.isfinite(weights) & valid_mask

    accumulator = WeightedHistogramAccumulator(edges)
    accumulator.update(values, weights, valid_mask=valid_mask)
    result = accumulator.finalize()

    raw, _ = np.histogram(values[valid], bins=edges)
    weighted, _ = np.histogram(values[valid], bins=edges, weights=weights[valid])
    squared, _ = np.histogram(values[valid], bins=edges, weights=np.square(weights[valid]))
    sigma = np.sqrt(squared)
    effective = np.divide(
        np.square(weighted),
        squared,
        out=np.zeros_like(weighted),
        where=squared > 0.0,
    )
    widths = np.diff(edges)

    np.testing.assert_array_equal(result.raw_counts, raw)
    # np.add.at preserves sample order; np.histogram uses block summation, so
    # their floating sums agree to accumulation-scale machine precision.
    np.testing.assert_allclose(result.weighted_counts_per_mpc3, weighted, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(result.weight_squared_counts_per_mpc6, squared, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(result.weighted_count_sigma_per_mpc3, sigma, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(result.effective_counts, effective, rtol=2e-12, atol=0.0)
    np.testing.assert_allclose(result.phi_per_mpc3_per_unit, weighted / widths, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(result.phi_sigma_per_mpc3_per_unit, sigma / widths, rtol=1e-12, atol=0.0)


def test_merge_tree_matches_single_update_and_does_not_modify_sources() -> None:
    rng = np.random.default_rng(42)
    edges = np.linspace(-2.0, 2.0, 17)
    values = rng.uniform(-3.0, 3.0, size=1_003)
    weights = rng.uniform(0.0, 1.0, size=1_003)
    full = WeightedHistogramAccumulator(edges)
    full.update(values, weights)

    leaves: list[WeightedHistogramAccumulator] = []
    for indices in np.array_split(np.arange(values.size), 7):
        leaf = WeightedHistogramAccumulator(edges)
        leaf.update(values[indices], weights[indices])
        leaves.append(leaf)
    other_before = leaves[1].finalize()
    leaves[0].merge(leaves[1])
    np.testing.assert_array_equal(
        leaves[1].finalize().weighted_counts_per_mpc3,
        other_before.weighted_counts_per_mpc3,
    )
    leaves[2].merge(leaves[3])
    leaves[4].merge(leaves[5])
    leaves[0].merge(leaves[2])
    leaves[4].merge(leaves[6])
    leaves[0].merge(leaves[4])

    merged = leaves[0].finalize()
    expected = full.finalize()
    np.testing.assert_array_equal(merged.raw_counts, expected.raw_counts)
    np.testing.assert_allclose(merged.weighted_counts_per_mpc3, expected.weighted_counts_per_mpc3)
    np.testing.assert_allclose(
        merged.weight_squared_counts_per_mpc6,
        expected.weight_squared_counts_per_mpc6,
    )
    assert merged.diagnostics == expected.diagnostics


def test_merge_rejects_self_and_incompatible_accumulators() -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="itself"):
        accumulator.merge(accumulator)
    with pytest.raises(ValueError, match="edges"):
        accumulator.merge(WeightedHistogramAccumulator(np.array([0.0, 1.5, 2.0])))
    with pytest.raises(ValueError, match="require_positive_values"):
        accumulator.merge(
            WeightedHistogramAccumulator(
                np.array([0.0, 1.0, 2.0]),
                require_positive_values=True,
            )
        )
    with pytest.raises(TypeError, match="WeightedHistogramAccumulator"):
        accumulator.merge(object())  # type: ignore[arg-type]


def _snapshot_accumulator(
    accumulator: WeightedHistogramAccumulator,
) -> tuple[tuple[bytes, bytes, bytes], HistogramRejectionDiagnostics]:
    return (
        (
            accumulator.weighted_sum.tobytes(),
            accumulator.weighted_square_sum.tobytes(),
            accumulator.sample_count.tobytes(),
        ),
        accumulator.finalize().diagnostics,
    )


def _assert_accumulator_snapshot_unchanged(
    accumulator: WeightedHistogramAccumulator,
    snapshot: tuple[tuple[bytes, bytes, bytes], HistogramRejectionDiagnostics],
) -> None:
    arrays, diagnostics = snapshot
    assert accumulator.weighted_sum.tobytes() == arrays[0]
    assert accumulator.weighted_square_sum.tobytes() == arrays[1]
    assert accumulator.sample_count.tobytes() == arrays[2]
    assert accumulator.finalize().diagnostics == diagnostics


def test_merge_rejects_shallow_copy_alias_without_mutating_either_side() -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0, 2.0]))
    accumulator.update(np.array([0.25, 1.25]), np.array([2.0, 3.0]))
    shallow = copy.copy(accumulator)
    left_before = _snapshot_accumulator(accumulator)
    right_before = _snapshot_accumulator(shallow)

    with pytest.raises(ValueError, match="share memory"):
        accumulator.merge(shallow)

    _assert_accumulator_snapshot_unchanged(accumulator, left_before)
    _assert_accumulator_snapshot_unchanged(shallow, right_before)


@pytest.mark.parametrize(
    ("left_name", "right_name"),
    [
        (left_name, right_name)
        for left_name in ("weighted_sum", "weighted_square_sum", "sample_count")
        for right_name in ("weighted_sum", "weighted_square_sum", "sample_count")
    ],
)
def test_merge_rejects_all_same_and_cross_field_manual_aliases_transactionally(
    left_name: str,
    right_name: str,
) -> None:
    left = WeightedHistogramAccumulator(np.array([0.0, 1.0, 2.0]))
    right = WeightedHistogramAccumulator(np.array([0.0, 1.0, 2.0]))
    source = getattr(left, left_name)
    target_dtype = getattr(right, right_name).dtype
    setattr(right, right_name, source.view(target_dtype))
    assert np.shares_memory(getattr(left, left_name), getattr(right, right_name))
    left_before = _snapshot_accumulator(left)
    right_before = _snapshot_accumulator(right)

    with pytest.raises(ValueError, match="share memory"):
        left.merge(right)

    _assert_accumulator_snapshot_unchanged(left, left_before)
    _assert_accumulator_snapshot_unchanged(right, right_before)


def test_empty_finalize_is_valid_but_empty_update_is_not() -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0, 2.0]))

    result = accumulator.finalize()

    np.testing.assert_array_equal(result.raw_counts, [0, 0])
    assert result.diagnostics.total_samples == 0
    with pytest.raises(ValueError, match="non-empty"):
        accumulator.update(np.array([]), np.array([]))


def test_finalize_rejects_corrupted_nonfinite_or_negative_state() -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0]))
    accumulator.weighted_sum[0] = np.inf
    with pytest.raises(ValueError, match="weighted_sum.*finite"):
        accumulator.finalize()

    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0]))
    accumulator.sample_count[0] = -1
    with pytest.raises(ValueError, match="sample_count.*non-negative"):
        accumulator.finalize()


def test_result_arrays_are_irreversibly_read_only() -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0, 2.0]))
    accumulator.update(np.array([0.25, 1.25]), np.array([2.0, 3.0]))

    result = accumulator.finalize()

    assert isinstance(result, WeightedHistogramResult)
    for array in (
        result.edges,
        result.centers,
        result.width,
        result.raw_counts,
        result.weighted_counts_per_mpc3,
        result.weight_squared_counts_per_mpc6,
        result.weighted_count_sigma_per_mpc3,
        result.effective_counts,
        result.phi_per_mpc3_per_unit,
        result.phi_sigma_per_mpc3_per_unit,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_edges_with_overflowing_width_fail_without_runtime_warning() -> None:
    with pytest.raises(ValueError, match="edges.*width.*finite.*positive"):
        WeightedHistogramAccumulator(np.array([-1.0e308, 1.0e308]))


def test_finalize_uses_stable_finite_center_for_large_positive_edges() -> None:
    edges = np.array([1.6e308, 1.7e308])
    accumulator = WeightedHistogramAccumulator(edges)

    result = accumulator.finalize()

    expected_width = edges[1] - edges[0]
    expected_center = edges[0] + 0.5 * expected_width
    assert np.isfinite(result.width[0])
    assert np.isfinite(result.centers[0])
    assert result.width[0] == expected_width
    assert result.centers[0] == expected_center


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("centers", np.array([np.inf, 1.5]), "centers.*finite"),
        ("width", np.array([1.0, 0.0]), "width.*finite.*positive"),
        ("width", np.array([1.0, np.inf]), "width.*finite.*positive"),
        ("centers", np.array([0.5, 1.6]), "centers.*stable"),
        ("width", np.array([1.0, 1.1]), "width.*diff"),
    ],
)
def test_weighted_histogram_result_rejects_invalid_or_inconsistent_axes(
    field: str,
    value: np.ndarray,
    match: str,
) -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0, 2.0]))
    result = accumulator.finalize()
    values = {
        "edges": result.edges,
        "centers": result.centers,
        "width": result.width,
        "raw_counts": result.raw_counts,
        "weighted_counts_per_mpc3": result.weighted_counts_per_mpc3,
        "weight_squared_counts_per_mpc6": result.weight_squared_counts_per_mpc6,
        "weighted_count_sigma_per_mpc3": result.weighted_count_sigma_per_mpc3,
        "effective_counts": result.effective_counts,
        "phi_per_mpc3_per_unit": result.phi_per_mpc3_per_unit,
        "phi_sigma_per_mpc3_per_unit": result.phi_sigma_per_mpc3_per_unit,
        "diagnostics": result.diagnostics,
    }
    values[field] = value

    with pytest.raises(ValueError, match=match):
        WeightedHistogramResult(**values)


def test_weighted_histogram_result_rejects_direct_overflowing_edge_width() -> None:
    accumulator = WeightedHistogramAccumulator(np.array([0.0, 1.0]))
    result = accumulator.finalize()
    with pytest.raises(ValueError, match="edges.*width.*finite.*positive"):
        WeightedHistogramResult(
            edges=np.array([-1.0e308, 1.0e308]),
            centers=np.array([0.0]),
            width=np.array([np.inf]),
            raw_counts=result.raw_counts,
            weighted_counts_per_mpc3=result.weighted_counts_per_mpc3,
            weight_squared_counts_per_mpc6=result.weight_squared_counts_per_mpc6,
            weighted_count_sigma_per_mpc3=result.weighted_count_sigma_per_mpc3,
            effective_counts=result.effective_counts,
            phi_per_mpc3_per_unit=result.phi_per_mpc3_per_unit,
            phi_sigma_per_mpc3_per_unit=result.phi_sigma_per_mpc3_per_unit,
            diagnostics=result.diagnostics,
        )


def test_wide_dynamic_range_weights_match_per_bin_math_fsum_reference() -> None:
    rng = np.random.default_rng(20260711)
    edges = np.linspace(-3.0, 3.0, 25)
    values = rng.uniform(edges[0], edges[-1], size=50_000)
    weights = np.power(10.0, rng.uniform(-12.0, 12.0, size=values.size))
    permutation = rng.permutation(values.size)
    values = values[permutation]
    weights = weights[permutation]
    accumulator = WeightedHistogramAccumulator(edges)
    for indices in np.array_split(np.arange(values.size), 19):
        accumulator.update(values[indices], weights[indices])

    result = accumulator.finalize()
    bin_indices = np.searchsorted(edges, values, side="right") - 1
    bin_indices[values == edges[-1]] = edges.size - 2
    reference_weighted = np.array(
        [
            math.fsum(float(weight) for weight in weights[bin_indices == bin_index])
            for bin_index in range(edges.size - 1)
        ]
    )
    reference_squared = np.array(
        [
            math.fsum(float(weight) ** 2 for weight in weights[bin_indices == bin_index])
            for bin_index in range(edges.size - 1)
        ]
    )
    counts = np.bincount(bin_indices, minlength=edges.size - 1)
    eps = np.finfo(float).eps
    weighted_tolerance = (
        8.0 * eps * np.maximum(counts, 1) * np.maximum(reference_weighted, 1.0)
    )
    squared_tolerance = (
        8.0 * eps * np.maximum(counts, 1) * np.maximum(reference_squared, 1.0)
    )

    np.testing.assert_array_equal(result.raw_counts, counts)
    assert np.all(np.abs(result.weighted_counts_per_mpc3 - reference_weighted) <= weighted_tolerance)
    assert np.all(
        np.abs(result.weight_squared_counts_per_mpc6 - reference_squared)
        <= squared_tolerance
    )


def test_large_streaming_state_is_o_bins_and_retains_no_input_references() -> None:
    edges = np.linspace(-5.0, 5.0, 101)
    accumulator = WeightedHistogramAccumulator(edges)
    last_values_ref: weakref.ReferenceType[np.ndarray] | None = None
    for start in range(0, 1_000_000, 10_000):
        values = np.linspace(-6.0, 6.0, 10_000) + start * 0.0
        weights = np.ones(10_000)
        accumulator.update(values, weights)
        last_values_ref = weakref.ref(values)
    del values, weights
    gc.collect()

    assert last_values_ref is not None and last_values_ref() is None
    state_nbytes = sum(
        array.nbytes
        for array in (
            accumulator.edges,
            accumulator.weighted_sum,
            accumulator.weighted_square_sum,
            accumulator.sample_count,
        )
    )
    assert state_nbytes < 10_000
    assert accumulator.finalize().diagnostics.total_samples == 1_000_000
