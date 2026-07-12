from __future__ import annotations

import numpy as np
import pytest

import auroralf.mah.generator as generator
import auroralf.mah.sampling as sampling


def test_sample_parameters_rejects_unknown_sampler_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailOnRngAccess:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"RNG must not be accessed for an invalid sampler: {name}")

    def fail_sampling(*args: object, **kwargs: object) -> None:
        raise AssertionError("No sampling path may run for an invalid sampler")

    monkeypatch.setattr(sampling, "sample_mcbride_appendix_a", fail_sampling)
    monkeypatch.setattr(sampling, "estimate_gaussian_approximation", fail_sampling)

    with pytest.raises(ValueError, match="mcbride_typo") as error:
        sampling.sample_parameters(
            mass_ref=1.0e10,
            size=2,
            sampler="mcbride_typo",
            rng=FailOnRngAccess(),  # type: ignore[arg-type]
            pilot_samples=2,
        )

    message = str(error.value)
    assert "mcbride" in message
    assert "gaussian_approximation" in message


@pytest.mark.parametrize(
    ("sampler", "expected"),
    [
        (" mcBride ", "mcbride"),
        ("GAUSSIAN_APPROXIMATION", "gaussian_approximation"),
    ],
)
def test_validate_parameter_sampler_normalizes_supported_names(sampler: str, expected: str) -> None:
    assert sampling.validate_parameter_sampler(sampler) == expected


def test_sample_parameters_rejects_unimplemented_validated_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future_sampler = "future_sampler"

    def fail_sampling(*args: object, **kwargs: object) -> None:
        raise AssertionError("An unimplemented validated sampler must not use an existing sampling path")

    monkeypatch.setattr(
        sampling,
        "SUPPORTED_PARAMETER_SAMPLERS",
        (*sampling.SUPPORTED_PARAMETER_SAMPLERS, future_sampler),
    )
    monkeypatch.setattr(sampling, "sample_mcbride_appendix_a", fail_sampling)
    monkeypatch.setattr(sampling, "estimate_gaussian_approximation", fail_sampling)

    with pytest.raises(RuntimeError, match=future_sampler):
        sampling.sample_parameters(
            mass_ref=1.0e10,
            size=2,
            sampler=future_sampler,
            rng=np.random.default_rng(8),
            pilot_samples=2,
        )


def test_sample_parameters_dispatches_mcbride_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = np.array([[-1.0, 0.0], [-2.0, 0.25]])
    calls: list[tuple[float, int, object]] = []
    rng = object()

    def fake_mcbride(mass_ref: float, size: int, received_rng: object) -> np.ndarray:
        calls.append((mass_ref, size, received_rng))
        return expected

    def fail_gaussian(*args: object, **kwargs: object) -> None:
        raise AssertionError("Gaussian approximation must not run for the McBride sampler")

    monkeypatch.setattr(sampling, "sample_mcbride_appendix_a", fake_mcbride)
    monkeypatch.setattr(sampling, "estimate_gaussian_approximation", fail_gaussian)

    draws, approximation = sampling.sample_parameters(
        mass_ref=1.0e10,
        size=2,
        sampler=" McBride ",
        rng=rng,  # type: ignore[arg-type]
        pilot_samples=3,
    )

    assert draws is expected
    assert approximation is None
    assert calls == [(1.0e10, 2, rng)]


def test_sample_parameters_dispatches_gaussian_approximation(monkeypatch: pytest.MonkeyPatch) -> None:
    approximation = sampling.GaussianApproximation(
        mean=np.array([-1.5, 0.2]),
        covariance=np.array([[0.1, 0.01], [0.01, 0.02]]),
    )
    expected = np.array([[-1.4, 0.21], [-1.6, 0.19]])
    estimate_calls: list[tuple[float, object, int]] = []

    class RecordingRng:
        def __init__(self) -> None:
            self.draw_calls: list[tuple[np.ndarray, np.ndarray, int]] = []

        def multivariate_normal(
            self,
            mean: np.ndarray,
            covariance: np.ndarray,
            *,
            size: int,
        ) -> np.ndarray:
            self.draw_calls.append((mean, covariance, size))
            return expected

    rng = RecordingRng()

    def fake_estimate(
        mass_ref: float,
        received_rng: object,
        pilot_samples: int,
    ) -> sampling.GaussianApproximation:
        estimate_calls.append((mass_ref, received_rng, pilot_samples))
        return approximation

    def fail_mcbride(*args: object, **kwargs: object) -> None:
        raise AssertionError("Exact McBride draw must not run for the Gaussian sampler")

    monkeypatch.setattr(sampling, "estimate_gaussian_approximation", fake_estimate)
    monkeypatch.setattr(sampling, "sample_mcbride_appendix_a", fail_mcbride)

    draws, returned_approximation = sampling.sample_parameters(
        mass_ref=2.0e10,
        size=2,
        sampler=" GAUSSIAN_APPROXIMATION ",
        rng=rng,  # type: ignore[arg-type]
        pilot_samples=4,
    )

    assert draws is expected
    assert returned_approximation is approximation
    assert estimate_calls == [(2.0e10, rng, 4)]
    assert len(rng.draw_calls) == 1
    mean, covariance, size = rng.draw_calls[0]
    assert mean is approximation.mean
    assert covariance is approximation.covariance
    assert size == 2


def test_generate_halo_histories_records_canonical_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    received_samplers: list[str] = []

    def fake_sample_parameters(
        mass_ref: float,
        size: int,
        sampler: str,
        rng: np.random.Generator,
        pilot_samples: int,
    ) -> tuple[np.ndarray, None]:
        del mass_ref, rng, pilot_samples
        received_samplers.append(sampler)
        return np.column_stack((np.full(size, -1.0), np.zeros(size))), None

    monkeypatch.setattr(generator, "sample_parameters", fake_sample_parameters)

    result = generator.generate_halo_histories(
        n_tracks=1,
        z_final=6.0,
        Mh_final=1.0e10,
        cosmology=generator.Cosmology(),
        z_start_max=7.0,
        M_min=1.0,
        dz=1.0,
        sampler=" McBride ",
    )

    assert received_samplers == ["mcbride"]
    assert result.metadata["sampler"] == "mcbride"
