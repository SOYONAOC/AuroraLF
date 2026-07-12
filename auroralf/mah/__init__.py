"""Public API for halo growth history generators."""

from .generator import generate_halo_histories
from .models import Cosmology, HaloHistoryResult
from .tng import (
    MAH_BACKEND_MCBRIDE,
    MAH_BACKEND_THESAN,
    MAH_BACKEND_TNG,
    MAH_BACKENDS,
    TNG_TIME_GRID_MODES,
    TNG_TIME_GRID_SNAPSHOT,
    TNG_TIME_GRID_UNIFORM_IN_T,
    generate_tng_halo_histories,
    preload_tng_mah_cache,
    validate_mah_backend,
    validate_tng_time_grid_mode,
)
from .thesan import (
    THESAN_MAH_CACHE_SCHEMA_VERSION,
    THESAN_TIME_GRID_MODES,
    THESAN_TIME_GRID_SNAPSHOT,
    THESAN_TIME_GRID_UNIFORM_IN_T,
    generate_thesan_halo_histories,
    preload_thesan_mah_cache,
    validate_thesan_time_grid_mode,
)

__all__ = [
    "Cosmology",
    "HaloHistoryResult",
    "MAH_BACKEND_MCBRIDE",
    "MAH_BACKEND_THESAN",
    "MAH_BACKEND_TNG",
    "MAH_BACKENDS",
    "THESAN_MAH_CACHE_SCHEMA_VERSION",
    "THESAN_TIME_GRID_MODES",
    "THESAN_TIME_GRID_SNAPSHOT",
    "THESAN_TIME_GRID_UNIFORM_IN_T",
    "TNG_TIME_GRID_MODES",
    "TNG_TIME_GRID_SNAPSHOT",
    "TNG_TIME_GRID_UNIFORM_IN_T",
    "generate_halo_histories",
    "generate_thesan_halo_histories",
    "generate_tng_halo_histories",
    "preload_thesan_mah_cache",
    "preload_tng_mah_cache",
    "validate_mah_backend",
    "validate_thesan_time_grid_mode",
    "validate_tng_time_grid_mode",
]
