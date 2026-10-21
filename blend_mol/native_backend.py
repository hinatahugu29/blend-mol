"""Optional Windows-native UFF kernels with an explicit Python fallback."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


BACKEND_ENVIRONMENT_VARIABLE = "BLEND_MOL_UFF_BACKEND"
_LIBRARY = None
_LOAD_ERROR: str | None = None
_PREFERENCE_OVERRIDE: str | None = None


def backend_preference() -> str:
    value = (_PREFERENCE_OVERRIDE or os.environ.get(BACKEND_ENVIRONMENT_VARIABLE, "auto")).strip().lower()
    return value if value in {"auto", "python", "rust"} else "auto"


def set_backend_preference(value: str) -> None:
    """Set the process-local backend choice used by Blender's scene setting."""

    global _PREFERENCE_OVERRIDE
    normalized = value.strip().lower()
    if normalized not in {"auto", "python", "rust"}:
        raise ValueError(f"Unknown UFF backend: {value}")
    _PREFERENCE_OVERRIDE = normalized


def _library_path() -> Path:
    return Path(__file__).with_name("native") / "blend_mol_uff.dll"


def _load_library():
    global _LIBRARY, _LOAD_ERROR
    if _LIBRARY is not None or _LOAD_ERROR is not None:
        return _LIBRARY
    try:
        library = ctypes.CDLL(str(_library_path()))
        library.blend_mol_uff_abi_version.argtypes = []
        library.blend_mol_uff_abi_version.restype = ctypes.c_uint32
        if library.blend_mol_uff_abi_version() != 1:
            raise RuntimeError("unsupported native UFF ABI")
        library.blend_mol_vdw_energy_gradient.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        library.blend_mol_vdw_energy_gradient.restype = ctypes.c_int
        library.blend_mol_residual_energy_gradient.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        library.blend_mol_residual_energy_gradient.restype = ctypes.c_int
        _LIBRARY = library
    except (OSError, RuntimeError) as exc:
        _LOAD_ERROR = str(exc)
    return _LIBRARY


def backend_status() -> dict[str, str | bool]:
    preference = backend_preference()
    library = None if preference == "python" else _load_library()
    return {
        "preference": preference,
        "active": "rust" if library is not None else "python",
        "native_available": library is not None,
        "detail": "available" if library is not None else (_LOAD_ERROR or "disabled by preference"),
    }


def vdw_energy_gradient(
    coordinates: list[tuple[float, float, float]],
    pairs: list[tuple[int, int, float, float]],
) -> tuple[float, list[list[float]]] | None:
    """Use Rust when enabled; return None to request the Python fallback."""

    preference = backend_preference()
    if preference == "python":
        return None
    library = _load_library()
    if library is None:
        if preference == "rust":
            raise RuntimeError(f"Rust UFF backend requested but unavailable: {_LOAD_ERROR}")
        return None
    coordinate_values = [value for point in coordinates for value in point]
    pair_atoms = [index for first, second, _depth, _distance in pairs for index in (first, second)]
    pair_parameters = [value for _first, _second, depth, distance in pairs for value in (depth, distance)]
    coordinate_array = (ctypes.c_double * len(coordinate_values))(*coordinate_values)
    pair_atom_array = (ctypes.c_uint32 * len(pair_atoms))(*pair_atoms)
    parameter_array = (ctypes.c_double * len(pair_parameters))(*pair_parameters)
    gradient_array = (ctypes.c_double * len(coordinate_values))()
    energy = ctypes.c_double()
    status = library.blend_mol_vdw_energy_gradient(
        len(coordinates), coordinate_array, len(pairs), pair_atom_array, parameter_array,
        ctypes.byref(energy), gradient_array,
    )
    if status != 0:
        if preference == "rust":
            raise RuntimeError(f"Rust UFF kernel failed with status {status}")
        return None
    gradient = [
        [gradient_array[index * 3 + axis] for axis in range(3)]
        for index in range(len(coordinates))
    ]
    return energy.value, gradient
def residual_energy_gradient(
    coordinates: list[list[float]] | list[tuple[float, float, float]],
    torsions: list[tuple[int, int, int, int, int, float, float]],
    inversions: list[tuple[int, int, int, int, float]],
    active_atoms: set[int],
    finite_difference: float,
) -> tuple[float, list[list[float]]] | None:
    """Evaluate residual energy and all requested numerical derivatives in Rust."""

    preference = backend_preference()
    if preference == "python":
        return None
    library = _load_library()
    if library is None:
        if preference == "rust":
            raise RuntimeError(f"Rust UFF backend requested but unavailable: {_LOAD_ERROR}")
        return None
    coordinate_values = [value for point in coordinates for value in point]
    torsion_atoms = [index for term in torsions for index in term[:4]]
    torsion_parameters = [
        value for term in torsions for value in (float(term[4]), term[5], term[6])
    ]
    inversion_atoms = [index for term in inversions for index in term[:4]]
    inversion_barriers = [term[4] for term in inversions]
    active = sorted(active_atoms)
    coordinate_array = (ctypes.c_double * len(coordinate_values))(*coordinate_values)
    torsion_atom_array = (ctypes.c_uint32 * len(torsion_atoms))(*torsion_atoms)
    torsion_parameter_array = (ctypes.c_double * len(torsion_parameters))(*torsion_parameters)
    inversion_atom_array = (ctypes.c_uint32 * len(inversion_atoms))(*inversion_atoms)
    inversion_barrier_array = (ctypes.c_double * len(inversion_barriers))(*inversion_barriers)
    active_array = (ctypes.c_uint32 * len(active))(*active)
    gradient_array = (ctypes.c_double * len(coordinate_values))()
    energy = ctypes.c_double()
    status = library.blend_mol_residual_energy_gradient(
        len(coordinates), coordinate_array,
        len(torsions), torsion_atom_array, torsion_parameter_array,
        len(inversions), inversion_atom_array, inversion_barrier_array,
        len(active), active_array, finite_difference,
        ctypes.byref(energy), gradient_array,
    )
    if status != 0:
        if preference == "rust":
            raise RuntimeError(f"Rust residual UFF kernel failed with status {status}")
        return None
    gradient = [
        [gradient_array[index * 3 + axis] for axis in range(3)]
        for index in range(len(coordinates))
    ]
    return energy.value, gradient
