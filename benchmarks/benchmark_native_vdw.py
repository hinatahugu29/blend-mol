"""Compare the reference Python and optional Rust bond/van-der-Waals kernel."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_uff_scaling import replicated_molecule
from molfile import read_v2000
from native_backend import backend_status, set_backend_preference
from uff import _bond_vdw_energy_gradient


def measure(molecule, backend: str, repeats: int = 7) -> dict:
    set_backend_preference(backend)
    samples = []
    energy = None
    for _repeat in range(repeats):
        started = perf_counter()
        energy, _gradient, messages = _bond_vdw_energy_gradient(molecule)
        samples.append((perf_counter() - started) * 1000.0)
        if messages:
            raise RuntimeError("; ".join(messages))
    return {
        "backend": backend,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "energy_kcal_mol": energy,
    }


def benchmark() -> dict:
    source = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    cases = []
    for copies in (1, 4, 10):
        molecule = replicated_molecule(source, copies)
        python = measure(molecule, "python")
        rust = measure(molecule, "rust")
        cases.append({
            "atom_count": len(molecule.atoms),
            "python": python,
            "rust": rust,
            "speedup": python["median_ms"] / rust["median_ms"],
            "absolute_energy_difference": abs(
                python["energy_kcal_mol"] - rust["energy_kcal_mol"]
            ),
        })
    set_backend_preference("auto")
    return {
        "benchmark": "Blend Mol native bond/vdw kernel",
        "native_status": backend_status(),
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
