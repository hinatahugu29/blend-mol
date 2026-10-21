"""Direct end-to-end comparison of Python and Rust partial-UFF steps."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_uff_scaling import measure, replicated_molecule
from molfile import read_v2000
from native_backend import backend_status, set_backend_preference


def benchmark(repeats: int = 3) -> dict:
    source = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    molecules = [replicated_molecule(source, copies) for copies in (1, 4, 10)]
    cases = {}
    for backend in ("python", "rust"):
        set_backend_preference(backend)
        cases[backend] = measure(
            "replicated explicit triphenylene (torsion/inversion heavy)",
            molecules,
            repeats,
        )["results"]
    comparisons = []
    for python, rust in zip(cases["python"], cases["rust"]):
        comparisons.append({
            "atom_count": python["atom_count"],
            "python_median_ms": python["median_ms"],
            "rust_median_ms": rust["median_ms"],
            "speedup": python["median_ms"] / rust["median_ms"],
        })
    set_backend_preference("auto")
    return {
        "benchmark": "Blend Mol Python/Rust full-step comparison",
        "repeats": repeats,
        "native_status": backend_status(),
        "comparisons": comparisons,
        "raw": cases,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
