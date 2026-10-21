"""Compare Blend Mol Cartesian optimizers on identical distorted structures."""

from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))

from molfile import Atom, Molecule, read_v2000
from uff import minimize_partial_energy


def distorted(source: Molecule, seed: int, amplitude: float = 0.12) -> Molecule:
    generator = random.Random(seed)
    return Molecule(
        f"{source.name} optimizer benchmark {seed}",
        tuple(
            Atom(
                atom.element,
                tuple(value + generator.uniform(-amplitude, amplitude) for value in atom.position),
                atom.formal_charge,
                atom.chirality,
            )
            for atom in source.atoms
        ),
        source.bonds,
    )


def benchmark(iterations: int = 100, seeds: range = range(3)) -> dict:
    source = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    algorithms = ("STEEPEST", "FIRE2", "LBFGS")
    results = {}
    for algorithm in algorithms:
        cases = []
        for seed in seeds:
            molecule = distorted(source, seed)
            started = perf_counter()
            result = minimize_partial_energy(
                molecule, iterations, 0.02, algorithm=algorithm,
                gradient_tolerance=1e-5, energy_tolerance=1e-6,
            )
            elapsed_ms = (perf_counter() - started) * 1000.0
            if not result.ready:
                raise RuntimeError(f"{algorithm} failed for seed {seed}: {result.messages}")
            cases.append({
                "seed": seed,
                "elapsed_ms": elapsed_ms,
                "iterations": result.iterations,
                "termination": result.termination,
                "initial_energy_kcal_mol": result.initial_energy_kcal_mol,
                "final_energy_kcal_mol": result.final_energy_kcal_mol,
            })
        results[algorithm] = {
            "median_elapsed_ms": statistics.median(case["elapsed_ms"] for case in cases),
            "median_final_energy_kcal_mol": statistics.median(
                case["final_energy_kcal_mol"] for case in cases
            ),
            "cases": cases,
        }
    return {
        "benchmark": "Blend Mol optimizer comparison",
        "molecule": source.name,
        "atoms": len(source.atoms),
        "iterations_requested": iterations,
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
