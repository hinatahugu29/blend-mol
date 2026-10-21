"""Stress numerical parity and repeatability of Python and Rust UFF backends."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))

from molfile import Atom, Molecule, read_v2000
from native_backend import set_backend_preference
from uff import minimize_partial_energy


def distorted(source: Molecule, seed: int, amplitude: float = 0.08) -> Molecule:
    generator = random.Random(seed)
    return Molecule(
        f"{source.name} distorted {seed}",
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


def max_coordinate_difference(first: Molecule, second: Molecule) -> float:
    return max(
        abs(first_atom.position[axis] - second_atom.position[axis])
        for first_atom, second_atom in zip(first.atoms, second.atoms)
        for axis in range(3)
    )


def compare(
    source: Molecule, seeds: range, iterations: int, fixed_atoms: set[int],
    algorithm: str = "STEEPEST",
) -> dict:
    cases = []
    for seed in seeds:
        molecule = distorted(source, seed)
        set_backend_preference("python")
        python = minimize_partial_energy(
            molecule, iterations, 0.02, fixed_atoms, algorithm=algorithm
        )
        set_backend_preference("rust")
        rust = minimize_partial_energy(
            molecule, iterations, 0.02, fixed_atoms, algorithm=algorithm
        )
        if not python.ready or not rust.ready:
            raise RuntimeError(f"backend failure at seed {seed}: {python.messages} / {rust.messages}")
        cases.append({
            "seed": seed,
            "iterations_requested": iterations,
            "python_iterations": python.iterations,
            "rust_iterations": rust.iterations,
            "energy_difference": abs(
                python.final_energy_kcal_mol - rust.final_energy_kcal_mol
            ),
            "maximum_coordinate_difference": max_coordinate_difference(
                python.molecule, rust.molecule
            ),
        })
    return {
        "fixed_atoms": sorted(fixed_atoms),
        "algorithm": algorithm,
        "cases": cases,
        "maximum_energy_difference": max(case["energy_difference"] for case in cases),
        "maximum_coordinate_difference": max(
            case["maximum_coordinate_difference"] for case in cases
        ),
    }


def repeatability(source: Molecule, repeats: int = 50) -> dict:
    molecule = distorted(source, 8675309)
    set_backend_preference("rust")
    results = [minimize_partial_energy(molecule, 3, 0.02) for _ in range(repeats)]
    if not all(result.ready for result in results):
        raise RuntimeError("Rust repeatability run failed")
    reference = results[0]
    return {
        "repeats": repeats,
        "maximum_energy_difference": max(
            abs(result.final_energy_kcal_mol - reference.final_energy_kcal_mol)
            for result in results
        ),
        "maximum_coordinate_difference": max(
            max_coordinate_difference(result.molecule, reference.molecule)
            for result in results
        ),
    }


def validate() -> dict:
    source = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    single_step = compare(source, range(10), 1, set())
    multi_step = compare(source, range(5), 20, {0, 7})
    fire2 = compare(source, range(3), 10, {0, 7}, "FIRE2")
    repeated = repeatability(source)
    set_backend_preference("auto")
    tolerances = {"energy": 1.0e-7, "coordinate": 1.0e-8}
    passed = (
        single_step["maximum_energy_difference"] <= tolerances["energy"]
        and single_step["maximum_coordinate_difference"] <= tolerances["coordinate"]
        and multi_step["maximum_energy_difference"] <= tolerances["energy"]
        and multi_step["maximum_coordinate_difference"] <= tolerances["coordinate"]
        and fire2["maximum_energy_difference"] <= tolerances["energy"]
        and fire2["maximum_coordinate_difference"] <= tolerances["coordinate"]
        and repeated["maximum_energy_difference"] == 0.0
        and repeated["maximum_coordinate_difference"] == 0.0
    )
    return {
        "validation": "Blend Mol Python/Rust parity and Rust repeatability",
        "passed": passed,
        "tolerances": tolerances,
        "single_step_randomized": single_step,
        "twenty_step_randomized_with_fixed_atoms": multi_step,
        "fire2_ten_step_randomized_with_fixed_atoms": fire2,
        "rust_repeatability": repeated,
    }


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
