"""Reproducible Blend Mol UFF scaling benchmark.

Uses separated, fully explicit methane fragments so atom typing and every
Cartesian gradient coordinate are exercised without introducing degenerate
angles. The benchmark measures computation only; Blender display timing is
measured separately by the add-on.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))

from molfile import Atom, Bond, Molecule, read_v2000
from uff import minimize_partial_energy, partial_energy


def methane_ensemble(atom_count: int) -> Molecule:
    if atom_count % 5:
        raise ValueError("Synthetic benchmark atom count must be divisible by five")
    atoms: list[Atom] = []
    bonds: list[Bond] = []
    tetrahedron = (
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
    bond_length = 1.09 / math.sqrt(3.0)
    for fragment in range(atom_count // 5):
        grid_x = fragment % 5
        grid_y = (fragment // 5) % 5
        grid_z = fragment // 25
        center = (grid_x * 12.0, grid_y * 12.0, grid_z * 12.0)
        carbon_index = len(atoms)
        atoms.append(Atom("C", center))
        for direction in tetrahedron:
            hydrogen_index = len(atoms)
            atoms.append(Atom("H", tuple(center[axis] + direction[axis] * bond_length for axis in range(3))))
            bonds.append(Bond(carbon_index, hydrogen_index, 1))
    return Molecule(f"{atom_count}-atom methane ensemble", tuple(atoms), tuple(bonds))


def replicated_molecule(source: Molecule, copies: int) -> Molecule:
    atoms: list[Atom] = []
    bonds: list[Bond] = []
    for copy in range(copies):
        offset = len(atoms)
        translation = (copy % 4 * 20.0, (copy // 4) % 4 * 20.0, copy // 16 * 20.0)
        atoms.extend(
            Atom(
                atom.element,
                tuple(atom.position[axis] + translation[axis] for axis in range(3)),
                atom.formal_charge,
                atom.chirality,
            )
            for atom in source.atoms
        )
        bonds.extend(
            Bond(bond.atom_a + offset, bond.atom_b + offset, bond.order, bond.stereo)
            for bond in source.bonds
        )
    return Molecule(f"{copies} x {source.name}", tuple(atoms), tuple(bonds))


def measure(label: str, molecules: list[Molecule], repeats: int) -> dict:
    results = []
    for molecule in molecules:
        atom_count = len(molecule.atoms)
        energy = partial_energy(molecule)
        if not energy.ready:
            raise RuntimeError("; ".join(energy.messages))
        samples = []
        for _repeat in range(repeats):
            started = perf_counter()
            result = minimize_partial_energy(molecule, 1, 0.02)
            samples.append((perf_counter() - started) * 1000.0)
            if not result.ready:
                raise RuntimeError("; ".join(result.messages))
        ordered = sorted(samples)
        median_ms = ordered[len(ordered) // 2]
        results.append({
            "atom_count": atom_count,
            "samples_ms": samples,
            "median_ms": median_ms,
            "estimated_hz": 1000.0 / median_ms,
        })
    return {"case": label, "results": results}


def benchmark(repeats=3) -> dict:
    triphenylene = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    return {
        "benchmark": "Blend Mol partial UFF scaling",
        "repeats": repeats,
        "cases": [
            measure(
                "separated explicit methane fragments (light residual terms)",
                [methane_ensemble(count) for count in (30, 100, 300)],
                repeats,
            ),
            measure(
                "replicated explicit triphenylene (torsion/inversion heavy)",
                [replicated_molecule(triphenylene, copies) for copies in (1, 4, 10)],
                repeats,
            ),
        ],
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
