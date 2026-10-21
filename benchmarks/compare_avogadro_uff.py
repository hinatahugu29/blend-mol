"""Compare Blend Mol energy components with Avogadro 2's bundled Open Babel UFF."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))

from molfile import read_v2000
from native_backend import set_backend_preference
from uff import (
    angle_bend_energy,
    bond_stretch_energy,
    partial_energy,
    torsion_energy,
    van_der_waals_energy,
    minimize_partial_energy,
)


AVOGADRO_BIN = Path(r"C:\Program Files\Avogadro2\bin")
INPUT = ROOT / "217-59-4(トリフェニレン)2.mol"
KCAL_TO_KJ = 4.184


def openbabel_energy() -> dict[str, float]:
    executable = AVOGADRO_BIN / "obenergy.exe"
    environment = os.environ.copy()
    environment["BABEL_DATADIR"] = str(AVOGADRO_BIN / "data")
    completed = subprocess.run(
        [str(executable), "-ff", "UFF", str(INPUT)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    patterns = {
        "bond": r"BOND STRETCHING ENERGY\s*=\s*([-+0-9.]+)",
        "angle": r"ANGLE BENDING ENERGY\s*=\s*([-+0-9.]+)",
        "torsion": r"TORSIONAL ENERGY\s*=\s*([-+0-9.]+)",
        "vdw": r"VAN DER WAALS ENERGY\s*=\s*([-+0-9.]+)",
        "total": r"TOTAL ENERGY\s*=\s*([-+0-9.]+)",
    }
    values = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, completed.stdout)
        if match is None:
            raise RuntimeError(f"Cannot parse Open Babel {name} energy")
        values[name] = float(match.group(1))
    return values


def blend_mol_energy() -> dict[str, float]:
    molecule = read_v2000(INPUT)
    return {
        "bond": bond_stretch_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
        "angle": angle_bend_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
        "torsion": torsion_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
        "vdw": van_der_waals_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
        "total": partial_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
    }


def compare() -> dict:
    blend_mol = blend_mol_energy()
    openbabel = openbabel_energy()
    differences = {
        name: blend_mol[name] - openbabel[name]
        for name in blend_mol
    }
    relative_percent = {
        name: abs(differences[name]) / max(abs(openbabel[name]), 1.0e-12) * 100.0
        for name in differences
    }
    with tempfile.TemporaryDirectory(prefix="blend_mol_avogadro_") as directory:
        optimized_path = Path(directory) / "openbabel_uff_500.mol"
        environment = os.environ.copy()
        environment["BABEL_DATADIR"] = str(AVOGADRO_BIN / "data")
        subprocess.run(
            [
                str(AVOGADRO_BIN / "obabel.exe"), str(INPUT), "-O", str(optimized_path),
                "--minimize", "--ff", "UFF", "--steps", "500", "--crit", "1e-6",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        source = read_v2000(INPUT)
        openbabel_optimized = read_v2000(optimized_path)
        set_backend_preference("rust")
        blend_mol_optimized = minimize_partial_energy(source, 500, 0.02)
        set_backend_preference("auto")
        if not blend_mol_optimized.ready:
            raise RuntimeError("Blend Mol optimization failed")
        all_pairs = [
            (first, second)
            for first in range(len(source.atoms))
            for second in range(first + 1, len(source.atoms))
        ]
        def distance_rms(pairs):
            return math.sqrt(sum(
                (
                    math.dist(
                        blend_mol_optimized.molecule.atoms[first].position,
                        blend_mol_optimized.molecule.atoms[second].position,
                    )
                    - math.dist(
                        openbabel_optimized.atoms[first].position,
                        openbabel_optimized.atoms[second].position,
                    )
                ) ** 2
                for first, second in pairs
            ) / len(pairs))
        geometry = {
            "openbabel_steps_requested": 500,
            "blend_mol_steps_accepted": blend_mol_optimized.iterations,
            "blend_mol_termination": blend_mol_optimized.termination,
            "all_pair_distance_rms_angstrom": distance_rms(all_pairs),
            "bond_length_rms_angstrom": distance_rms([
                (bond.atom_a, bond.atom_b) for bond in source.bonds
            ]),
        }
    return {
        "comparison": "Blend Mol vs Avogadro 2.0.0 bundled Open Babel 3 UFF",
        "units": "kJ/mol",
        "input": str(INPUT),
        "blend_mol": blend_mol,
        "openbabel": openbabel,
        "blend_mol_minus_openbabel": differences,
        "absolute_relative_difference_percent": relative_percent,
        "optimized_geometry": geometry,
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
