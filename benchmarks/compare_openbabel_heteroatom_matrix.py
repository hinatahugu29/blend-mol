"""Compare Blend Mol energies against standalone Open Babel for P/S/Si/B.

Unlike benchmarks/compare_avogadro_uff.py (which needs Avogadro 2 and its
bundled Open Babel), this targets a standalone Open Babel 3.1.1 install,
using tests/openbabel_uff_311.prm (the same fixture
tests/test_uff_parameters.py checks the parameter table against) as its
UFF.prm data file, since a standalone Open Babel install does not ship one
on Windows.

Requires Open Babel 3.1.1 installed at its default Windows location.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))

from molfile import Atom, Bond, Molecule
from uff import angle_bend_energy, bond_stretch_energy, partial_energy, torsion_energy, van_der_waals_energy


OPENBABEL_BIN = Path(r"C:\Program Files\OpenBabel-3.1.1")
UFF_PRM_FIXTURE = ROOT / "tests" / "openbabel_uff_311.prm"
KCAL_TO_KJ = 4.184

# Neutral structures whose heteroatom has no dedicated Open Babel SMARTS
# rule for this exact coordination (a tetravalent P=O / S=O center without
# an explicit +4/+5/+6 formal charge), so both tools fall back to the same
# generic type (P_3+3 / S_3+2) -- see the comments in uff.py's
# type_molecule for how this was confirmed against this same Open Babel
# install.
def _methyl(center_position, carbon_index, hydrogen_start_index):
    """A CH3 group's carbon position and its three explicit hydrogens."""
    cx, cy, cz = center_position
    atoms = [Atom("C", center_position)]
    offsets = ((0.9, 0.4, 0.0), (-0.3, 0.9, 0.3), (-0.3, -0.5, -0.8))
    bonds = []
    for index, (dx, dy, dz) in enumerate(offsets):
        atoms.append(Atom("H", (cx + dx, cy + dy, cz + dz)))
        bonds.append(Bond(carbon_index, hydrogen_start_index + index, 1))
    return atoms, bonds


def _trimethyl_heteroatom(element: str, bond_order: int) -> Molecule:
    """R3X=O with three full CH3 groups, for a complete, valid valence."""
    atoms = [Atom(element, (0.0, 0.0, 0.0)), Atom("O", (1.4, 0.3, 0.0))]
    bonds = [Bond(0, 1, bond_order)]
    next_index = 2
    for position in ((-0.6, 1.3, 0.0), (-0.6, -1.3, 0.0), (-1.2, 0.0, 1.3)):
        methyl_atoms, methyl_bonds = _methyl(position, next_index, next_index + 1)
        atoms.extend(methyl_atoms)
        bonds.append(Bond(0, next_index, 1))
        bonds.extend(methyl_bonds)
        next_index += 4
    return Molecule(f"{element}3X=O", tuple(atoms), tuple(bonds))


def _dimethyl_heteroatom(element: str, extra_oxygens: int) -> Molecule:
    """R2X(=O)n with two full CH3 groups and n double-bonded oxygens."""
    atoms = [Atom(element, (0.0, 0.0, 0.0))]
    bonds = []
    oxygen_positions = ((1.4, 0.3, 0.0), (-1.4, 0.3, 0.0))
    for index in range(extra_oxygens):
        atoms.append(Atom("O", oxygen_positions[index]))
        bonds.append(Bond(0, len(atoms) - 1, 2))
    next_index = len(atoms)
    for position in ((0.0, 1.3, 1.0), (0.0, -1.3, 1.0)):
        methyl_atoms, methyl_bonds = _methyl(position, next_index, next_index + 1)
        atoms.extend(methyl_atoms)
        bonds.append(Bond(0, next_index, 1))
        bonds.extend(methyl_bonds)
        next_index += 4
    return Molecule(f"{element}(=O){extra_oxygens}", tuple(atoms), tuple(bonds))


MOLECULES = {
    "trimethylphosphine_oxide": _trimethyl_heteroatom("P", 2),
    "dimethyl_sulfoxide": _dimethyl_heteroatom("S", 1),
    "dimethyl_sulfone": _dimethyl_heteroatom("S", 2),
    "silane": Molecule(
        "SiH4",
        (Atom("Si", (0.0, 0.0, 0.0)),) + tuple(
            Atom("H", (1.1, float(index), 0.3 * index)) for index in range(4)
        ),
        tuple(Bond(0, index + 1, 1) for index in range(4)),
    ),
    "borane": Molecule(
        "BH3",
        (
            Atom("B", (0.0, 0.0, 0.0)),
            Atom("H", (1.2, 0.0, 0.0)),
            Atom("H", (-0.6, 1.0, 0.0)),
            Atom("H", (-0.6, -1.0, 0.0)),
        ),
        (Bond(0, 1, 1), Bond(0, 2, 1), Bond(0, 3, 1)),
    ),
}


def _write_v2000(molecule: Molecule, path: Path) -> None:
    from molfile import write_v2000
    write_v2000(molecule, path)


def openbabel_energy(mol_path: Path, data_dir: Path) -> dict[str, float]:
    executable = OPENBABEL_BIN / "obenergy.exe"
    environment = os.environ.copy()
    environment["BABEL_DATADIR"] = str(data_dir)
    completed = subprocess.run(
        [str(executable), "-ff", "UFF", "-v", str(mol_path)],
        check=True, capture_output=True, text=True, env=environment,
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
            raise RuntimeError(f"Cannot parse Open Babel {name} energy:\n{completed.stdout}")
        values[name] = float(match.group(1))
    return values


def blend_mol_energy(molecule: Molecule) -> dict[str, float]:
    return {
        "bond": bond_stretch_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
        "angle": angle_bend_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
        "torsion": torsion_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
        "vdw": van_der_waals_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
        "total": partial_energy(molecule).energy_kcal_mol * KCAL_TO_KJ,
    }


def compare() -> dict:
    if not (OPENBABEL_BIN / "obenergy.exe").is_file():
        raise RuntimeError(f"Open Babel not found at {OPENBABEL_BIN}")
    results = {}
    with tempfile.TemporaryDirectory(prefix="blend_mol_openbabel_") as directory:
        data_dir = Path(directory)
        shutil.copy(UFF_PRM_FIXTURE, data_dir / "UFF.prm")
        for name, molecule in MOLECULES.items():
            mol_path = data_dir / f"{name}.mol"
            _write_v2000(molecule, mol_path)
            blend_mol = blend_mol_energy(molecule)
            openbabel = openbabel_energy(mol_path, data_dir)
            differences = {key: blend_mol[key] - openbabel[key] for key in blend_mol}
            results[name] = {
                "blend_mol": blend_mol,
                "openbabel": openbabel,
                "blend_mol_minus_openbabel": differences,
                "max_absolute_difference_kj_mol": max(abs(value) for value in differences.values()),
            }
    return {
        "comparison": "Blend Mol vs standalone Open Babel 3.1.1 UFF (P/S/Si/B typing)",
        "units": "kJ/mol",
        "molecules": results,
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
