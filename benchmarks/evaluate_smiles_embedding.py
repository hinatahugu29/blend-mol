"""Evaluate SMILES starting geometries: clashes, bond lengths, stereo, time.

Run from the repository root with any Python 3.11+:

    python benchmarks/evaluate_smiles_embedding.py

For every molecule it reports the closest non-bonded pair (atoms more than
two bonds apart), the worst bond-length error against covalent radii, the
generation time, and -- when a standalone Open Babel 3.1.1 is installed --
whether Open Babel perceives the same tetrahedral and E/Z stereochemistry
from the generated 3D coordinates as from the input SMILES.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from elements import COVALENT_RADII  # noqa: E402
from molfile import read_smiles, write_mol  # noqa: E402

OPENBABEL = Path(r"C:\Program Files\OpenBabel-3.1.1\obabel.exe")

MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "beta-glucose": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    # The anomeric carbon is unspecified, so either configuration is correct.
    "reported pyranose": ("OC[C@@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O",
                          "OC[C@@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O",
                          "OC[C@@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@H]1O"),
    "chain polyol": "OC[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)CO",
    "hexatriene": "C/C=C/C=C\\C=C/C",
    "trans-difluoroethene": "F/C=C/F",
    "cis-difluoroethene": "F/C=C\\F",
    "L-alanine": "N[C@@H](C)C(=O)O",
    "leading-H stereocenter": "[C@@H](N)(C)C(=O)O",
    "ring-closure stereocenter": "C[C@H]1CCCCO1",
    "menthol": "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O",
    "cholesterol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
    "adamantane": "C1C2CC3CC1CC(C2)C3",
    "naphthalene": "c1ccc2ccccc2c1",
    "cyclododecane": "C1CCCCCCCCCCC1",
    "C100 branched alkane": "C" + "C(C)" * 33,
}


def _distance(a, b) -> float:
    return math.dist(a.position, b.position)


def geometry_metrics(molecule) -> tuple[float, tuple[int, int], float]:
    count = len(molecule.atoms)
    neighbours = [set() for _ in range(count)]
    for bond in molecule.bonds:
        neighbours[bond.atom_a].add(bond.atom_b)
        neighbours[bond.atom_b].add(bond.atom_a)
    within_two = [set(n) for n in neighbours]
    for index in range(count):
        for neighbour in neighbours[index]:
            within_two[index] |= neighbours[neighbour]
    closest, closest_pair = math.inf, (-1, -1)
    for first in range(count):
        for second in range(first + 1, count):
            if second in within_two[first]:
                continue
            distance = _distance(molecule.atoms[first], molecule.atoms[second])
            if distance < closest:
                closest, closest_pair = distance, (first, second)
    worst_bond = 0.0
    for bond in molecule.bonds:
        first, second = molecule.atoms[bond.atom_a], molecule.atoms[bond.atom_b]
        ideal = (COVALENT_RADII.get(first.element, 0.8) + COVALENT_RADII.get(second.element, 0.8)) * (
            1.0 - 0.12 * (bond.order - 1)
        )
        worst_bond = max(worst_bond, abs(_distance(first, second) - ideal))
    return closest, closest_pair, worst_bond


def _canonical(arguments: list[str]) -> str:
    result = subprocess.run(
        [str(OPENBABEL), *arguments, "-ocan"], capture_output=True, text=True, check=False,
    )
    return result.stdout.split()[0] if result.stdout.split() else "?"


def stereo_matches(smiles: str, accepted: tuple[str, ...], molecule, directory: Path) -> str:
    """'ok', 'MISMATCH ...', or 'n/a' (Open Babel missing, or no stereo in the SMILES)."""
    if not OPENBABEL.is_file():
        return "n/a"
    if not any(mark in smiles for mark in "@/\\"):
        return "n/a (no stereo specified)"
    path = directory / "embedded.mol"
    write_mol(molecule, path, "V2000")
    expected = {_canonical([f"-:{option}"]) for option in accepted}
    got = _canonical([str(path)])
    return "ok" if got in expected else f"MISMATCH got {got} expected {sorted(expected)}"


def main() -> int:
    failures = 0
    print(f"{'molecule':26s} {'atoms':>5s} {'closest':>8s} {'bond err':>8s} {'ms':>7s}  stereo")
    with tempfile.TemporaryDirectory() as directory:
        for name, entry in MOLECULES.items():
            smiles, accepted = (entry[0], entry[1:]) if isinstance(entry, tuple) else (entry, (entry,))
            start = time.perf_counter()
            molecule = read_smiles(smiles, name)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            closest, pair, worst_bond = geometry_metrics(molecule)
            stereo = stereo_matches(smiles, accepted, molecule, Path(directory))
            flag = ""
            if closest < 1.0 or worst_bond > 0.15 or stereo.startswith("MISMATCH"):
                failures += 1
                flag = "  <-- FAIL"
            print(
                f"{name:26s} {len(molecule.atoms):5d} {closest:8.3f} {worst_bond:8.3f} "
                f"{elapsed_ms:7.1f}  {stereo}{flag}"
            )
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
