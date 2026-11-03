"""Compare UFF typing and every energy term against Open Babel 3.1.1.

Covers the heavier main-group elements (Ge, Sn, As, Sb, Se, Te) and the
group 15 / three-coordinate sulfur cases. Each molecule is built from a
SMILES with explicit hydrogens using Blend Mol's 3D embedding, written as a
MOL file, and evaluated by both programs at that exact geometry (a single
point, no minimization).

Requires a standalone Open Babel 3.1.1 at its default Windows location; it
uses tests/openbabel_uff_311.prm as UFF.prm. Run from the repository root:

    python benchmarks/compare_openbabel_main_group.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from molfile import read_smiles, write_mol  # noqa: E402
from uff import (  # noqa: E402
    angle_bend_energy, bond_stretch_energy, out_of_plane_energy, torsion_energy, type_molecule,
    van_der_waals_energy,
)

OPENBABEL_BIN = Path(r"C:\Program Files\OpenBabel-3.1.1")
UFF_PRM_FIXTURE = ROOT / "tests" / "openbabel_uff_311.prm"
# Open Babel's own conversion factor (forcefield.h), used for its kJ/mol output.
KCAL_TO_KJ = 4.1868
TOLERANCE_KJ = 0.5

M0 = "[H]C([H])([H])"  # explicit methyl written before the centre
M = "C([H])([H])[H]"  # explicit methyl as a branch or chain end
MOLECULES = {
    "trimethylphosphine": f"{M0}P({M}){M}",
    "trimethylarsine": f"{M0}[As]({M}){M}",
    "trimethylstibine": f"{M0}[Sb]({M}){M}",
    "dimethyl sulfoxide": f"{M0}S(=O){M}",
    "dimethyl selenide": f"{M0}[Se]{M}",
    "dimethyl telluride": f"{M0}[Te]{M}",
    "dimethyl selenoxide": f"{M0}[Se](=O){M}",
    "diethyl selenide": f"[H]C([H])([H])C([H])([H])[Se]C([H])([H])C([H])([H])[H]",
    "tetramethylgermane": f"{M0}[Ge]({M})({M}){M}",
    "tetramethylstannane": f"{M0}[Sn]({M})({M}){M}",
    "thiophene": "[H]c1c([H])c([H])sc1[H]",
    "selenophene": "[H]c1c([H])c([H])[se]c1[H]",
    "furan": "[H]c1c([H])c([H])oc1[H]",
    "pyrrole": "[H]c1c([H])c([H])n([H])c1[H]",
    "trimethylarsine oxide": f"{M0}[As](=O)({M}){M}",
    # Conjugated but not aromatic: which bonds get order 1.5 must match.
    "butadiene": "[H]C([H])=C([H])C([H])=C([H])[H]",
    "acetic acid": f"{M0}C(=O)O[H]",
    "methyl acetate": f"{M0}C(=O)O{M}",
    "N-methylacetamide": f"{M0}C(=O)N([H]){M}",
    "styrene": "[H]C([H])=C([H])c1c([H])c([H])c([H])c([H])c1[H]",
    "benzene": "[H]c1c([H])c([H])c([H])c([H])c1[H]",
    "pyridine": "[H]c1c([H])c([H])nc([H])c1[H]",
    "imine": f"{M0}C([H])=N{M}",
    "thiophene-2-carbaldehyde": "[H]C(=O)c1c([H])c([H])c([H])s1",
}

# Every molecule is also compared at a randomly perturbed copy of its
# geometry, so torsion and out-of-plane terms are tested away from the flat
# or staggered minima where both programs trivially agree.
JITTER = 0.15

TERM_PATTERNS = {
    "bond": r"TOTAL BOND STRETCHING ENERGY\s*=\s*([-+0-9.]+)",
    "angle": r"TOTAL ANGLE BENDING ENERGY\s*=\s*([-+0-9.]+)",
    "torsion": r"TOTAL TORSIONAL ENERGY\s*=\s*([-+0-9.]+)",
    "oop": r"TOTAL OUT-OF-PLANE BENDING ENERGY\s*=\s*([-+0-9.]+)",
    "vdw": r"TOTAL VAN DER WAALS ENERGY\s*=\s*([-+0-9.]+)",
}


def openbabel(mol_path: Path, data_dir: Path) -> tuple[dict[str, float], list[str]]:
    environment = os.environ.copy()
    environment["BABEL_DATADIR"] = str(data_dir)
    result = subprocess.run(
        [str(OPENBABEL_BIN / "obenergy.exe"), "-ff", "UFF", "-v", str(mol_path)],
        capture_output=True, text=True, env=environment, check=True,
    )
    terms = {}
    for name, pattern in TERM_PATTERNS.items():
        match = re.search(pattern, result.stdout)
        terms[name] = float(match.group(1)) if match else 0.0
    types = []
    section = result.stdout.split("A T O M   T Y P E S", 1)[1].split("C A L C U L A T E", 1)[0]
    for line in section.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].isdigit():
            types.append(fields[1])
    return terms, types


def blend_mol(molecule) -> tuple[dict[str, float] | None, list[str]]:
    typing = type_molecule(molecule)
    reports = {
        "bond": bond_stretch_energy(molecule), "angle": angle_bend_energy(molecule),
        "torsion": torsion_energy(molecule), "oop": out_of_plane_energy(molecule),
        "vdw": van_der_waals_energy(molecule),
    }
    if any(report.energy_kcal_mol is None for report in reports.values()):
        messages = [m for report in reports.values() for m in report.messages]
        print("   Blend Mol could not evaluate:", messages[:2])
        return None, list(typing.types)
    return {name: report.energy_kcal_mol * KCAL_TO_KJ for name, report in reports.items()}, list(typing.types)


def main() -> int:
    if not (OPENBABEL_BIN / "obenergy.exe").is_file():
        print(f"Open Babel not found at {OPENBABEL_BIN}")
        return 2
    failures = 0
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        data_dir = work / "data"
        data_dir.mkdir()
        shutil.copy(UFF_PRM_FIXTURE, data_dir / "UFF.prm")
        print(f"{'molecule':24s} " + " ".join(f"{name:>16s}" for name in TERM_PATTERNS) + "  types")
        import random
        from molfile import Molecule
        jittered = {}
        for name, smiles in MOLECULES.items():
            base = read_smiles(smiles, name)
            generator = random.Random(len(name))
            jittered[name] = base
            jittered[name + " (perturbed)"] = Molecule(base.name, tuple(
                type(atom)(atom.element, tuple(v + generator.uniform(-JITTER, JITTER) for v in atom.position),
                           atom.formal_charge, atom.chirality, atom.isotope)
                for atom in base.atoms
            ), base.bonds)
        for name, molecule in jittered.items():
            path = work / "molecule.mol"
            write_mol(molecule, path, "V2000")
            ob_terms, ob_types = openbabel(path, data_dir)
            ours, our_types = blend_mol(molecule)
            types_ok = [t if t else "None" for t in our_types] == ob_types
            if ours is None:
                failures += 1
                print(f"{name:24s} FAIL (not evaluated)")
                continue
            cells = []
            bad = not types_ok
            for term in TERM_PATTERNS:
                difference = ours[term] - ob_terms[term]
                cells.append(f"{ob_terms[term]:9.2f}{difference:+7.2f}")
                bad = bad or abs(difference) > TOLERANCE_KJ
            failures += bad
            print(f"{name:24s} " + " ".join(f"{cell:>16s}" for cell in cells)
                  + f"  {'ok' if types_ok else 'TYPES DIFFER ' + str(sorted(set(our_types) ^ set(ob_types)))}"
                  + ("  <-- FAIL" if bad else ""))
    print("Each cell: Open Babel kJ/mol, then Blend Mol minus Open Babel.")
    print("FAILURES:", failures)
    print("BLEND_MOL_OPENBABEL_MAIN_GROUP_OK" if not failures else "BLEND_MOL_OPENBABEL_MAIN_GROUP_FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
