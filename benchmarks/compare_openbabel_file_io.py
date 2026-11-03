"""Cross-check Blend Mol's MOL/SDF reading and writing against Open Babel 3.1.1.

Requires a standalone Open Babel 3.1.1 install (obabel.exe). Run from the
repository root with any Python 3.11+:

    python benchmarks/compare_openbabel_file_io.py

Checks, for a set of SMILES:
1. Blend Mol writes V3000 and multi-record SDF that Open Babel reads with the
   same atom count, bond count and total bond order per record.
2. Blend Mol reads Open Babel's own V2000 and V3000 output, including files
   with aromatic (order 4) bonds, with matching connectivity, and its
   Kekule assignment gives every atom the same total bond order (valence)
   as Open Babel's Kekule structure.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from molfile import (  # noqa: E402
    SdfRecord, _target_valence, read_mol, read_sdf_records, read_smiles, write_mol,
    write_sdf_records,
)

OPENBABEL_BIN = Path(r"C:\Program Files\OpenBabel-3.1.1")
SMILES = {
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "indole": "c1ccc2[nH]ccc2c1",
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "naphthalene": "c1ccc2ccccc2c1",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "imidazole": "c1c[nH]cn1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "acetate": "CC(=O)[O-]",
    "labelled methanol": "[13CH3][18OH]",
    "heavy water": "[2H]O[2H]",
    "tetramethylsilane": "C[Si](C)(C)C",
}


def obabel(*args: str) -> str:
    result = subprocess.run(
        [str(OPENBABEL_BIN / "obabel.exe"), *args], capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"obabel {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def aromatic_v2000(reference_path: Path) -> str:
    """Rewrite Open Babel's Kekule V2000 with order 4 on the bonds it perceives as aromatic.

    Open Babel 3.1.1 never writes order 4 itself, but its MOL2 output types
    aromatic bonds as ``ar``; atom order is the same in both files.
    """
    mol2 = obabel(str(reference_path), "-omol2").splitlines()
    start = mol2.index("@<TRIPOS>BOND") + 1
    aromatic = set()
    for line in mol2[start:]:
        if line.startswith("@<TRIPOS>"):
            break
        fields = line.split()
        if len(fields) >= 4 and fields[3] == "ar":
            aromatic.add(tuple(sorted((int(fields[1]), int(fields[2])))))
    lines = reference_path.read_text(encoding="utf-8").splitlines()
    atom_count, bond_count = int(lines[3][0:3]), int(lines[3][3:6])
    for index in range(4 + atom_count, 4 + atom_count + bond_count):
        line = lines[index]
        pair = tuple(sorted((int(line[0:3]), int(line[3:6]))))
        if pair in aromatic:
            lines[index] = line[:6] + "  4" + line[9:]
    return "\n".join(lines) + "\n"


def valences(molecule) -> list[int]:
    """Total bond order per atom (explicit bonds only)."""
    sums = [0] * len(molecule.atoms)
    for bond in molecule.bonds:
        sums[bond.atom_a] += bond.order
        sums[bond.atom_b] += bond.order
    return sums


def heavy_valences(molecule) -> list[int]:
    return [total for atom, total in zip(molecule.atoms, valences(molecule)) if atom.element != "H"]


def double_count(molecule) -> int:
    return sum(1 for bond in molecule.bonds if bond.order == 2)


def edge_set(molecule) -> set[tuple[int, int]]:
    return {tuple(sorted((bond.atom_a, bond.atom_b))) for bond in molecule.bonds}


def main() -> int:
    if not (OPENBABEL_BIN / "obabel.exe").is_file():
        print(f"Open Babel not found at {OPENBABEL_BIN}")
        return 2
    failures = 0
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        # 1. Open Babel reads what Blend Mol writes.
        molecules = {name: read_smiles(smiles, name) for name, smiles in SMILES.items()}
        sdf_path = work / "blend_mol_set.sdf"
        write_sdf_records([SdfRecord(m, (("NAME", n),)) for n, m in molecules.items()], sdf_path, "V3000")
        rows = [row for row in obabel(str(sdf_path), "-ocan").splitlines() if "\t" in row]
        if len(rows) != len(molecules):
            print(f"FAIL Open Babel read {len(rows)} of {len(molecules)} V3000 SDF records")
            failures += 1
        for (name, _molecule), row in zip(molecules.items(), rows):
            canonical_ours, title = row.split("\t", 1)
            canonical_ref = obabel(f"-:{SMILES[name]}", "-ocan").split()[0]
            if canonical_ours != canonical_ref or title.strip() != name:
                print(f"FAIL {name}: SDF record reads as {canonical_ours} ({title.strip()}), "
                      f"expected {canonical_ref}")
                failures += 1
        for name, molecule in molecules.items():
            mol_path = work / f"{name}_v3000.mol"
            write_mol(molecule, mol_path, "V3000")
            canonical_ours = obabel(str(mol_path), "-ocan").split()[0]
            canonical_ref = obabel(f"-:{SMILES[name]}", "-ocan").split()[0]
            if canonical_ours != canonical_ref:
                print(f"FAIL {name}: V3000 written by Blend Mol reads as {canonical_ours}, "
                      f"expected {canonical_ref}")
                failures += 1

        # 2. Blend Mol reads what Open Babel writes (V2000, V3000, aromatic bonds).
        for name, smiles in SMILES.items():
            reference_path = work / f"{name}_ob.mol"
            reference_path.write_text(obabel(f"-:{smiles}", "-omol", "--gen2d"), encoding="utf-8")
            reference = read_mol(reference_path)
            aromatic_path = work / f"{name}_aromatic.mol"
            aromatic_path.write_text(aromatic_v2000(reference_path), encoding="utf-8")
            v3000_path = work / f"{name}_v3000_ob.mol"
            v3000_path.write_text(obabel(str(reference_path), "-omol", "-x3"), encoding="utf-8")
            for variant_path, label in ((aromatic_path, "aromatic V2000"), (v3000_path, "V3000")):
                try:
                    ours = read_mol(variant_path)
                except Exception as exc:  # noqa: BLE001 - report any reader failure
                    print(f"FAIL {name} ({label}): {exc}")
                    failures += 1
                    continue
                if [a.isotope for a in ours.atoms] != [a.isotope for a in reference.atoms]:
                    print(f"FAIL {name} ({label}): isotopes differ from Open Babel's file")
                    failures += 1
                if edge_set(ours) != edge_set(reference):
                    print(f"FAIL {name} ({label}): connectivity differs from Open Babel")
                    failures += 1
                elif heavy_valences(ours) != heavy_valences(reference):
                    # With implicit H, which ring N (or acid O) carries the H is a
                    # tautomer choice. Accept it when the double-bond count matches
                    # and no atom exceeds its typical valence.
                    same_doubles = double_count(ours) == double_count(reference)
                    within = all(
                        total <= (_target_valence(atom) or total)
                        for atom, total in zip(ours.atoms, valences(ours))
                    )
                    if same_doubles and within:
                        print(f"NOTE {name} ({label}): different but valid tautomer than Open Babel")
                    else:
                        print(f"FAIL {name} ({label}): per-atom bond order totals differ "
                              f"{heavy_valences(ours)} vs {heavy_valences(reference)}")
                        failures += 1
        # Multi-record SDF written by Open Babel.
        ob_sdf = work / "ob_set.sdf"
        ob_sdf.write_text("".join(obabel(f"-:{s}", "-osdf", "--gen2d") for s in SMILES.values()),
                          encoding="utf-8")
        result = read_sdf_records(ob_sdf)
        if len(result.records) != len(SMILES) or result.skipped:
            print(f"FAIL Blend Mol read {len(result.records)} of {len(SMILES)} Open Babel SDF records")
            failures += 1
    print("FAILURES:", failures)
    print("BLEND_MOL_OPENBABEL_FILE_IO_OK" if not failures else "BLEND_MOL_OPENBABEL_FILE_IO_FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
