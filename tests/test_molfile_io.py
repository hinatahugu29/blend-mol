"""Phase 2 file I/O: aromatic MOL bonds, V3000, multi-record SDF, bond perception."""

import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from molfile import (
    Atom, Bond, KekulizationError, Molecule, MolfileError, SdfRecord, perceive_bonds, read_mol,
    read_pdb, read_sdf, read_sdf_records, read_smiles, read_v2000, read_xyz, write_mol,
    write_sdf, write_sdf_records, write_v2000,
)


def _v2000(name, atoms, bonds):
    """Build V2000 text from (element, x, y, z) atoms and (a, b, order) 1-based bonds."""
    lines = [name, "test", "", f"{len(atoms):>3}{len(bonds):>3}  0  0  0  0  0  0  0  0999 V2000"]
    for element, x, y, z in atoms:
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {element:<3} 0  0  0  0  0  0  0  0  0  0  0  0")
    for a, b, order in bonds:
        lines.append(f"{a:>3}{b:>3}{order:>3}  0  0  0  0")
    lines.append("M  END")
    return "\n".join(lines) + "\n"


def _ring(element_list, radius=1.4):
    count = len(element_list)
    return [
        (element, radius * math.cos(2 * math.pi * i / count), radius * math.sin(2 * math.pi * i / count), 0.0)
        for i, element in enumerate(element_list)
    ]


def _double_count(molecule):
    counts = [0] * len(molecule.atoms)
    for bond in molecule.bonds:
        if bond.order == 2:
            counts[bond.atom_a] += 1
            counts[bond.atom_b] += 1
    return counts


class TestAromaticMolBonds(unittest.TestCase):
    def _read(self, text):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aromatic.mol"
            path.write_text(text, encoding="utf-8")
            return read_v2000(path)

    def test_benzene_with_aromatic_bonds_in_any_order_is_kekulized(self):
        # Bond order chosen so a naive first-come greedy pass would strand
        # two carbons without a double bond.
        bonds = [(1, 2, 4), (4, 5, 4), (2, 3, 4), (3, 4, 4), (5, 6, 4), (6, 1, 4)]
        molecule = self._read(_v2000("benzene", _ring(["C"] * 6), bonds))
        self.assertEqual(_double_count(molecule), [1] * 6)
        self.assertEqual(sum(1 for bond in molecule.bonds if bond.order == 2), 3)
        self.assertTrue(all(bond.order in (1, 2) for bond in molecule.bonds))

    def test_pyrrole_nitrogen_with_implicit_h_stays_single(self):
        bonds = [(1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 1, 4)]
        molecule = self._read(_v2000("pyrrole", _ring(["N", "C", "C", "C", "C"]), bonds))
        self.assertEqual(_double_count(molecule), [0, 1, 1, 1, 1])

    def test_pyridine_nitrogen_takes_a_double_bond(self):
        bonds = [(1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 6, 4), (6, 1, 4)]
        molecule = self._read(_v2000("pyridine", _ring(["N", "C", "C", "C", "C", "C"]), bonds))
        self.assertEqual(_double_count(molecule), [1] * 6)

    def test_thiophene_sulfur_never_takes_a_double_bond(self):
        bonds = [(1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 1, 4)]
        molecule = self._read(_v2000("thiophene", _ring(["S", "C", "C", "C", "C"]), bonds))
        self.assertEqual(_double_count(molecule), [0, 1, 1, 1, 1])

    def test_mol2_style_aromatic_carboxylic_acid(self):
        # MOL2-derived files can mark a carboxylic acid's two C-O bonds as
        # aromatic; one must become C=O and the other C-OH.
        atoms = [("C", 0.0, 0.0, 0.0), ("C", 1.5, 0.0, 0.0), ("O", 2.2, 1.1, 0.0), ("O", 2.2, -1.1, 0.0)]
        molecule = self._read(_v2000("acetic acid", atoms, [(1, 2, 1), (2, 3, 4), (2, 4, 4)]))
        self.assertEqual(sorted(bond.order for bond in molecule.bonds), [1, 1, 2])
        self.assertEqual(_double_count(molecule)[1], 1)

    def test_impossible_aromatic_pattern_is_reported(self):
        bonds = [(1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4), (5, 1, 4)]
        with self.assertRaises(KekulizationError) as caught:
            self._read(_v2000("cyclopentadienyl", _ring(["C"] * 5), bonds))
        self.assertIn("order 4", str(caught.exception))

    def test_order_5_and_above_is_still_rejected(self):
        with self.assertRaises(MolfileError):
            self._read(_v2000("bad", _ring(["C", "C"]), [(1, 2, 5)]))


class TestSmilesKekulization(unittest.TestCase):
    def test_five_membered_heteroaromatics(self):
        # Regression: a greedy in-order pass put a double bond on thiophene's
        # sulfur and left furan/pyrrole carbons with no double bond at all.
        cases = {
            "thiophene": "c1ccsc1", "furan": "c1ccoc1", "pyrrole": "c1cc[nH]c1",
            "thiazole": "c1cscn1", "benzothiophene": "c1ccc2sccc2c1", "azulene": "c1cc2cccccc2c1",
        }
        for name, smiles in cases.items():
            with self.subTest(molecule=name):
                molecule = read_smiles(smiles, name)
                doubles = _double_count(molecule)
                for index, atom in enumerate(molecule.atoms):
                    if atom.element == "C":
                        self.assertEqual(doubles[index], 1, f"{name}: carbon {index}")
                    elif atom.element in ("S", "O"):
                        self.assertEqual(doubles[index], 0, f"{name}: {atom.element} {index}")

    def test_benzene_result_is_unchanged(self):
        molecule = read_smiles("c1ccccc1", "benzene")
        self.assertEqual([bond.order for bond in molecule.bonds], [2, 1, 2, 1, 2, 1])


class TestV3000(unittest.TestCase):
    V3000_TEXT = "\n".join([
        "ammonium acetate",
        "  test",
        "",
        "  0  0  0     0  0            999 V3000",
        "M  V30 BEGIN CTAB",
        "M  V30 COUNTS 5 3 0 0 0",
        "M  V30 BEGIN ATOM",
        "M  V30 1 C 0.0 0.0 0.0 0",
        "M  V30 2 C 1.5 0.0 0.0 0",
        "M  V30 3 O 2.2 1.1 0.0 0",
        "M  V30 4 O 2.2 -1.1 0.0 0 -",
        "M  V30 CHG=-1",
        'M  V30 5 "N" 5.0 0.0 0.0 0 CHG=1',
        "M  V30 END ATOM",
        "M  V30 BEGIN BOND",
        "M  V30 1 1 1 2",
        "M  V30 2 2 2 3",
        "M  V30 3 1 2 4 CFG=0",
        "M  V30 END BOND",
        "M  V30 END CTAB",
        "M  END",
    ]) + "\n"

    def test_reads_atoms_bonds_charges_and_continuation_lines(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "v3000.mol"
            path.write_text(self.V3000_TEXT, encoding="utf-8")
            molecule = read_mol(path)
            with self.assertRaises(MolfileError):
                read_v2000(path)
        self.assertEqual(molecule.name, "ammonium acetate")
        self.assertEqual([atom.element for atom in molecule.atoms], ["C", "C", "O", "O", "N"])
        self.assertEqual([atom.formal_charge for atom in molecule.atoms], [0, 0, 0, -1, 1])
        self.assertEqual([(b.atom_a, b.atom_b, b.order) for b in molecule.bonds],
                         [(0, 1, 1), (1, 2, 2), (1, 3, 1)])
        self.assertEqual(molecule.atoms[3].position, (2.2, -1.1, 0.0))

    def test_counts_mismatch_is_rejected(self):
        text = self.V3000_TEXT.replace("COUNTS 5 3", "COUNTS 6 3")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.mol"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(MolfileError):
                read_mol(path)

    def test_round_trip_and_automatic_version(self):
        small = read_smiles("C[N+](C)(C)C.[O-]C(=O)C", "salt")
        atoms = tuple(Atom("C", (index * 1.5, 0.0, 0.0)) for index in range(1200))
        bonds = tuple(Bond(index, index + 1, 1) for index in range(1199))
        large = Molecule("long chain", atoms, bonds)
        with TemporaryDirectory() as directory:
            small_path = Path(directory) / "small.mol"
            large_path = Path(directory) / "large.mol"
            v3000_path = Path(directory) / "small_v3000.mol"
            self.assertEqual(write_mol(small, small_path), "V2000")
            self.assertEqual(write_mol(large, large_path), "V3000")
            self.assertEqual(write_mol(small, v3000_path, "V3000"), "V3000")
            with self.assertRaises(MolfileError):
                write_v2000(large, Path(directory) / "too_big.mol")
            for path, original in ((large_path, large), (v3000_path, small)):
                text = path.read_text(encoding="utf-8")
                self.assertTrue(all(len(line) <= 80 for line in text.splitlines()))
                read_back = read_mol(path)
                self.assertEqual(len(read_back.atoms), len(original.atoms))
                self.assertEqual(
                    [(b.atom_a, b.atom_b, b.order) for b in read_back.bonds],
                    [(b.atom_a, b.atom_b, b.order) for b in original.bonds],
                )
                self.assertEqual(
                    [a.formal_charge for a in read_back.atoms], [a.formal_charge for a in original.atoms],
                )
                for got, expected in zip(read_back.atoms, original.atoms):
                    for axis in range(3):
                        self.assertAlmostEqual(got.position[axis], expected.position[axis], places=4)


class TestMultiRecordSdf(unittest.TestCase):
    def _records(self):
        return (
            SdfRecord(read_smiles("CCO", "ethanol"), (("ID", "1"), ("NOTE", "first line\nsecond line"))),
            SdfRecord(read_smiles("c1ccccc1", "benzene"), (("ID", "2"),)),
            SdfRecord(read_smiles("O", ""), ()),
        )

    def test_round_trip_with_data_items(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "set.sdf"
            write_sdf_records(self._records(), path)
            result = read_sdf_records(path)
            first_only = read_sdf(path)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.skipped, ())
        self.assertEqual(result.remaining, 0)
        self.assertEqual(result.records[0].molecule.name, "ethanol")
        self.assertEqual(result.records[0].data, (("ID", "1"), ("NOTE", "first line\nsecond line")))
        self.assertEqual(result.records[1].data, (("ID", "2"),))
        self.assertEqual(result.records[2].molecule.name, "Blend Mol molecule")
        self.assertEqual(first_only.name, "ethanol")
        self.assertEqual(len(result.records[1].molecule.atoms), 6)

    def test_max_records_and_skip_invalid(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "set.sdf"
            write_sdf_records(self._records(), path)
            text = path.read_text(encoding="utf-8").split("$$$$\n")
            broken = text[0] + "$$$$\nnot a molfile\n\n\n  x  y\nM  END\n$$$$\n" + "$$$$\n".join(text[1:])
            path.write_text(broken, encoding="utf-8")
            limited = read_sdf_records(path, max_records=1)
            with self.assertRaises(MolfileError) as caught:
                read_sdf_records(path)
            lenient = read_sdf_records(path, skip_invalid=True)
        self.assertEqual(len(limited.records), 1)
        self.assertEqual(limited.remaining, 3)
        self.assertIn("SDF record 2", str(caught.exception))
        self.assertEqual(len(lenient.records), 3)
        self.assertEqual([number for number, _reason in lenient.skipped], [2])

    def test_single_record_write_keeps_data(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "one.sdf"
            write_sdf(read_smiles("N", "ammonia"), path, data=(("SOURCE", "test"),))
            self.assertTrue(path.read_text(encoding="utf-8").rstrip("\n").endswith("$$$$"))
            result = read_sdf_records(path)
        self.assertEqual(result.records[0].data, (("SOURCE", "test"),))


class TestBondPerception(unittest.TestCase):
    def test_water_and_methane(self):
        water = Molecule("water", (
            Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.96, 0.0, 0.0)), Atom("H", (-0.24, 0.93, 0.0)),
        ), ())
        perceived = perceive_bonds(water)
        self.assertEqual(sorted((b.atom_a, b.atom_b) for b in perceived.bonds), [(0, 1), (0, 2)])
        self.assertTrue(all(bond.order == 1 for bond in perceived.bonds))

        d = 1.09 / math.sqrt(3)
        methane = Molecule("methane", (Atom("C", (0.0, 0.0, 0.0)),) + tuple(
            Atom("H", (sx * d, sy * d, sz * d))
            for sx, sy, sz in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
        ), ())
        self.assertEqual(len(perceive_bonds(methane).bonds), 4)

    def test_hydrogen_bonds_only_to_nearest_and_far_atoms_stay_apart(self):
        molecule = Molecule("close contact", (
            Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.97, 0.0, 0.0)), Atom("O", (2.0, 0.0, 0.0)),
            Atom("C", (20.0, 0.0, 0.0)),
        ), ())
        bonds = perceive_bonds(molecule).bonds
        self.assertEqual([(b.atom_a, b.atom_b) for b in bonds], [(0, 1)])

    def test_xyz_and_pdb_readers(self):
        with TemporaryDirectory() as directory:
            xyz = Path(directory) / "water.xyz"
            xyz.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n", encoding="utf-8")
            self.assertEqual(len(read_xyz(xyz).bonds), 0)
            self.assertEqual(len(perceive_bonds(read_xyz(xyz)).bonds), 2)

            pdb = Path(directory) / "water.pdb"
            rows = [("O", 0.0, 0.0), ("H", 0.96, 0.0), ("H", -0.24, 0.93)]
            pdb.write_text("".join(
                f"HETATM{i + 1:>5} {e:<4} HOH A   1    {x:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00          {e:>2}\n"
                for i, (e, x, y) in enumerate(rows)
            ) + "END\n", encoding="utf-8")
            self.assertEqual(len(read_pdb(pdb).bonds), 0)
            self.assertEqual(len(read_pdb(pdb, perceive_if_no_conect=True).bonds), 2)
            pdb.write_text(pdb.read_text(encoding="utf-8").replace("END\n", "CONECT    1    2\nEND\n"),
                           encoding="utf-8")
            # Any CONECT record means the file's own connectivity is kept as-is.
            self.assertEqual(len(read_pdb(pdb, perceive_if_no_conect=True).bonds), 1)


if __name__ == "__main__":
    unittest.main()
