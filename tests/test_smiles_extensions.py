"""Phase 3 SMILES extensions: isotopes, Si, aromatic H counts, @TH, wildcard."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from molfile import (
    MolfileError, SdfRecord, _target_valence, read_mol, read_sdf_records, read_smiles,
    write_mol, write_sdf_records,
)
from uff import _implicit_hydrogen_count, type_molecule


def _hydrogens_on(molecule, index):
    return sum(
        1 for bond in molecule.bonds
        if index in (bond.atom_a, bond.atom_b)
        and molecule.atoms[bond.atom_b if bond.atom_a == index else bond.atom_a].element == "H"
    )


class TestIsotopes(unittest.TestCase):
    def test_smiles_isotopes(self):
        methane = read_smiles("[13CH4]")
        self.assertEqual(methane.atoms[0].isotope, 13)
        heavy_water = read_smiles("[2H]O[2H]")
        self.assertEqual([atom.isotope for atom in heavy_water.atoms], [2, None, 2])
        self.assertIsNone(read_smiles("CO").atoms[0].isotope)
        labelled_benzene = read_smiles("[13cH]1ccccc1")
        self.assertEqual(labelled_benzene.atoms[0].isotope, 13)
        self.assertEqual(sum(1 for bond in labelled_benzene.bonds if bond.order == 2), 3)
        with self.assertRaises(MolfileError):
            read_smiles("[0C]")

    def test_isotopes_survive_mol_v2000_v3000_and_sdf(self):
        molecule = read_smiles("[13CH3][18OH]", "labelled methanol")
        with TemporaryDirectory() as directory:
            for version in ("V2000", "V3000"):
                path = Path(directory) / f"labelled_{version}.mol"
                write_mol(molecule, path, version)
                text = path.read_text(encoding="utf-8")
                self.assertIn("M  ISO" if version == "V2000" else "MASS=13", text)
                read_back = read_mol(path)
                self.assertEqual([atom.isotope for atom in read_back.atoms], [13, 18], version)
            sdf = Path(directory) / "labelled.sdf"
            write_sdf_records([SdfRecord(molecule)], sdf, "V2000")
            self.assertEqual([a.isotope for a in read_sdf_records(sdf).records[0].molecule.atoms], [13, 18])

    def test_isotope_does_not_change_uff_typing(self):
        plain = type_molecule(read_smiles("CO"))
        labelled = type_molecule(read_smiles("[13CH3]O"))
        self.assertEqual(plain.types, labelled.types)


class TestSilicon(unittest.TestCase):
    def test_bracket_silicon(self):
        silane = read_smiles("[SiH4]")
        self.assertEqual(silane.atoms[0].element, "Si")
        self.assertEqual(_implicit_hydrogen_count(silane), 4)
        tetramethylsilane = read_smiles("C[Si](C)(C)C")
        self.assertEqual(type_molecule(tetramethylsilane).types[1], "Si3")

    def test_silicon_needs_brackets(self):
        # Si is outside the SMILES organic subset; "CSi" is C, S and a stray "i".
        with self.assertRaises(MolfileError):
            read_smiles("C[Si")
        with self.assertRaises(MolfileError):
            read_smiles("CSi")


class TestAromaticHydrogens(unittest.TestCase):
    def test_bracket_aromatic_atoms_keep_their_hydrogen_count(self):
        pyrrole = read_smiles("c1cc[nH]c1")
        nitrogen = next(i for i, atom in enumerate(pyrrole.atoms) if atom.element == "N")
        self.assertEqual(_hydrogens_on(pyrrole, nitrogen), 1)
        two = read_smiles("C[nH2+]")
        nitrogen = next(i for i, atom in enumerate(two.atoms) if atom.element == "N")
        self.assertEqual(_hydrogens_on(two, nitrogen), 2)
        self.assertEqual(two.atoms[nitrogen].formal_charge, 1)


class TestStereoClasses(unittest.TestCase):
    def test_th1_th2_are_aliases(self):
        self.assertEqual(read_smiles("N[C@TH1H](C)C(=O)O").atoms[1].chirality, "@")
        self.assertEqual(read_smiles("N[C@TH2H](C)C(=O)O").atoms[1].chirality, "@@")
        same = read_smiles("N[C@H](C)C(=O)O")
        alias = read_smiles("N[C@TH1H](C)C(=O)O")
        self.assertEqual(
            [atom.position for atom in same.atoms], [atom.position for atom in alias.atoms],
        )

    def test_other_stereo_classes_are_named_in_the_error(self):
        with self.assertRaises(MolfileError) as caught:
            read_smiles("C=[C@AL1]=CC")
        self.assertIn("@AL1", str(caught.exception))


class TestWildcard(unittest.TestCase):
    def test_wildcard_atom(self):
        for smiles in ("*CC", "[*]CC"):
            with self.subTest(smiles=smiles):
                molecule = read_smiles(smiles)
                self.assertEqual(molecule.atoms[0].element, "*")
                self.assertEqual(_target_valence(molecule.atoms[0]), 0)
                report = type_molecule(molecule)
                self.assertFalse(report.ready)
                self.assertTrue(any("wildcard" in message for message in report.messages))

    def test_wildcard_round_trips_through_mol(self):
        molecule = read_smiles("*c1ccccc1", "phenyl placeholder")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wildcard.mol"
            write_mol(molecule, path)
            read_back = read_mol(path)
        self.assertEqual(read_back.atoms[0].element, "*")
        self.assertEqual(sum(1 for bond in read_back.bonds if bond.order == 2), 3)


if __name__ == "__main__":
    unittest.main()
