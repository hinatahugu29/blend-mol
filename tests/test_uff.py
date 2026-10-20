import unittest
import math
from pathlib import Path

from molfile import read_v2000
from uff import angle_bend_energy, bond_stretch_energy, partial_energy, torsion_energy, type_molecule, van_der_waals_energy


ROOT = Path(__file__).resolve().parents[1]


class TestUffTyping(unittest.TestCase):
    def test_types_triphenylene_without_external_dependencies(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")

        report = type_molecule(molecule)

        self.assertTrue(report.ready)
        self.assertEqual(len(report.types), 30)
        self.assertEqual(report.types.count("H_"), 12)
        self.assertTrue(all(atom_type in {"H_", "C_R"} for atom_type in report.types))
        energy = bond_stretch_energy(molecule)
        self.assertTrue(energy.ready)
        self.assertEqual(energy.terms, 33)
        self.assertGreaterEqual(energy.energy_kcal_mol, 0.0)
        angles = angle_bend_energy(molecule)
        self.assertTrue(angles.ready)
        self.assertEqual(angles.terms, 54)
        self.assertGreaterEqual(angles.energy_kcal_mol, 0.0)
        torsions = torsion_energy(molecule)
        self.assertTrue(torsions.ready)
        self.assertGreater(torsions.terms, 0)
        self.assertGreaterEqual(torsions.energy_kcal_mol, 0.0)
        vdw = van_der_waals_energy(molecule)
        self.assertTrue(vdw.ready)
        self.assertGreater(vdw.terms, 0)
        self.assertTrue(math.isfinite(vdw.energy_kcal_mol))
        total = partial_energy(molecule)
        self.assertTrue(total.ready)
        self.assertTrue(math.isfinite(total.energy_kcal_mol))

    def test_requires_explicit_hydrogens_for_bond_stretch_energy(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン).mol")

        energy = bond_stretch_energy(molecule)

        self.assertFalse(energy.ready)
        self.assertIsNone(energy.energy_kcal_mol)
        self.assertIn("implicit hydrogen", " ".join(energy.messages))
        angles = angle_bend_energy(molecule)
        self.assertFalse(angles.ready)
        self.assertIsNone(angles.energy_kcal_mol)
        torsions = torsion_energy(molecule)
        self.assertFalse(torsions.ready)
        self.assertIsNone(torsions.energy_kcal_mol)
        vdw = van_der_waals_energy(molecule)
        self.assertFalse(vdw.ready)
        self.assertIsNone(vdw.energy_kcal_mol)
        total = partial_energy(molecule)
        self.assertFalse(total.ready)
        self.assertIsNone(total.energy_kcal_mol)

    def test_reports_an_element_outside_the_first_supported_subset(self):
        molecule = read_v2000(ROOT / "131159-39-2(フラーレン).mol")

        report = type_molecule(molecule)

        self.assertTrue(report.ready)


if __name__ == "__main__":
    unittest.main()
