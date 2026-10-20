import unittest
from pathlib import Path

from molfile import read_v2000


ROOT = Path(__file__).resolve().parents[1]


class TestTriphenyleneMolfile(unittest.TestCase):
    def test_reads_atoms_bonds_and_orders(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン).mol")

        self.assertEqual(len(molecule.atoms), 18)
        self.assertEqual(len(molecule.bonds), 21)
        self.assertTrue(all(atom.element == "C" for atom in molecule.atoms))
        self.assertEqual(sum(bond.order == 2 for bond in molecule.bonds), 9)
        self.assertTrue(all(atom.position[2] == 0.0 for atom in molecule.atoms))

    def test_reads_optimized_3d_structure_with_explicit_hydrogens(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")

        self.assertEqual(len(molecule.atoms), 30)
        self.assertEqual(len(molecule.bonds), 33)
        self.assertEqual(sum(atom.element == "C" for atom in molecule.atoms), 18)
        self.assertEqual(sum(atom.element == "H" for atom in molecule.atoms), 12)
        self.assertTrue(any(atom.position[2] != 0.0 for atom in molecule.atoms))


if __name__ == "__main__":
    unittest.main()
