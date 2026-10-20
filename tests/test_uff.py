import unittest
from pathlib import Path

from molfile import read_v2000
from uff import type_molecule


ROOT = Path(__file__).resolve().parents[1]


class TestUffTyping(unittest.TestCase):
    def test_types_triphenylene_without_external_dependencies(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")

        report = type_molecule(molecule)

        self.assertTrue(report.ready)
        self.assertEqual(len(report.types), 30)
        self.assertEqual(report.types.count("H_"), 12)
        self.assertTrue(all(atom_type in {"H_", "C_2"} for atom_type in report.types))

    def test_reports_an_element_outside_the_first_supported_subset(self):
        molecule = read_v2000(ROOT / "131159-39-2(フラーレン).mol")

        report = type_molecule(molecule)

        self.assertTrue(report.ready)


if __name__ == "__main__":
    unittest.main()
