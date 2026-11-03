import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from elements import (
    COVALENT_RADII, DUMMY_SYMBOL, ELEMENT_COLORS, ELEMENT_RADII, ELEMENT_SYMBOLS, ELEMENTS,
    MAX_VALENCE, SMILES_ELEMENTS, TYPICAL_VALENCE, charged_typical_valence,
)
from molfile import Atom, Bond, Molecule, _max_legitimate_valence, _target_valence, read_smiles
from uff import UFF_PARAMETERS, _implicit_hydrogen_count, type_molecule


def _uff_type_element(type_name: str) -> str:
    return re.match(r"[A-Z][a-z]?", type_name).group(0)


class TestElementTable(unittest.TestCase):
    def test_derived_tables_cover_every_element(self):
        for table in (ELEMENT_COLORS, ELEMENT_RADII, COVALENT_RADII, TYPICAL_VALENCE, MAX_VALENCE):
            self.assertEqual(set(table), set(ELEMENTS))
        # The editor lists every element except the wildcard placeholder.
        self.assertEqual(set(ELEMENT_SYMBOLS), set(ELEMENTS) - {DUMMY_SYMBOL})
        self.assertEqual(len(ELEMENT_SYMBOLS), len(set(ELEMENT_SYMBOLS)))

    def test_editor_element_order_is_stable(self):
        # The editor's EnumProperty stores an index, so the original ten
        # elements must keep their positions; new ones are appended.
        self.assertEqual(
            ELEMENT_SYMBOLS[:10], ("H", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I")
        )

    def test_values_are_sane(self):
        for symbol, info in ELEMENTS.items():
            with self.subTest(element=symbol):
                self.assertEqual(len(info.color), 4)
                self.assertTrue(all(0.0 <= channel <= 1.0 for channel in info.color))
                self.assertGreater(info.display_radius, 0.0)
                self.assertGreater(info.covalent_radius, 0.0)
                self.assertGreaterEqual(info.max_valence, info.typical_valence)

    def test_every_uff_element_has_an_element_record(self):
        # Regression: B and Si had UFF types but no colour, radius, editor
        # entry or valence, so they could not be added from the panel.
        uff_elements = {_uff_type_element(name) for name in UFF_PARAMETERS}
        self.assertLessEqual(uff_elements, set(ELEMENTS))

    def test_every_smiles_element_has_a_uff_type(self):
        uff_elements = {_uff_type_element(name) for name in UFF_PARAMETERS}
        self.assertLessEqual(SMILES_ELEMENTS - {DUMMY_SYMBOL}, uff_elements)

    def test_halogens_have_their_own_display_radius(self):
        radii = [ELEMENT_RADII[symbol] for symbol in ("F", "Cl", "Br", "I")]
        self.assertEqual(radii, sorted(radii))
        self.assertGreater(ELEMENT_RADII["I"], ELEMENT_RADII["C"])


class TestBoronSiliconValence(unittest.TestCase):
    def test_typical_valence_for_boron_and_silicon(self):
        self.assertEqual(_target_valence(Atom("B", (0.0, 0.0, 0.0))), 3)
        self.assertEqual(_target_valence(Atom("B", (0.0, 0.0, 0.0), formal_charge=-1)), 4)
        self.assertEqual(_target_valence(Atom("Si", (0.0, 0.0, 0.0))), 4)
        self.assertEqual(_max_legitimate_valence(Atom("Si", (0.0, 0.0, 0.0))), 4)
        self.assertEqual(charged_typical_valence("N", 1), 4)
        self.assertEqual(charged_typical_valence("O", -1), 1)
        self.assertIsNone(charged_typical_valence("Xx", 0))

    def test_uff_counts_implicit_hydrogens_on_boron_and_silicon(self):
        silicon = Molecule("Si", (Atom("Si", (0.0, 0.0, 0.0)),), ())
        borate = Molecule("BH4-", (Atom("B", (0.0, 0.0, 0.0), formal_charge=-1),), ())
        self.assertEqual(_implicit_hydrogen_count(silicon), 4)
        self.assertEqual(_implicit_hydrogen_count(borate), 4)

    def test_explicit_silane_and_borane_are_typed(self):
        def star(center: str, count: int, charge: int = 0) -> Molecule:
            directions = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))[:count]
            atoms = (Atom(center, (0.0, 0.0, 0.0), formal_charge=charge),) + tuple(
                Atom("H", tuple(0.85 * value for value in direction)) for direction in directions
            )
            bonds = tuple(Bond(0, index + 1, 1) for index in range(count))
            return Molecule(center, atoms, bonds)

        cases = {"SiH4": (star("Si", 4), "Si3"), "BH3": (star("B", 3), "B_2"),
                 "BH4-": (star("B", 4, -1), "B_3")}
        for name, (molecule, expected) in cases.items():
            with self.subTest(molecule=name):
                report = type_molecule(molecule)
                self.assertEqual(report.messages, ())
                self.assertEqual(report.types[0], expected)
                self.assertEqual(_implicit_hydrogen_count(molecule), 0)

    def test_boronic_acid_smiles_keeps_boron_within_valence(self):
        molecule = read_smiles("OB(O)c1ccccc1", "Phenylboronic acid")
        bond_order_sum = [0] * len(molecule.atoms)
        for bond in molecule.bonds:
            bond_order_sum[bond.atom_a] += bond.order
            bond_order_sum[bond.atom_b] += bond.order
        boron = next(index for index, atom in enumerate(molecule.atoms) if atom.element == "B")
        self.assertEqual(bond_order_sum[boron], 3)


if __name__ == "__main__":
    unittest.main()
