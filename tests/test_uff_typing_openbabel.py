"""UFF typing, bond orders and group 15 inversion as in Open Babel 3.1.1.

Expected types and values below were read from Open Babel 3.1.1 itself
(benchmarks/compare_openbabel_main_group.py compares every energy term).
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from molfile import read_smiles  # noqa: E402
from uff import (  # noqa: E402
    _coordination, _inversion_coefficients, _torsion_bonds, out_of_plane_energy, type_molecule,
)

M0, M = "[H]C([H])([H])", "C([H])([H])[H]"


def heavy_types(smiles):
    molecule = read_smiles(smiles)
    report = type_molecule(molecule)
    return [t for atom, t in zip(molecule.atoms, report.types) if atom.element != "H"], report, molecule


class TestOpenBabelTyping(unittest.TestCase):
    def test_conjugated_but_not_aromatic(self):
        cases = {
            "[H]C([H])=C([H])C([H])=C([H])[H]": ["C_2"] * 4,              # butadiene
            f"{M0}C(=O){M}": ["C_3", "C_2", "O_2", "C_3"],                  # acetone
            f"{M0}C(=O)O[H]": ["C_3", "C_2", "O_2", "O_2"],                 # acetic acid
            f"{M0}C(=O)N([H]){M}": ["C_3", "C_2", "O_2", "N_2", "C_3"],     # N-methylacetamide
            f"[H]C([H])=C([H])O{M}": ["C_2", "C_2", "O_2", "C_3"],          # vinyl ether
            f"{M0}N([H]){M}": ["C_3", "N_3", "C_3"],                        # dimethylamine stays sp3
            f"{M0}C(=S){M}": ["C_3", "C_2", "S_2", "C_3"],                  # thioketone
        }
        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(heavy_types(smiles)[0], expected)

    def test_aromatic_rings(self):
        cases = {
            "[H]c1c([H])c([H])sc1[H]": ["C_R", "C_R", "C_R", "S_R", "C_R"],      # thiophene
            "[H]c1c([H])c([H])[se]c1[H]": ["C_R", "C_R", "C_R", "Se3+2", "C_R"],  # no Se_R in UFF
            "[H]c1c([H])c([H])oc1[H]": ["C_R", "C_R", "C_R", "O_R", "C_R"],      # furan
            "[H]N([H])c1c([H])c([H])c([H])c([H])c1[H]": ["N_2"] + ["C_R"] * 6,   # aniline
            "[H]Oc1c([H])c([H])c([H])c([H])c1[H]": ["O_2"] + ["C_R"] * 6,        # phenol
            # Open Babel's [#6H1D3r5] rule: an sp2 CH in a five-membered ring.
            "[H]C1=C([H])C([H])([H])C([H])=C1[H]": ["C_R", "C_R", "C_3", "C_R", "C_R"],
        }
        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(heavy_types(smiles)[0], expected)

    def test_heavier_main_group_types(self):
        cases = {
            f"{M0}[Ge]({M})({M}){M}": "Ge3", f"{M0}[Sn]({M})({M}){M}": "Sn3",
            f"{M0}[As]({M}){M}": "As3+3", f"{M0}[Sb]({M}){M}": "Sb3+3",
            f"{M0}[Se]{M}": "Se3+2", f"{M0}[Te]{M}": "Te3+2",
        }
        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(heavy_types(smiles)[0][1], expected)


class TestBondOrders(unittest.TestCase):
    def _order(self, smiles, elements):
        _types, report, molecule = heavy_types(smiles)
        for index, bond in enumerate(molecule.bonds):
            pair = {molecule.atoms[bond.atom_a].element, molecule.atoms[bond.atom_b].element}
            if pair == set(elements) and bond.order == 1:
                return report.bond_orders[index]
        raise AssertionError("bond not found")

    def test_amide_aromatic_and_biphenyl(self):
        self.assertAlmostEqual(self._order(f"{M0}C(=O)N([H]){M}", ("C", "N")), 1.41)
        self.assertAlmostEqual(self._order("[H]c1c([H])c([H])sc1[H]", ("C", "S")), 1.5)
        biphenyl = read_smiles("[H]c1c([H])c([H])c(c([H])c1[H])-c1c([H])c([H])c([H])c([H])c1[H]")
        report = type_molecule(biphenyl)
        link = next(i for i, b in enumerate(biphenyl.bonds)
                    if all(biphenyl.atoms[x].element == "C" for x in (b.atom_a, b.atom_b))
                    and report.bond_orders[i] == 1.0)
        self.assertEqual(report.bond_orders[link], 1.0, "a non-ring bond between aromatic atoms is single")

    def test_torsions_about_double_bonds_exist(self):
        molecule = read_smiles("[H]C([H])=C([H])[H]")
        report = type_molecule(molecule)
        double = [b for b, _p in _torsion_bonds(molecule, report) if b.order == 2]
        self.assertEqual(len(double), 1)


class TestGroup15Inversion(unittest.TestCase):
    def test_coefficients_reproduce_open_babel(self):
        # Open Babel 3.1.1 printed -35.232 kJ/mol (4.1868 kJ/kcal) for one
        # C-P(-C)-C inversion term of trimethylphosphine at 42.667 degrees.
        k, c0, c1, c2 = _inversion_coefficients("P_3+3", "P", ["C_3"] * 3)
        omega = math.radians(42.667)
        energy = k * (c0 + c1 * math.cos(omega) + c2 * math.cos(2.0 * omega)) * 4.1868
        self.assertAlmostEqual(energy, -35.232, delta=0.01)
        self.assertIsNotNone(_inversion_coefficients("As3+3", "As", ["C_3"] * 3))
        self.assertIsNone(_inversion_coefficients("C_3", "C", ["H_"] * 3))

    def test_trimethylphosphine_gets_inversion_terms(self):
        report = out_of_plane_energy(read_smiles(f"{M0}P({M}){M}"))
        self.assertEqual(report.terms, 3)
        self.assertLess(report.energy_kcal_mol, 0.0)


class TestCoordination(unittest.TestCase):
    def test_oxidation_state_suffix_is_not_hybridisation(self):
        # The trailing "2" of S_3+2/Se3+2 is an oxidation state; a three-
        # coordinate sulfoxide S was bent with the trigonal form until 0.1.87.
        for name in ("S_3+2", "Se3+2", "Te3+2", "P_3+3", "As3+3"):
            self.assertEqual(_coordination(name), "3", name)
        self.assertEqual(_coordination("S_R"), "R")
        self.assertEqual(_coordination("N_2"), "2")


if __name__ == "__main__":
    unittest.main()
