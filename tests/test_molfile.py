import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from molfile import (
    Atom, Bond, Molecule, MolfileError, _max_legitimate_valence, _target_valence,
    read_pdb, read_sdf, read_smiles, read_v2000, read_xyz,
    write_pdb, write_sdf, write_v2000, write_xyz,
)


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

    def test_reads_formal_charge_from_m_chg_record(self):
        molecule = read_v2000(ROOT / "tests" / "charged_ammonium.mol")

        self.assertEqual(molecule.atoms[0].element, "N")
        self.assertEqual(molecule.atoms[0].formal_charge, 1)
        self.assertEqual(molecule.atoms[0].formal_charge, 1)
        self.assertEqual(molecule.atoms[1].formal_charge, 0)

    def test_writes_a_v2000_file_that_round_trips_atoms_bonds_and_charges(self):
        original = Molecule(
            "Charged test molecule",
            (Atom("N", (0.0, 0.0, 0.0), 1), Atom("Cl", (1.7500, 0.0, 0.0))),
            (Bond(0, 1, 1),),
        )
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "round_trip.mol"
            write_v2000(original, target)
            restored = read_v2000(target)

        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.atoms, original.atoms)
        self.assertEqual(restored.bonds, original.bonds)

    def test_writes_standard_xyz_coordinates(self):
        molecule = Molecule("Water", (Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.9572, 0.0, 0.0))), ())
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "water.xyz"
            write_xyz(molecule, target)
            lines = target.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[0], "2")
        self.assertEqual(lines[1], "Water")
        self.assertEqual(lines[2], "O 0.00000000 0.00000000 0.00000000")
        self.assertEqual(lines[3], "H 0.95720000 0.00000000 0.00000000")

    def test_reads_xyz_atom_coordinates_without_bonds(self):
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "water.xyz"
            target.write_text("2\nWater\nO 0.0 0.0 0.0\nH 0.9572 0.0 0.0\n", encoding="utf-8")
            molecule = read_xyz(target)

        self.assertEqual(molecule.name, "Water")
        self.assertEqual(molecule.atoms, (Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.9572, 0.0, 0.0))))
        self.assertEqual(molecule.bonds, ())

    def test_reads_pdb_coordinates_and_conect_bonds(self):
        def atom_line(serial, element, x):
            return (
                f"HETATM{serial:>5} {element:<4} LIG A   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
                f"{1.0:6.2f}{0.0:6.2f}".ljust(76) + f"{element:>2}"
            )

        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "carbon_monoxide.pdb"
            target.write_text(
                "\n".join((atom_line(1, "C", 0.0), atom_line(2, "O", 1.13), "CONECT    1    2", "END")) + "\n",
                encoding="utf-8",
            )
            molecule = read_pdb(target)

        self.assertEqual(molecule.atoms, (Atom("C", (0.0, 0.0, 0.0)), Atom("O", (1.13, 0.0, 0.0))))
        self.assertEqual(molecule.bonds, (Bond(0, 1, 1),))

    def test_writes_and_reads_back_pdb_coordinates_and_connectivity(self):
        molecule = Molecule(
            "ethanol-like",
            (
                Atom("C", (0.0, 0.0, 0.0)),
                Atom("C", (1.5, 0.0, 0.0)),
                Atom("O", (2.2, 1.1, 0.0)),
            ),
            (Bond(0, 1, 1), Bond(1, 2, 2)),
        )
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "roundtrip.pdb"
            write_pdb(molecule, target)
            roundtripped = read_pdb(target)

        self.assertEqual(roundtripped.atoms, molecule.atoms)
        # PDB CONECT has no bond-order field; every bond reads back as
        # order 1, matching how read_pdb already treats any PDB file.
        self.assertEqual(
            roundtripped.bonds, (Bond(0, 1, 1), Bond(1, 2, 1)),
        )

    def test_reads_first_sdf_record_without_using_later_records(self):
        mol = (ROOT / "tests" / "charged_ammonium.mol").read_text(encoding="utf-8")
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "records.sdf"
            target.write_text(mol + "$$$$\nLater record\n", encoding="utf-8")
            molecule = read_sdf(target)

        self.assertEqual(molecule.atoms[0].element, "N")

    def test_writes_and_reads_back_sdf_atoms_bonds_and_charges(self):
        molecule = Molecule(
            "ethanol-like",
            (
                Atom("C", (0.0, 0.0, 0.0)),
                Atom("C", (1.5, 0.0, 0.0)),
                Atom("O", (2.2, 1.1, 0.0), formal_charge=-1),
            ),
            (Bond(0, 1, 1), Bond(1, 2, 2)),
        )
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "roundtrip.sdf"
            write_sdf(molecule, target)
            contents = target.read_text(encoding="utf-8")
            roundtripped = read_sdf(target)

        self.assertTrue(contents.rstrip("\n").endswith("$$$$"))
        self.assertEqual(roundtripped.atoms, molecule.atoms)
        self.assertEqual(roundtripped.bonds, (Bond(0, 1, 1), Bond(1, 2, 2)))

    def test_reads_basic_smiles_with_branch_ring_bond_orders_and_charge(self):
        branched = read_smiles("CC(=O)O", "Acetic acid")
        ring = read_smiles("C1CCCCC1")
        charged = read_smiles("C[NH3+]")

        self.assertEqual(branched.name, "Acetic acid")
        self.assertEqual([atom.element for atom in branched.atoms], ["C", "C", "O", "O"])
        self.assertEqual([bond.order for bond in branched.bonds], [1, 2, 1])
        self.assertEqual(len(ring.atoms), 6)
        self.assertEqual(len(ring.bonds), 6)
        self.assertEqual(charged.atoms[1].formal_charge, 1)
        self.assertTrue(any(atom.position != (0.0, 0.0, 0.0) for atom in branched.atoms))

    def test_reads_aromatic_smiles_as_editable_alternating_bonds(self):
        benzene = read_smiles("c1ccccc1", "Benzene")
        pyridine = read_smiles("n1ccccc1", "Pyridine")
        pyrrole = read_smiles("c1cc[nH]c1", "Pyrrole")

        self.assertEqual([atom.element for atom in benzene.atoms], ["C"] * 6)
        self.assertEqual(sum(bond.order == 2 for bond in benzene.bonds), 3)
        self.assertEqual(len(benzene.bonds), 6)
        self.assertEqual(pyridine.atoms[0].element, "N")
        self.assertEqual(sum(atom.element == "H" for atom in pyrrole.atoms), 1)

    def test_fused_ring_kekulization_keeps_trisubstituted_aromatic_n_trivalent(self):
        # Caffeine: each ring nitrogen bonded to a methyl plus two ring bonds
        # donates its lone pair into the aromatic system, so both of its ring
        # bonds must stay single in any valid Kekule structure. A naive
        # Kekulization that ignores valence when picking which aromatic bond
        # to promote to a double bond can push one of these nitrogens to
        # bond order 4, one above its valence of 3.
        caffeine = read_smiles("Cn1cnc2c1c(=O)n(c(=O)n2C)C", "Caffeine")
        bond_order_sum = [0] * len(caffeine.atoms)
        for bond in caffeine.bonds:
            bond_order_sum[bond.atom_a] += bond.order
            bond_order_sum[bond.atom_b] += bond.order
        for index, atom in enumerate(caffeine.atoms):
            target = _target_valence(atom)
            if target is not None:
                self.assertLessEqual(
                    bond_order_sum[index], target,
                    f"atom {index} ({atom.element}) exceeds its valence after Kekulization",
                )
        nitrogen_sums = [
            bond_order_sum[index] for index, atom in enumerate(caffeine.atoms) if atom.element == "N"
        ]
        self.assertEqual(nitrogen_sums, [3, 3, 3, 3])

    def test_hypervalent_sulfur_and_phosphorus_are_not_flagged_as_over_valence(self):
        # Sulfoxide/sulfone/sulfonamide sulfur and phosphate/phosphonate
        # phosphorus routinely exceed _target_valence's typical, neutral-state
        # value (used for implicit hydrogen counts) through legitimate
        # expanded-octet bonding. _max_legitimate_valence is the ceiling that
        # should actually gate an over-valence warning/rejection.
        cases = {
            "dimethyl sulfone": "CS(=O)(=O)C",
            "DMSO": "CS(=O)C",
            "phosphoric acid": "OP(=O)(O)O",
            "sulfonamide": "CS(=O)(=O)N",
            "sulfonic acid": "CS(=O)(=O)O",
        }
        for name, smiles in cases.items():
            with self.subTest(name=name):
                molecule = read_smiles(smiles, name)
                bond_order_sum = [0] * len(molecule.atoms)
                for bond in molecule.bonds:
                    bond_order_sum[bond.atom_a] += bond.order
                    bond_order_sum[bond.atom_b] += bond.order
                for index, atom in enumerate(molecule.atoms):
                    if atom.element not in ("S", "P"):
                        continue
                    self.assertLessEqual(
                        bond_order_sum[index], _max_legitimate_valence(atom),
                        f"{name}: atom {index} ({atom.element}) falsely over valence",
                    )
        # A genuinely impossible sulfur (more connections than any real
        # oxidation state allows) must still be flagged.
        overbonded_sulfur = Atom("S", (0.0, 0.0, 0.0))
        self.assertEqual(_max_legitimate_valence(overbonded_sulfur), 6)

    def test_preserves_tetrahedral_smiles_stereochemistry_in_3d_layout(self):
        clockwise = read_smiles("N[C@H](C)C(=O)O")
        anticlockwise = read_smiles("N[C@@H](C)C(=O)O")

        self.assertEqual(clockwise.atoms[1].chirality, "@")
        self.assertEqual(anticlockwise.atoms[1].chirality, "@@")
        self.assertTrue(any(atom.position[2] != 0.0 for atom in clockwise.atoms))

        def signed_volume(molecule):
            center = molecule.atoms[1].position
            vectors = [
                tuple(molecule.atoms[index].position[axis] - center[axis] for axis in range(3))
                for index in (0, 2, 3)
            ]
            first, second, third = vectors
            return (
                first[0] * (second[1] * third[2] - second[2] * third[1])
                - first[1] * (second[0] * third[2] - second[2] * third[0])
                + first[2] * (second[0] * third[1] - second[1] * third[0])
            )

        self.assertLess(signed_volume(clockwise) * signed_volume(anticlockwise), 0.0)

    def test_preserves_directional_double_bond_stereochemistry(self):
        opposite = read_smiles("F/C=C/F")
        matching = read_smiles("F/C=C\\F")

        self.assertEqual([bond.stereo for bond in opposite.bonds], ["/", None, "/"])
        self.assertEqual([bond.stereo for bond in matching.bonds], ["/", None, "\\"])

        def side_product(molecule):
            first, left, right, second = (atom.position for atom in molecule.atoms)
            axis_x, axis_y = right[0] - left[0], right[1] - left[1]
            first_side = axis_x * (first[1] - left[1]) - axis_y * (first[0] - left[0])
            second_side = axis_x * (second[1] - right[1]) - axis_y * (second[0] - right[0])
            return first_side * second_side

        self.assertLess(side_product(opposite), 0.0)
        self.assertGreater(side_product(matching), 0.0)


if __name__ == "__main__":
    unittest.main()
