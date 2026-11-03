"""3D distance-geometry embedding of SMILES (blend_mol/embed.py)."""

import math
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
import embed  # noqa: E402
from molfile import read_smiles  # noqa: E402


def _volume(positions, center, a, b, c):
    pc = positions[center]
    u = [positions[a][k] - pc[k] for k in range(3)]
    v = [positions[b][k] - pc[k] for k in range(3)]
    w = [positions[c][k] - pc[k] for k in range(3)]
    return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


def _dihedral(positions, i, j, k, l):
    return math.degrees(embed._dihedral_and_gradient(positions, i, j, k, l)[0])


def _closest_nonbonded(molecule):
    count = len(molecule.atoms)
    near = [set() for _ in range(count)]
    for bond in molecule.bonds:
        near[bond.atom_a].add(bond.atom_b)
        near[bond.atom_b].add(bond.atom_a)
    within_two = [set(n) | {i} for i, n in enumerate(near)]
    for i in range(count):
        for n in near[i]:
            within_two[i] |= near[n]
    return min(
        (math.dist(molecule.atoms[i].position, molecule.atoms[j].position)
         for i in range(count) for j in range(i + 1, count) if j not in within_two[i]),
        default=math.inf,
    )


class TestStereo(unittest.TestCase):
    def test_tetrahedral_handedness_matches_open_babel_convention(self):
        # Verified once against Open Babel's perception from 3D coordinates
        # (benchmarks/evaluate_smiles_embedding.py); pinned here so the sign
        # convention cannot silently flip. Atoms: N0, CA1, CB2, C3. For
        # L-alanine the N, CH3, COOH triple product about CA is negative.
        l_alanine = [a.position for a in read_smiles("N[C@@H](C)C(=O)O").atoms]
        d_alanine = [a.position for a in read_smiles("N[C@H](C)C(=O)O").atoms]
        self.assertLess(_volume(l_alanine, 1, 0, 2, 3), -1.0)
        self.assertGreater(_volume(d_alanine, 1, 0, 2, 3), 1.0)

    def test_neighbour_order_rules(self):
        # L-alanine written three ways (all confirmed as L by Open Babel):
        # neighbours in a different order, and the implicit H first. Each is
        # (center, N, CH3, carboxyl C) and must give the L handedness.
        variants = {
            "N[C@@H](C)C(=O)O": (1, 0, 2, 3),
            "[C@@H](N)(C(=O)O)C": (0, 1, 5, 2),
            "C[C@H](N)C(=O)O": (1, 2, 0, 3),
        }
        for smiles, (center, n, cb, c) in variants.items():
            with self.subTest(smiles=smiles):
                positions = [a.position for a in read_smiles(smiles).atoms]
                self.assertLess(_volume(positions, center, n, cb, c), -1.0)

    def test_double_bond_geometry(self):
        trans = [a.position for a in read_smiles("F/C=C/F").atoms]
        cis = [a.position for a in read_smiles("F/C=C\\F").atoms]
        self.assertGreater(abs(_dihedral(trans, 0, 1, 2, 3)), 170.0)
        self.assertLess(abs(_dihedral(cis, 0, 1, 2, 3)), 10.0)
        # Each of the three double bonds of a mixed E/Z triene.
        triene = [a.position for a in read_smiles("C/C=C/C=C\\C=C/C").atoms]
        self.assertGreater(abs(_dihedral(triene, 0, 1, 2, 3)), 170.0)
        self.assertLess(abs(_dihedral(triene, 2, 3, 4, 5)), 10.0)
        self.assertLess(abs(_dihedral(triene, 4, 5, 6, 7)), 10.0)


class TestGeometry(unittest.TestCase):
    def test_bond_lengths_and_no_clashes(self):
        for smiles in ("CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
                       "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "C1C2CC3CC1CC(C2)C3"):
            with self.subTest(smiles=smiles):
                molecule = read_smiles(smiles)
                self.assertGreater(_closest_nonbonded(molecule), 2.0)
                for bond in molecule.bonds:
                    a, b = molecule.atoms[bond.atom_a], molecule.atoms[bond.atom_b]
                    ideal = embed._bond_length(a.element, b.element, bond.order)
                    self.assertAlmostEqual(math.dist(a.position, b.position), ideal, delta=0.05)

    def test_aromatic_ring_is_planar(self):
        positions = [a.position for a in read_smiles("c1ccccc1").atoms]
        for start in range(6):
            angle = _dihedral(positions, start, (start + 1) % 6, (start + 2) % 6, (start + 3) % 6)
            self.assertLess(abs(angle), 5.0)

    def test_cyclohexane_is_a_chair(self):
        positions = [a.position for a in read_smiles("C1CCCCC1").atoms]
        angles = [_dihedral(positions, i, (i + 1) % 6, (i + 2) % 6, (i + 3) % 6) for i in range(6)]
        for index, angle in enumerate(angles):
            self.assertTrue(38.0 <= abs(angle) <= 78.0, angles)
            self.assertNotEqual(angle > 0, angles[index - 1] > 0, f"signs must alternate: {angles}")

    def test_linear_centres(self):
        for smiles in ("O=C=O", "CC#N", "C#CC#C"):
            with self.subTest(smiles=smiles):
                p = [a.position for a in read_smiles(smiles).atoms]
                u = [p[0][k] - p[1][k] for k in range(3)]
                v = [p[2][k] - p[1][k] for k in range(3)]
                cosine = sum(x * y for x, y in zip(u, v)) / (math.dist(p[0], p[1]) * math.dist(p[2], p[1]))
                self.assertLess(cosine, -0.999)

    def test_deterministic_and_reasonably_fast(self):
        smiles = "C" + "C(C)" * 49  # 99 atoms
        start = time.perf_counter()
        first = read_smiles(smiles)
        elapsed = time.perf_counter() - start
        second = read_smiles(smiles)
        self.assertEqual([a.position for a in first.atoms], [a.position for a in second.atoms])
        # The target is under 1 s on the development machine; allow headroom
        # for slower CI runners.
        self.assertLess(elapsed, 5.0)

    def test_planar_layout_is_still_available(self):
        molecule = read_smiles("CCO", embed_3d=False)
        self.assertTrue(all(atom.position[2] == 0.0 for atom in molecule.atoms))


if __name__ == "__main__":
    unittest.main()
