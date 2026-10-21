"""Cross-check UFF_PARAMETERS against the published Open Babel UFF table.

``tests/openbabel_uff_311.prm`` is the unmodified ``data/UFF.prm`` data file
from Open Babel 3.1.1 (https://github.com/openbabel/openbabel, GPL-2.0-only),
fetched from the ``openbabel-3-1-1`` tag to match the version already
installed and used elsewhere in this repository's external validation
(``benchmarks/compare_avogadro_uff.py``). It transcribes the parameters
published in Rappe et al., J. Am. Chem. Soc. 1992, 114, 10024-10035 for every
element UFF defines.

This test does not implement additional atom types; it only guards the
per-atom-type parameter table already in use (and any added later) against
silent transcription errors, which is how the Cl/Br/I electronegativity
values were found to hold the "Hard" column instead of "Xi".
"""

from pathlib import Path
import math
import unittest

try:
    from .uff import UFF_PARAMETERS
except ImportError:
    from uff import UFF_PARAMETERS


ROOT = Path(__file__).parent
PRM_PATH = ROOT / "openbabel_uff_311.prm"


def _parse_reference_table() -> dict[str, tuple[float, ...]]:
    entries: dict[str, tuple[float, ...]] = {}
    for line in PRM_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("param"):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        name = parts[1]
        try:
            r1, theta0, x1, d1, _zeta, z1, vi, uj, xi, _hard, _radius = (
                float(value) for value in parts[2:13]
            )
        except ValueError:
            continue
        # Matches the UffParameter field order: skip zeta/Hard/Radius, which
        # this module does not use (no charge equilibration is implemented).
        entries[name] = (r1, theta0, x1, d1, z1, vi, uj, xi)
    return entries


class TestUffParametersAgainstOpenBabel(unittest.TestCase):
    def setUp(self):
        self.reference = _parse_reference_table()
        self.assertGreater(len(self.reference), 100, "reference fixture failed to parse")

    def test_every_implemented_atom_type_matches_the_published_table(self):
        for name, param in UFF_PARAMETERS.items():
            with self.subTest(atom_type=name):
                self.assertIn(name, self.reference, f"{name} is not a published UFF atom type")
                reference = self.reference[name]
                ours = (
                    param.bond_radius, param.ideal_angle, param.vdw_distance,
                    param.vdw_well_depth, param.effective_charge, param.torsion_barrier,
                    param.sp2_torsion_barrier, param.electronegativity,
                )
                for field, (expected, actual) in enumerate(zip(reference, ours)):
                    self.assertTrue(
                        math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6),
                        f"{name} field {field}: expected {expected}, got {actual}",
                    )

    def test_halogen_electronegativity_was_not_swapped_with_hardness(self):
        # Regression guard for the specific transcription error this fixture
        # caught: Cl/Br/I previously held the Hard column value in place of
        # Xi, silently shifting every C/N/O-halogen equilibrium bond length.
        self.assertEqual(UFF_PARAMETERS["Cl"].electronegativity, 8.564)
        self.assertEqual(UFF_PARAMETERS["Br"].electronegativity, 7.790)
        self.assertEqual(UFF_PARAMETERS["I_"].electronegativity, 6.822)


if __name__ == "__main__":
    unittest.main()
