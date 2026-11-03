"""First, dependency-free UFF atom typing support for Blend Mol.

This module deliberately contains no Blender API calls.  It is the boundary
between the editable molecular graph and a future energy/minimisation engine.
Only the well-defined organic subset is enabled at this stage; unknown or
ambiguous atoms are reported rather than silently receiving a generic type.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable


_KJ_TO_KCAL = 1.0 / 4.184

# A single small-energy-change step is not reliable evidence of convergence: a
# short interactive chunk (Live UFF settles in batches of 1-10 iterations) can
# take one incidentally tiny step -- e.g. from line-search backtracking --
# while the gradient is still well above gradient_tolerance. Require this many
# *consecutive* steps under energy_tolerance before trusting it.
_ENERGY_TOLERANCE_STREAK = 3

try:
    from .embed import _rings
    from .elements import charged_typical_valence
    from .molfile import Atom, Molecule
    from .native_backend import (
        residual_energy_gradient as _native_residual_energy_gradient,
        vdw_energy_gradient as _native_vdw_energy_gradient,
    )
except ImportError:  # Enables the small module to be tested outside Blender.
    from embed import _rings
    from elements import charged_typical_valence
    from molfile import Atom, Molecule
    from native_backend import (
        residual_energy_gradient as _native_residual_energy_gradient,
        vdw_energy_gradient as _native_vdw_energy_gradient,
    )


@dataclass(frozen=True)
class UffParameter:
    """UFF per-atom parameters in the published UFF ordering and units."""

    name: str
    bond_radius: float
    ideal_angle: float
    vdw_distance: float
    vdw_well_depth: float
    effective_charge: float
    torsion_barrier: float
    sp2_torsion_barrier: float
    electronegativity: float


# Organic subset transcribed from the UFF parameter set.  The remaining UFF
# columns are added only when their associated energy terms are implemented.
UFF_PARAMETERS = {
    "H_": UffParameter("H_", 0.354, 180.0, 2.886, 0.044, 0.712, 0.0, 0.0, 4.528),
    "H_b": UffParameter("H_b", 0.460, 83.5, 2.886, 0.044, 0.712, 0.0, 0.0, 4.528),
    "C_3": UffParameter("C_3", 0.757, 109.47, 3.851, 0.105, 1.912, 2.119, 2.0, 5.343),
    "C_2": UffParameter("C_2", 0.732, 120.0, 3.851, 0.105, 1.912, 0.0, 2.0, 5.343),
    "C_1": UffParameter("C_1", 0.706, 180.0, 3.851, 0.105, 1.912, 0.0, 2.0, 5.343),
    "C_R": UffParameter("C_R", 0.729, 120.0, 3.851, 0.105, 1.912, 0.0, 2.0, 5.343),
    "N_3": UffParameter("N_3", 0.700, 106.7, 3.660, 0.069, 2.544, 0.450, 2.0, 6.899),
    "N_2": UffParameter("N_2", 0.685, 111.2, 3.660, 0.069, 2.544, 0.0, 2.0, 6.899),
    "N_1": UffParameter("N_1", 0.656, 180.0, 3.660, 0.069, 2.544, 0.0, 2.0, 6.899),
    "N_R": UffParameter("N_R", 0.699, 120.0, 3.660, 0.069, 2.544, 0.0, 2.0, 6.899),
    "O_3": UffParameter("O_3", 0.658, 104.51, 3.500, 0.060, 2.300, 0.018, 2.0, 8.741),
    "O_2": UffParameter("O_2", 0.634, 120.0, 3.500, 0.060, 2.300, 0.0, 2.0, 8.741),
    "O_1": UffParameter("O_1", 0.639, 180.0, 3.500, 0.060, 2.300, 0.0, 2.0, 8.741),
    "O_R": UffParameter("O_R", 0.680, 110.0, 3.500, 0.060, 2.300, 0.0, 2.0, 8.741),
    "F_": UffParameter("F_", 0.668, 180.0, 3.364, 0.050, 1.735, 0.0, 2.0, 10.874),
    "P_3+3": UffParameter("P_3+3", 1.101, 93.8, 4.147, 0.305, 2.863, 2.4, 1.25, 5.463),
    "P_3+5": UffParameter("P_3+5", 1.056, 109.47, 4.147, 0.305, 2.863, 2.4, 1.25, 5.463),
    "S_3+2": UffParameter("S_3+2", 1.064, 92.1, 4.035, 0.274, 2.703, 0.484, 1.25, 6.928),
    "S_R": UffParameter("S_R", 1.077, 92.2, 4.035, 0.274, 2.703, 0.0, 1.25, 6.928),
    "S_2": UffParameter("S_2", 0.854, 120.0, 4.035, 0.274, 2.703, 0.0, 1.25, 6.928),
    "Si3": UffParameter("Si3", 1.117, 109.47, 4.295, 0.402, 2.323, 1.225, 1.25, 4.168),
    "B_2": UffParameter("B_2", 0.828, 120.0, 4.083, 0.18, 1.755, 0.0, 2.0, 5.11),
    "B_3": UffParameter("B_3", 0.838, 109.47, 4.083, 0.18, 1.755, 0.0, 2.0, 5.11),
    # Xi (electronegativity) corrected from the published UFF/Open Babel
    # table: these three previously held the Hard (QEq hardness) column
    # value instead, silently shifting the C/N/O-X equilibrium bond length
    # by the electronegativity correction term in _equilibrium_bond_length.
    "Cl": UffParameter("Cl", 1.044, 180.0, 3.947, 0.227, 2.348, 0.0, 1.25, 8.564),
    "Br": UffParameter("Br", 1.192, 180.0, 4.189, 0.251, 2.519, 0.0, 0.70, 7.790),
    "I_": UffParameter("I_", 1.382, 180.0, 4.500, 0.339, 2.650, 0.0, 0.20, 6.822),
    # Heavier main-group elements (0.1.87), one generic type each, as in
    # Open Babel 3.1.1's UFF.prm atom typing.
    "Ge3": UffParameter("Ge3", 1.197, 109.47, 4.280, 0.379, 2.789, 0.701, 0.7, 4.051),
    "Sn3": UffParameter("Sn3", 1.398, 109.47, 4.392, 0.567, 2.961, 0.199, 0.2, 3.987),
    "As3+3": UffParameter("As3+3", 1.211, 92.1, 4.230, 0.309, 2.864, 1.5, 0.7, 5.188),
    "Sb3+3": UffParameter("Sb3+3", 1.407, 91.6, 4.420, 0.449, 2.704, 1.1, 0.2, 4.899),
    "Se3+2": UffParameter("Se3+2", 1.190, 90.6, 4.205, 0.291, 2.764, 0.335, 0.7, 6.428),
    "Te3+2": UffParameter("Te3+2", 1.386, 90.25, 4.470, 0.398, 2.882, 0.3, 0.2, 5.816),
}

# Hybridisation per atom type ("1" linear, "2"/"R" sp2 or resonant, "3"
# sp3), used for torsion dispatch and for choosing the angle-bend form.
# Open Babel reads it from the type name's third character. It cannot be
# taken from the last character: the "+N" suffix of P_3+3, S_3+2, Se3+2 and
# Te3+2 marks oxidation state and collides with the sp2/sp3 digits (a
# three-coordinate S_3+2 was bent with the trigonal form until 0.1.87).
_TORSION_COORDINATION: dict[str, str] = {
    "H_": "1", "H_b": "1",
    "C_3": "3", "C_2": "2", "C_1": "1", "C_R": "R",
    "N_3": "3", "N_2": "2", "N_1": "1", "N_R": "R",
    "O_3": "3", "O_2": "2", "O_1": "1", "O_R": "R",
    "F_": "1",
    "P_3+3": "3", "P_3+5": "3",
    "S_3+2": "3", "S_R": "R", "S_2": "2",
    "Si3": "3",
    "B_2": "2", "B_3": "3",
    "Cl": "1", "Br": "1", "I_": "1",
    "Ge3": "3", "Sn3": "3", "As3+3": "3", "Sb3+3": "3", "Se3+2": "3", "Te3+2": "3",
}


def _coordination(type_name: str) -> str:
    return _TORSION_COORDINATION.get(type_name, type_name[-1])


@dataclass(frozen=True)
class UffTypingReport:
    types: tuple[str | None, ...]
    messages: tuple[str, ...]
    # Bond order UFF uses for each molecule.bonds entry: 1.5 in an aromatic
    # ring, 1.41 for an amide C-N, otherwise the stored order (Open Babel 3.1.1).
    bond_orders: tuple[float, ...] = ()

    def bond_order(self, index: int, stored: int) -> float:
        return self.bond_orders[index] if index < len(self.bond_orders) else float(stored)

    @property
    def ready(self) -> bool:
        return not self.messages and all(atom_type is not None for atom_type in self.types)


@dataclass(frozen=True)
class PreparedUffContext:
    """Coordinate-independent UFF data reusable across optimization steps."""

    topology_key: tuple
    typing: UffTypingReport
    torsion_terms: tuple
    inversion_terms: tuple
    numeric_atoms: frozenset[int]
    neighbors: tuple[tuple[tuple[int, int], ...], ...]
    vdw_pairs: tuple[tuple[int, int, float, float], ...]


@dataclass(frozen=True)
class UffBondStretchReport:
    """Partial UFF energy from all explicit bonds, in kcal mol⁻¹."""

    energy_kcal_mol: float | None
    terms: int
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.energy_kcal_mol is not None and not self.messages


@dataclass(frozen=True)
class UffAngleBendReport:
    """Partial UFF angle-bend energy, in kcal mol⁻¹."""

    energy_kcal_mol: float | None
    terms: int
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.energy_kcal_mol is not None and not self.messages


@dataclass(frozen=True)
class UffTorsionReport:
    """Partial UFF proper-torsion energy, in kcal mol⁻¹."""

    energy_kcal_mol: float | None
    terms: int
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.energy_kcal_mol is not None and not self.messages


@dataclass(frozen=True)
class UffVanDerWaalsReport:
    """Partial UFF Lennard-Jones 12-6 energy, in kcal mol⁻¹."""

    energy_kcal_mol: float | None
    terms: int
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.energy_kcal_mol is not None and not self.messages


@dataclass(frozen=True)
class UffPartialEnergyReport:
    """The currently implemented, explicitly labelled subset of UFF energy."""

    energy_kcal_mol: float | None
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.energy_kcal_mol is not None and not self.messages


@dataclass(frozen=True)
class UffOutOfPlaneReport:
    """Partial UFF out-of-plane inversion energy, in kcal mol⁻¹."""

    energy_kcal_mol: float | None
    terms: int
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.energy_kcal_mol is not None and not self.messages


@dataclass(frozen=True)
class UffOptimizerState:
    """Coordinate optimizer state that can be continued across Live UFF chunks."""

    algorithm: str
    velocity: tuple[tuple[float, float, float], ...] = ()
    time_step: float = 0.05
    alpha: float = 0.25
    positive_steps: int = 0
    initialized: bool = False
    history_s: tuple[tuple[float, ...], ...] = ()
    history_y: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class UffMinimizationResult:
    molecule: Molecule | None
    initial_energy_kcal_mol: float | None
    final_energy_kcal_mol: float | None
    iterations: int
    termination: str
    messages: tuple[str, ...]
    optimizer_state: UffOptimizerState | None = None

    @property
    def ready(self) -> bool:
        return self.molecule is not None and not self.messages


def _type_from_bond_orders(element: str, bond_orders: list[int]) -> str:
    """Hybridisation from bond orders, recognising cumulated double bonds.

    A plain "highest bond order" rule mistypes a cumulated system such as
    CO2, an isocyanate, or an allene: the central atom has two separate
    double bonds (max order 2) but is sp-hybridised (linear, 180°), not
    sp2. Open Babel's SMARTS-based typing reaches the same conclusion from
    real computed hybridisation; this is the bond-order equivalent for an
    atom with two or more order-2 bonds.
    """

    max_order = max(bond_orders, default=1)
    double_bond_count = sum(1 for order in bond_orders if order == 2)
    if max_order >= 3 or double_bond_count >= 2:
        return f"{element}_1"
    if max_order == 2:
        return f"{element}_2"
    return f"{element}_3"


def type_molecule(molecule: Molecule) -> UffTypingReport:
    """Assign conservative UFF types from explicit bonds.

    MOL V2000 bond order does not reliably preserve aromaticity in every
    source file, so this first version does not infer ``*_R`` types.  It uses
    the corresponding non-aromatic hybridisation type and reports only cases
    that cannot be assigned safely.
    """

    neighbors: list[list[tuple[int, int]]] = [[] for _ in molecule.atoms]
    messages: list[str] = []
    for bond in molecule.bonds:
        if not 0 <= bond.atom_a < len(molecule.atoms) or not 0 <= bond.atom_b < len(molecule.atoms):
            messages.append("UFF: a bond references an atom that does not exist")
            continue
        neighbors[bond.atom_a].append((bond.atom_b, bond.order))
        neighbors[bond.atom_b].append((bond.atom_a, bond.order))

    types: list[str | None] = []
    for index, atom in enumerate(molecule.atoms):
        bond_orders = [order for _neighbor, order in neighbors[index]]
        degree = len(bond_orders)
        max_order = max(bond_orders, default=1)
        atom_type: str | None = None
        if atom.element == "H":
            if degree <= 1:
                atom_type = "H_"
            elif degree == 2:
                atom_type = "H_b"
        elif atom.element == "C" and atom.formal_charge == 0:
            atom_type = _type_from_bond_orders("C", bond_orders)
        elif atom.element == "N" and -1 <= atom.formal_charge <= 1:
            atom_type = _type_from_bond_orders("N", bond_orders)
        elif atom.element == "O" and -1 <= atom.formal_charge <= 1:
            atom_type = _type_from_bond_orders("O", bond_orders)
        elif atom.element == "F" and atom.formal_charge == 0 and degree <= 1:
            atom_type = "F_"
        elif atom.element == "P" and atom.formal_charge == 0 and degree == 5 and max_order == 1:
            # Hypervalent P (PCl5/PF5-like); Open Babel's own SMARTS rule for
            # this ([#15D5]) keys only on having five single-bonded neighbors.
            atom_type = "P_3+5"
        elif atom.element == "P" and atom.formal_charge == 0 and (
            (degree <= 3 and max_order == 1) or (degree == 4 and max_order == 2)
        ):
            # Open Babel's UFF typing has no dedicated phosphate/phosphonate
            # rule for a neutral, tetravalent P=O center (its SMARTS P_3+5
            # rule requires five single-bonded neighbors, not four with one
            # double bond); it falls back to generic P_3+3 for that shape,
            # confirmed by running Open Babel 3.1.1 on a neutral R3P=O
            # structure. Matching that fallback keeps this typing consistent
            # with the same reference this project validates energies
            # against, rather than inventing an unvalidated phosphonate type.
            atom_type = "P_3+3"
        elif atom.element == "S" and atom.formal_charge == 0 and degree == 1 and max_order == 2:
            # [S^2] S_2: thiocarbonyl sulfur (C=S).
            atom_type = "S_2"
        elif atom.element == "S" and atom.formal_charge == 0 and (
            (degree <= 2 and max_order == 1) or (degree in (3, 4) and max_order == 2)
        ):
            # Likewise, Open Babel's S_3+4/S_3+6 rules key on explicit +4/+6
            # formal charges (common for SO2/SO3 drawn with charge-separated
            # Lewis structures), not on a neutral sulfoxide/sulfone's double
            # bonds; confirmed it falls back to generic S_3+2 for neutral
            # dimethyl sulfoxide (one S=O, degree 3).
            atom_type = "S_3+2"
        elif atom.element == "Si" and atom.formal_charge == 0 and degree <= 4 and max_order == 1:
            # UFF has only one silicon type; a multiply-bonded Si (silene,
            # silyne) has no correct type available and is left unsupported.
            atom_type = "Si3"
        elif atom.element == "B" and -1 <= atom.formal_charge <= 0 and max_order == 1 and degree <= 3:
            atom_type = "B_2"
        elif atom.element == "B" and -1 <= atom.formal_charge <= 0 and max_order == 1 and degree == 4:
            # Tetracoordinate boron is almost always anionic (BF4-,
            # borohydrides); formal_charge is not restricted to 0 here.
            atom_type = "B_3"
        elif atom.element in {"Ge", "Sn"} and atom.formal_charge == 0 and degree <= 4 and max_order == 1:
            # Like silicon: one generic sp3 type, single bonds only.
            atom_type = {"Ge": "Ge3", "Sn": "Sn3"}[atom.element]
        elif atom.element in {"As", "Sb"} and atom.formal_charge == 0 and (
            (degree <= 3 and max_order == 1) or (degree == 4 and max_order == 2)
        ):
            # Same coordinations as P_3+3 (arsine/stibine, R3As=O).
            atom_type = {"As": "As3+3", "Sb": "Sb3+3"}[atom.element]
        elif atom.element in {"Se", "Te"} and atom.formal_charge == 0 and (
            (degree <= 2 and max_order == 1) or (degree in (3, 4) and max_order == 2)
        ):
            # Same coordinations as S_3+2 (selenide/telluride, selenoxide).
            atom_type = {"Se": "Se3+2", "Te": "Te3+2"}[atom.element]
        elif atom.element in {"Cl", "Br"} and atom.formal_charge == 0 and degree <= 1:
            atom_type = atom.element
        elif atom.element == "I" and atom.formal_charge == 0 and degree <= 1:
            atom_type = "I_"

        if atom_type not in UFF_PARAMETERS:
            types.append(None)
            if atom.element == "*":
                messages.append(
                    f"UFF: atom {index + 1} is a wildcard (*) placeholder; change it to a real element first"
                )
            else:
                messages.append(
                    f"UFF: atom {index + 1} ({atom.element}, charge {atom.formal_charge:+d}) is not supported"
                )
        else:
            types.append(atom_type)
    # Resonant (_R) types follow Open Babel 3.1.1's UFF.prm, which reserves
    # them for aromatic atoms ([c] C_R, [n] N_R, [o] O_R, [s] S_R). A
    # conjugated but non-aromatic C stays C_2 (until 0.1.87 it became C_R,
    # e.g. every carbonyl carbon, which shifted torsions by tens of kJ/mol).
    aromatic_atoms, aromatic_bonds = _perceive_aromaticity(molecule, neighbors)
    aromatic_type = {"C": "C_R", "N": "N_R", "O": "O_R", "S": "S_R"}
    for index in aromatic_atoms:
        element = molecule.atoms[index].element
        if element in aromatic_type and types[index] is not None:
            types[index] = aromatic_type[element]
    rings = _small_rings(molecule, neighbors)
    for index, atom in enumerate(molecule.atoms):
        # [#6H1D3r5] C_R: a CH with three connections in a five-membered ring.
        if (
            atom.element == "C" and types[index] is not None and len(neighbors[index]) == 3
            and sum(1 for n, _o in neighbors[index] if molecule.atoms[n].element == "H") == 1
            and any(len(ring) == 5 and index in ring for ring in rings)
        ):
            types[index] = "C_R"
    # Open Babel perceives a single-bonded N or O next to a multiply bonded
    # or aromatic atom as sp2 (amide, aniline, phenol, ester, enamine, nitro).
    sp2_types = {"C_2", "C_R", "N_2", "N_R", "O_2", "O_R", "S_2", "S_R"}
    snapshot = list(types)
    for index, atom in enumerate(molecule.atoms):
        if snapshot[index] not in ("N_3", "O_3") or len(neighbors[index]) > 3:
            continue
        if any(snapshot[n] in sp2_types for n, _order in neighbors[index]):
            types[index] = "N_2" if snapshot[index] == "N_3" else "O_2"
    bond_orders = []
    for index, bond in enumerate(molecule.bonds):
        if (min(bond.atom_a, bond.atom_b), max(bond.atom_a, bond.atom_b)) in aromatic_bonds:
            bond_orders.append(1.5)
        elif _is_amide_bond(molecule, neighbors, bond):
            bond_orders.append(1.41)
        else:
            bond_orders.append(float(bond.order))
    return UffTypingReport(tuple(types), tuple(messages), tuple(bond_orders))


def _small_rings(molecule: Molecule, neighbors) -> list[frozenset[int]]:
    plain = [[n for n, _order in attached] for attached in neighbors]
    return _rings(len(molecule.atoms), plain, max_size=8)


def _is_amide_bond(molecule: Molecule, neighbors, bond) -> bool:
    """C(=O)-N single bond (Open Babel's OBBond::IsAmide)."""
    if bond.order != 1:
        return False
    pair = (bond.atom_a, bond.atom_b)
    elements = {molecule.atoms[i].element for i in pair}
    if elements != {"C", "N"}:
        return False
    carbon = bond.atom_a if molecule.atoms[bond.atom_a].element == "C" else bond.atom_b
    return any(molecule.atoms[n].element == "O" and order == 2 for n, order in neighbors[carbon])


def _ring_pi_electrons(molecule: Molecule, neighbors, ring: frozenset[int], ring_atoms: set[int]) -> int | None:
    """Pi electrons a ring contributes in a simple Hueckel model, or None if not conjugated."""
    total = 0
    for index in ring:
        atom = molecule.atoms[index]
        orders = neighbors[index]
        if any(order >= 3 for _n, order in orders):
            return None
        double_partners = [n for n, order in orders if order == 2]
        if double_partners:
            partner = double_partners[0]
            # A double bond inside this ring, or to an atom of a fused ring,
            # puts one electron in the ring; an exocyclic C=O/C=S/C=N none.
            total += 1 if (partner in ring or partner in ring_atoms) else 0
        elif atom.element in ("N", "P") and atom.formal_charge == 0:
            total += 2
        elif atom.element in ("O", "S", "Se", "Te") and atom.formal_charge == 0 and len(orders) == 2:
            total += 2
        elif atom.element == "C" and atom.formal_charge == -1:
            total += 2
        elif atom.element == "C" and atom.formal_charge == 1:
            total += 0
        elif atom.element == "B" and atom.formal_charge == 0:
            total += 0
        else:
            return None
    return total


def _perceive_aromaticity(molecule: Molecule, neighbors) -> tuple[set[int], set[tuple[int, int]]]:
    """Atoms and bonds of 5-7 membered (and fused-pair) rings with 4n+2 pi electrons."""
    rings = [ring for ring in _small_rings(molecule, neighbors) if 5 <= len(ring) <= 7]
    ring_atoms = set().union(*rings) if rings else set()
    candidates = list(rings)
    for first in range(len(rings)):
        for second in range(first + 1, len(rings)):
            if len(rings[first] & rings[second]) >= 2:
                candidates.append(rings[first] | rings[second])
    bonded = {
        (min(bond.atom_a, bond.atom_b), max(bond.atom_a, bond.atom_b)) for bond in molecule.bonds
    }
    aromatic_atoms: set[int] = set()
    aromatic_bonds: set[tuple[int, int]] = set()
    for ring in candidates:
        electrons = _ring_pi_electrons(molecule, neighbors, ring, ring_atoms)
        if electrons is None or electrons % 4 != 2:
            continue
        aromatic_atoms |= ring
        component_rings = [r for r in rings if r <= ring]
        for r in component_rings:
            for a in r:
                for b in r:
                    if a < b and (a, b) in bonded:
                        aromatic_bonds.add((a, b))
    return aromatic_atoms, aromatic_bonds


def _implicit_hydrogen_count(molecule: Molecule) -> int:
    """Return virtual hydrogens that would make a bond-only UFF energy partial."""

    bond_sums = [0] * len(molecule.atoms)
    for bond in molecule.bonds:
        if 0 <= bond.atom_a < len(molecule.atoms) and 0 <= bond.atom_b < len(molecule.atoms):
            bond_sums[bond.atom_a] += bond.order
            bond_sums[bond.atom_b] += bond.order
    count = 0
    for index, atom in enumerate(molecule.atoms):
        target = charged_typical_valence(atom.element, atom.formal_charge) or 0
        count += max(0, target - bond_sums[index])
    return count


def _effective_bond_order(first: UffParameter, second: UffParameter, order: float) -> float:
    """The order UFF uses for a bond; callers pass UffTypingReport.bond_order().

    Until 0.1.87 any two resonance-typed atoms got 1.5 here, including a
    non-ring bond between aromatic atoms (biphenyl), which Open Babel keeps at 1.
    """
    return float(order)


def _equilibrium_bond_length(first: UffParameter, second: UffParameter, order: float) -> float:
    """UFF r_ij with bond-order and electronegativity corrections, in Å."""

    radii = first.bond_radius + second.bond_radius
    bond_order_correction = -0.1332 * radii * math.log(order)
    electronegativity_correction = (
        first.bond_radius * second.bond_radius
        * (math.sqrt(first.electronegativity) - math.sqrt(second.electronegativity)) ** 2
        / (first.electronegativity * first.bond_radius + second.electronegativity * second.bond_radius)
    )
    return radii + bond_order_correction - electronegativity_correction


def bond_stretch_energy(
    molecule: Molecule, _typing: UffTypingReport | None = None,
) -> UffBondStretchReport:
    """Evaluate only UFF bond-stretch energy for a fully explicit structure.

    Other UFF terms (angles, torsions, inversion and non-bonded interactions)
    are intentionally excluded here.
    """

    typing = _typing or type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffBondStretchReport(None, 0, tuple(messages))

    energy = 0.0
    for bond_index, bond in enumerate(molecule.bonds):
        first_type = typing.types[bond.atom_a]
        second_type = typing.types[bond.atom_b]
        first = UFF_PARAMETERS[first_type]
        second = UFF_PARAMETERS[second_type]
        equilibrium = _equilibrium_bond_length(
            first, second, _effective_bond_order(first, second, typing.bond_order(bond_index, bond.order))
        )
        force_constant = 664.12 * first.effective_charge * second.effective_charge / equilibrium ** 3
        first_position = molecule.atoms[bond.atom_a].position
        second_position = molecule.atoms[bond.atom_b].position
        distance = math.dist(first_position, second_position)
        energy += 0.5 * force_constant * (distance - equilibrium) ** 2
    return UffBondStretchReport(energy, len(molecule.bonds), ())


def _angle_force_constant(
    outer_first: UffParameter, center: UffParameter, outer_second: UffParameter,
    first_bond_order: int, second_bond_order: int,
) -> float:
    """UFF angle constant for i-j-k, with j as the central atom."""

    first_length = _equilibrium_bond_length(
        outer_first, center, _effective_bond_order(outer_first, center, first_bond_order)
    )
    second_length = _equilibrium_bond_length(
        center, outer_second, _effective_bond_order(center, outer_second, second_bond_order)
    )
    theta0 = math.radians(center.ideal_angle)
    outer_distance_squared = (
        first_length ** 2 + second_length ** 2
        - 2.0 * first_length * second_length * math.cos(theta0)
    )
    outer_distance = math.sqrt(outer_distance_squared)
    numerator = (
        3.0 * first_length * second_length * math.sin(theta0) ** 2
        - outer_distance_squared * math.cos(theta0)
    )
    return 664.12 * outer_first.effective_charge * outer_second.effective_charge * numerator / outer_distance ** 5


def _uff_angle_energy(
    force_constant: float, observed: float, ideal: float, center_type: str, degree: int,
) -> float:
    """UFF's coordination-specific bend term."""

    coordination = _coordination(center_type)
    if coordination == "1" or degree == 1:
        return force_constant * (1.0 + math.cos(observed))
    # Open Babel picks the trigonal form from the type's coordination alone, so
    # two-coordinate sp2 centres (imine N_2, ester O_2, thiophene S_R) use it too.
    if coordination in {"2", "R"} and degree >= 2:
        short_angle_repulsion = math.exp(-20.0 * (observed - ideal + 0.25)) * _KJ_TO_KCAL
        return (
            force_constant / 9.0 * (1.0 - math.cos(3.0 * observed))
            + short_angle_repulsion
        )
    sine = math.sin(ideal)
    if abs(sine) < 1e-6:
        return force_constant * (observed - ideal) ** 2
    cosine0 = math.cos(ideal)
    coefficient2 = 1.0 / (4.0 * sine ** 2)
    coefficient1 = -4.0 * coefficient2 * cosine0
    coefficient0 = coefficient2 * (2.0 * cosine0 ** 2 + 1.0)
    return force_constant * (
        coefficient0
        + coefficient1 * math.cos(observed)
        + coefficient2 * math.cos(2.0 * observed)
    )


def _uff_angle_derivative(
    force_constant: float, observed: float, ideal: float, center_type: str, degree: int,
) -> float:
    """Derivative of the coordination-specific angle energy with respect to theta."""

    coordination = _coordination(center_type)
    if coordination == "1" or degree == 1:
        return -force_constant * math.sin(observed)
    # Open Babel picks the trigonal form from the type's coordination alone, so
    # two-coordinate sp2 centres (imine N_2, ester O_2, thiophene S_R) use it too.
    if coordination in {"2", "R"} and degree >= 2:
        short_angle_repulsion = math.exp(-20.0 * (observed - ideal + 0.25)) * _KJ_TO_KCAL
        return force_constant / 3.0 * math.sin(3.0 * observed) - 20.0 * short_angle_repulsion
    sine = math.sin(ideal)
    if abs(sine) < 1e-6:
        return 2.0 * force_constant * (observed - ideal)
    cosine0 = math.cos(ideal)
    coefficient2 = 1.0 / (4.0 * sine ** 2)
    coefficient1 = -4.0 * coefficient2 * cosine0
    return force_constant * (
        -coefficient1 * math.sin(observed)
        - 2.0 * coefficient2 * math.sin(2.0 * observed)
    )


def _build_neighbors(
    molecule: Molecule, typing: UffTypingReport | None = None,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    """(neighbour, bond order) per atom; UFF's effective orders when typing is given."""
    neighbors: list[list[tuple[int, float]]] = [[] for _atom in molecule.atoms]
    for index, bond in enumerate(molecule.bonds):
        order = typing.bond_order(index, bond.order) if typing is not None else bond.order
        neighbors[bond.atom_a].append((bond.atom_b, order))
        neighbors[bond.atom_b].append((bond.atom_a, order))
    return tuple(tuple(attached) for attached in neighbors)


def angle_bend_energy(
    molecule: Molecule, _typing: UffTypingReport | None = None,
    _neighbors: tuple[tuple[tuple[int, int], ...], ...] | None = None,
) -> UffAngleBendReport:
    """Evaluate only UFF angle-bend energy for a fully explicit structure."""

    typing = _typing or type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffAngleBendReport(None, 0, tuple(messages))

    neighbors = _build_neighbors(molecule, typing) if _neighbors is None else _neighbors

    energy = 0.0
    terms = 0
    for center_index, attached in enumerate(neighbors):
        center_type = typing.types[center_index]
        center_parameter = UFF_PARAMETERS[center_type]
        center_position = molecule.atoms[center_index].position
        for first_offset in range(len(attached)):
            first_index, first_order = attached[first_offset]
            first_parameter = UFF_PARAMETERS[typing.types[first_index]]
            first_vector = tuple(
                molecule.atoms[first_index].position[axis] - center_position[axis] for axis in range(3)
            )
            first_length = math.sqrt(sum(component ** 2 for component in first_vector))
            for second_index, second_order in attached[first_offset + 1:]:
                second_parameter = UFF_PARAMETERS[typing.types[second_index]]
                second_vector = tuple(
                    molecule.atoms[second_index].position[axis] - center_position[axis] for axis in range(3)
                )
                second_length = math.sqrt(sum(component ** 2 for component in second_vector))
                if first_length < 1e-10 or second_length < 1e-10:
                    return UffAngleBendReport(None, terms, ("UFF: overlapping atoms cannot form an angle",))
                cosine = sum(a * b for a, b in zip(first_vector, second_vector)) / (first_length * second_length)
                observed = math.acos(max(-1.0, min(1.0, cosine)))
                force_constant = _angle_force_constant(
                    first_parameter, center_parameter, second_parameter, first_order, second_order
                )
                energy += _uff_angle_energy(
                    force_constant, observed, math.radians(center_parameter.ideal_angle), center_type, len(attached)
                )
                terms += 1
    return UffAngleBendReport(energy, terms, ())


def _dihedral_angle(first, second, third, fourth) -> float | None:
    """Return the signed i-j-k-l torsion in radians, or None if degenerate."""

    bond_first = tuple(second[axis] - first[axis] for axis in range(3))
    bond_center = tuple(third[axis] - second[axis] for axis in range(3))
    bond_last = tuple(fourth[axis] - third[axis] for axis in range(3))
    cross_first = (
        bond_first[1] * bond_center[2] - bond_first[2] * bond_center[1],
        bond_first[2] * bond_center[0] - bond_first[0] * bond_center[2],
        bond_first[0] * bond_center[1] - bond_first[1] * bond_center[0],
    )
    cross_last = (
        bond_center[1] * bond_last[2] - bond_center[2] * bond_last[1],
        bond_center[2] * bond_last[0] - bond_center[0] * bond_last[2],
        bond_center[0] * bond_last[1] - bond_center[1] * bond_last[0],
    )
    first_norm = math.sqrt(sum(value * value for value in cross_first))
    last_norm = math.sqrt(sum(value * value for value in cross_last))
    center_norm = math.sqrt(sum(value * value for value in bond_center))
    if min(first_norm, last_norm, center_norm) < 1e-10:
        return None
    cosine = sum(a * b for a, b in zip(cross_first, cross_last)) / (first_norm * last_norm)
    cross_cross = (
        cross_first[1] * cross_last[2] - cross_first[2] * cross_last[1],
        cross_first[2] * cross_last[0] - cross_first[0] * cross_last[2],
        cross_first[0] * cross_last[1] - cross_first[1] * cross_last[0],
    )
    sine = sum(a * b for a, b in zip(cross_cross, bond_center)) / (first_norm * last_norm * center_norm)
    return math.atan2(sine, max(-1.0, min(1.0, cosine)))


def _chalcogen_sp3_torsion_barrier(parameter: UffParameter) -> float | None:
    """Open Babel's UFF torsion barrier override for a Group 6 sp3 atom.

    Transcribed from Open Babel 3.1.1's OBForceFieldUFF::SetupCalculations
    (forcefielduff.cpp): oxygen uses 2.0 kcal/mol, while S/Se/Te/Po use a
    distinct 6.8 kcal/mol -- not the same value, and not derivable from this
    project's type-name suffix (S_3+2's "+2" marks oxidation state, not
    hybridisation, so it must be matched by element prefix instead).
    """

    if parameter.name.startswith("O_"):
        return 2.0
    if parameter.name.startswith(("S_", "Se", "Te")):
        return 6.8
    return None


def _torsion_parameters(
    first: UffParameter, second: UffParameter, bond_order: float = 1.0,
) -> tuple[int, float, float] | None:
    """Return periodicity, cos(n*phi0), and UFF torsional barrier.

    Dispatch mirrors Open Babel 3.1.1's OBForceFieldUFF::SetupCalculations:
    the sp3-sp3, sp2-sp2 (including aromatic), and mixed sp2-sp3 cases are
    matched as exact hybridisation pairs, not by any single coordination
    character being present in either name -- an earlier version of this
    function used an "early union-membership" check that let the sp3-sp3
    branch also (incorrectly) swallow every sp2-sp3 pair, since both
    contain a "3".
    """

    first_coordination = _TORSION_COORDINATION[first.name]
    second_coordination = _TORSION_COORDINATION[second.name]
    if "1" in {first_coordination, second_coordination}:
        # A linear (sp) atom provides no stable reference plane for a
        # dihedral through it: rotating the rest of the molecule around its
        # own bond axis is a symmetry of that atom, not a torsion with a
        # barrier. Excluding it here (rather than only catching the
        # resulting degenerate-angle error later) matches how real UFF
        # implementations define torsions only between non-linear centers.
        return None
    # "_R" (resonant/aromatic) counts as sp2 for this dispatch, matching how
    # Open Babel's own per-atom hybridisation field treats both the same.
    first_is_sp2 = first_coordination in {"2", "R"}
    second_is_sp2 = second_coordination in {"2", "R"}
    first_is_sp3 = first_coordination == "3"
    second_is_sp3 = second_coordination == "3"
    if first_is_sp3 and second_is_sp3:
        periodicity = 3
        ideal = math.radians(60.0)
        first_barrier = first.torsion_barrier
        second_barrier = second.torsion_barrier
        chalcogen_first = _chalcogen_sp3_torsion_barrier(first)
        chalcogen_second = _chalcogen_sp3_torsion_barrier(second)
        if chalcogen_first is not None:
            periodicity, ideal, first_barrier = 2, math.radians(90.0), chalcogen_first
        if chalcogen_second is not None:
            periodicity, ideal, second_barrier = 2, math.radians(90.0), chalcogen_second
        return periodicity, math.cos(periodicity * ideal), 0.5 * math.sqrt(first_barrier * second_barrier)
    if first_is_sp2 and second_is_sp2:
        # Open Babel scales by the central bond's UFF order (1.5 aromatic,
        # 1.41 amide, 2 for a double bond), not by the atom types.
        barrier = 5.0 * math.sqrt(first.sp2_torsion_barrier * second.sp2_torsion_barrier)
        barrier *= 0.5 * (1.0 + 4.18 * math.log(max(bond_order, 1.0)))
        return 2, 1.0, barrier
    if (first_is_sp2 and second_is_sp3) or (first_is_sp3 and second_is_sp2):
        # The barrier itself (0.5 kcal/mol) is unchanged by the chalcogen
        # exception here: only the periodicity and phase angle change for
        # a Group 6 sp3 atom on either side, per Open Babel's source.
        periodicity, ideal = 6, 0.0
        sp3_parameter = first if first_is_sp3 else second
        if _chalcogen_sp3_torsion_barrier(sp3_parameter) is not None:
            periodicity, ideal = 2, 90.0
        return periodicity, math.cos(periodicity * math.radians(ideal)), 0.5
    return None


def _torsion_bonds(molecule: Molecule, typing: UffTypingReport):
    """(bond, parameters) for every bond that carries torsion terms."""
    for bond_index, bond in enumerate(molecule.bonds):
        if bond.order != 1 and typing.bond_order(bond_index, bond.order) == bond.order:
            # A genuine double/triple bond (not aromatic or amide).
            if not TORSIONS_ABOUT_MULTIPLE_BONDS:
                continue
        parameters = _torsion_parameters(
            UFF_PARAMETERS[typing.types[bond.atom_a]], UFF_PARAMETERS[typing.types[bond.atom_b]],
            typing.bond_order(bond_index, bond.order),
        )
        if parameters is not None:
            yield bond, parameters


# Open Babel defines torsions about double bonds too; kept switchable while
# that is verified against it at non-planar geometries.
TORSIONS_ABOUT_MULTIPLE_BONDS = True


def torsion_energy(
    molecule: Molecule, _typing: UffTypingReport | None = None, _check_hydrogens: bool = True,
) -> UffTorsionReport:
    """Evaluate UFF proper torsions around explicit, single central bonds."""

    typing = _typing or type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule) if _check_hydrogens else 0
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffTorsionReport(None, 0, tuple(messages))

    neighbors: list[list[tuple[int, int]]] = [[] for _ in molecule.atoms]
    for bond in molecule.bonds:
        neighbors[bond.atom_a].append((bond.atom_b, bond.order))
        neighbors[bond.atom_b].append((bond.atom_a, bond.order))
    energy = 0.0
    terms = 0
    for bond, parameters in _torsion_bonds(molecule, typing):
        first_center, second_center = bond.atom_a, bond.atom_b
        periodicity, cosine_ideal, barrier = parameters
        for first_outer, _first_order in neighbors[first_center]:
            if first_outer == second_center:
                continue
            for second_outer, _second_order in neighbors[second_center]:
                if second_outer == first_center:
                    continue
                angle = _dihedral_angle(
                    molecule.atoms[first_outer].position,
                    molecule.atoms[first_center].position,
                    molecule.atoms[second_center].position,
                    molecule.atoms[second_outer].position,
                )
                if angle is None:
                    return UffTorsionReport(None, terms, ("UFF: collinear atoms cannot form a torsion",))
                energy += barrier * (1.0 - cosine_ideal * math.cos(periodicity * angle))
                terms += 1
    return UffTorsionReport(energy, terms, ())


def _build_vdw_pairs(
    molecule: Molecule, typing: UffTypingReport,
) -> tuple[tuple[int, int, float, float], ...]:
    excluded: set[tuple[int, int]] = set()
    neighbors: list[set[int]] = [set() for _atom in molecule.atoms]
    for bond in molecule.bonds:
        first, second = sorted((bond.atom_a, bond.atom_b))
        excluded.add((first, second))
        neighbors[bond.atom_a].add(bond.atom_b)
        neighbors[bond.atom_b].add(bond.atom_a)
    for attached in neighbors:
        ordered = sorted(attached)
        for first_offset in range(len(ordered)):
            for second_offset in range(first_offset + 1, len(ordered)):
                excluded.add((ordered[first_offset], ordered[second_offset]))
    pairs = []
    for first in range(len(molecule.atoms)):
        first_parameter = UFF_PARAMETERS[typing.types[first]]
        for second in range(first + 1, len(molecule.atoms)):
            if (first, second) in excluded:
                continue
            second_parameter = UFF_PARAMETERS[typing.types[second]]
            pairs.append((
                first, second,
                math.sqrt(first_parameter.vdw_well_depth * second_parameter.vdw_well_depth),
                math.sqrt(first_parameter.vdw_distance * second_parameter.vdw_distance),
            ))
    return tuple(pairs)


def van_der_waals_energy(
    molecule: Molecule, _typing: UffTypingReport | None = None,
    _pairs: tuple[tuple[int, int, float, float], ...] | None = None,
) -> UffVanDerWaalsReport:
    """Evaluate UFF's non-bonded 12-6 term for non-1-2/non-1-3 pairs.

    The current prototype is for isolated molecules: it uses a 10 Å cutoff and
    has no periodic-cell or electrostatic treatment.
    """

    typing = _typing or type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffVanDerWaalsReport(None, 0, tuple(messages))

    energy = 0.0
    terms = 0
    cutoff_squared = 100.0
    pairs = _build_vdw_pairs(molecule, typing) if _pairs is None else _pairs
    positions = [atom.position for atom in molecule.atoms]
    for first, second, depth, distance in pairs:
        first_position = positions[first]
        second_position = positions[second]
        # A flat expression avoids the per-pair cost of building and
        # draining a 3-element generator through sum(); this loop runs once
        # per non-bonded pair on every line-search energy evaluation, so it
        # dominates minimize_partial_energy's wall time for larger molecules
        # (profiled at ~55% of total time for a 300-atom case).
        delta_x = first_position[0] - second_position[0]
        delta_y = first_position[1] - second_position[1]
        delta_z = first_position[2] - second_position[2]
        distance_squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        if distance_squared > cutoff_squared:
            continue
        distance_sixth = distance ** 6
        observed_sixth = max(distance_squared ** 3, 1e-6)
        energy += depth * (distance_sixth ** 2 / observed_sixth ** 2 - 2.0 * distance_sixth / observed_sixth)
        terms += 1
    return UffVanDerWaalsReport(energy, terms, ())


def _out_of_plane_angle(center, first, second, outer) -> float | None:
    """Angle of one neighbor from the plane through center and two others."""

    center_to_first = tuple(first[axis] - center[axis] for axis in range(3))
    center_to_second = tuple(second[axis] - center[axis] for axis in range(3))
    center_to_outer = tuple(outer[axis] - center[axis] for axis in range(3))
    normal = (
        center_to_first[1] * center_to_second[2] - center_to_first[2] * center_to_second[1],
        center_to_first[2] * center_to_second[0] - center_to_first[0] * center_to_second[2],
        center_to_first[0] * center_to_second[1] - center_to_first[1] * center_to_second[0],
    )
    normal_length = math.sqrt(sum(component * component for component in normal))
    outer_length = math.sqrt(sum(component * component for component in center_to_outer))
    if min(normal_length, outer_length) < 1e-10:
        return None
    sine = sum(a * b for a, b in zip(center_to_outer, normal)) / (normal_length * outer_length)
    return math.asin(max(-1.0, min(1.0, sine)))


# Group 15 inversion (Open Babel 3.1.1, forcefielduff.cpp): equilibrium
# out-of-plane angle per element, K = 22 kcal/mol split over three terms.
_GROUP15_INVERSION_ANGLE = {"P": 84.4339, "As": 86.9735, "Sb": 87.7047}
_CARBON_NITROGEN_OXYGEN_INVERSION = {"N_3", "N_R", "N_2", "O_R", "O_2", "C_R", "C_2"}


def _inversion_coefficients(center_type: str, element: str, neighbour_types) -> tuple[float, float, float, float] | None:
    """(K, c0, c1, c2) for E = K (c0 + c1 cos w + c2 cos 2w), per inversion term."""
    if center_type in _CARBON_NITROGEN_OXYGEN_INVERSION:
        barrier = 6.0
        if center_type in {"C_R", "C_2"} and any(t == "O_2" for t in neighbour_types):
            barrier = 50.0
        # 1 - cos w: minimum at a planar centre.
        return barrier / 3.0, 1.0, -1.0, 0.0
    if center_type in {"P_3+3", "As3+3", "Sb3+3"}:
        phi = math.radians(_GROUP15_INVERSION_ANGLE[element])
        c1 = -4.0 * math.cos(phi)
        c2 = 1.0
        c0 = -c1 * math.cos(phi) + c2 * math.cos(2.0 * phi)
        return 22.0 / 3.0, c0, c1, c2
    return None


def _inversion_terms(molecule: Molecule, types) -> list[tuple[int, int, int, int, float, float, float, float]]:
    """(center, plane a, plane b, outer, K, c0, c1, c2) for every three-coordinate centre."""
    neighbors: list[list[int]] = [[] for _ in molecule.atoms]
    for bond in molecule.bonds:
        neighbors[bond.atom_a].append(bond.atom_b)
        neighbors[bond.atom_b].append(bond.atom_a)
    terms = []
    for center_index, attached in enumerate(neighbors):
        if len(attached) != 3:
            continue
        coefficients = _inversion_coefficients(
            types[center_index], molecule.atoms[center_index].element, [types[i] for i in attached],
        )
        if coefficients is None:
            continue
        for outer_offset in range(3):
            plane = [attached[offset] for offset in range(3) if offset != outer_offset]
            terms.append((center_index, plane[0], plane[1], attached[outer_offset]) + coefficients)
    return terms


def out_of_plane_energy(
    molecule: Molecule, _typing: UffTypingReport | None = None, _check_hydrogens: bool = True,
) -> UffOutOfPlaneReport:
    """Evaluate UFF inversion energy for supported, three-coordinate centers."""

    typing = _typing or type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule) if _check_hydrogens else 0
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffOutOfPlaneReport(None, 0, tuple(messages))

    energy = 0.0
    terms = 0
    for center, first, second, outer, k, c0, c1, c2 in _inversion_terms(molecule, typing.types):
        angle = _out_of_plane_angle(
            molecule.atoms[center].position,
            molecule.atoms[first].position,
            molecule.atoms[second].position,
            molecule.atoms[outer].position,
        )
        if angle is None:
            return UffOutOfPlaneReport(None, terms, ("UFF: collinear neighbors cannot form an inversion plane",))
        energy += k * (c0 + c1 * math.cos(angle) + c2 * math.cos(2.0 * angle))
        terms += 1
    return UffOutOfPlaneReport(energy, terms, ())


def partial_energy(
    molecule: Molecule, _typing: UffTypingReport | None = None,
    _prepared: PreparedUffContext | None = None,
) -> UffPartialEnergyReport:
    """Sum the implemented UFF terms; this is not a complete UFF energy yet."""

    typing = _typing or type_molecule(molecule)
    reports = (
        bond_stretch_energy(molecule, typing),
        angle_bend_energy(molecule, typing, _prepared.neighbors if _prepared else None),
        torsion_energy(molecule, typing),
        out_of_plane_energy(molecule, typing),
        van_der_waals_energy(molecule, typing, _prepared.vdw_pairs if _prepared else None),
    )
    messages = tuple(message for report in reports for message in report.messages)
    if messages:
        return UffPartialEnergyReport(None, messages)
    return UffPartialEnergyReport(sum(report.energy_kcal_mol for report in reports), ())


def _residual_numeric_energy(
    molecule: Molecule, _typing: UffTypingReport | None = None,
) -> UffPartialEnergyReport:
    """Terms still differentiated numerically by the hybrid optimizer."""

    typing = _typing or type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffPartialEnergyReport(None, tuple(messages))
    reports = (
        torsion_energy(molecule, typing, False),
        out_of_plane_energy(molecule, typing, False),
    )
    messages = tuple(message for report in reports for message in report.messages)
    if messages:
        return UffPartialEnergyReport(None, messages)
    return UffPartialEnergyReport(sum(report.energy_kcal_mol for report in reports), ())


def numeric_gradient_atom_indices(
    molecule: Molecule, _typing: UffTypingReport | None = None,
) -> set[int]:
    """Atoms whose coordinates can affect the still-numeric torsion/inversion terms."""

    typing = _typing or type_molecule(molecule)
    if not typing.ready:
        return set(range(len(molecule.atoms)))
    neighbors: list[list[tuple[int, int]]] = [[] for _atom in molecule.atoms]
    for bond in molecule.bonds:
        neighbors[bond.atom_a].append((bond.atom_b, bond.order))
        neighbors[bond.atom_b].append((bond.atom_a, bond.order))
    active: set[int] = set()
    # Same term selection as torsion_energy / out_of_plane_energy, so every
    # atom a numeric term depends on is perturbed (including group 15 centres).
    for bond, _parameters in _torsion_bonds(molecule, typing):
        first_center, second_center = bond.atom_a, bond.atom_b
        first_outer = [index for index, _order in neighbors[first_center] if index != second_center]
        second_outer = [index for index, _order in neighbors[second_center] if index != first_center]
        if first_outer and second_outer:
            active.update((first_center, second_center))
            active.update(first_outer)
            active.update(second_outer)
    for center, first, second, outer, *_coefficients in _inversion_terms(molecule, typing.types):
        active.update((center, first, second, outer))
    return active


def _residual_terms(
    molecule: Molecule, _typing: UffTypingReport | None = None,
) -> tuple[
    list[tuple[int, int, int, int, int, float, float]],
    list[tuple[int, int, int, int, float, float, float, float]],
]:
    """Build coordinate-independent torsion and inversion terms for native evaluation."""

    typing = _typing or type_molecule(molecule)
    if not typing.ready:
        return [], []
    neighbors: list[list[tuple[int, int]]] = [[] for _atom in molecule.atoms]
    for bond in molecule.bonds:
        neighbors[bond.atom_a].append((bond.atom_b, bond.order))
        neighbors[bond.atom_b].append((bond.atom_a, bond.order))
    torsions = []
    for bond, parameters in _torsion_bonds(molecule, typing):
        first_center, second_center = bond.atom_a, bond.atom_b
        periodicity, cosine_ideal, barrier = parameters
        for first_outer, _first_order in neighbors[first_center]:
            if first_outer == second_center:
                continue
            for second_outer, _second_order in neighbors[second_center]:
                if second_outer == first_center:
                    continue
                torsions.append((
                    first_outer, first_center, second_center, second_outer,
                    periodicity, cosine_ideal, barrier,
                ))
    inversions = _inversion_terms(molecule, typing.types)
    return torsions, inversions


def _uff_topology_key(molecule: Molecule) -> tuple:
    return (
        tuple((atom.element, atom.formal_charge, atom.chirality) for atom in molecule.atoms),
        tuple((bond.atom_a, bond.atom_b, bond.order, bond.stereo) for bond in molecule.bonds),
    )


def prepare_uff_context(molecule: Molecule) -> PreparedUffContext:
    """Prepare atom typing and coordinate-independent terms once per topology."""

    typing = type_molecule(molecule)
    torsions, inversions = _residual_terms(molecule, typing)
    numeric_atoms = numeric_gradient_atom_indices(molecule, typing)
    neighbors = _build_neighbors(molecule, typing)
    vdw_pairs = _build_vdw_pairs(molecule, typing) if typing.ready else ()
    return PreparedUffContext(
        _uff_topology_key(molecule), typing, tuple(torsions), tuple(inversions),
        frozenset(numeric_atoms), neighbors, vdw_pairs,
    )


def _unverified_torsion_energy_gradient(
    molecule: Molecule,
) -> tuple[float | None, list[list[float]], tuple[str, ...]]:
    """Reserved experiment; intentionally unavailable until all coordinate derivatives agree."""

    raise NotImplementedError("torsion analytic gradient is not validated")

    typing = type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    gradient = [[0.0, 0.0, 0.0] for _atom in molecule.atoms]
    if messages:
        return None, gradient, tuple(messages)
    neighbors: list[list[tuple[int, int]]] = [[] for _atom in molecule.atoms]
    for bond in molecule.bonds:
        neighbors[bond.atom_a].append((bond.atom_b, bond.order))
        neighbors[bond.atom_b].append((bond.atom_a, bond.order))

    def cross(first, second):
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    energy = 0.0
    for bond, parameters in _torsion_bonds(molecule, typing):
        first_center, second_center = bond.atom_a, bond.atom_b
        periodicity, cosine_ideal, barrier = parameters
        for first_outer, _first_order in neighbors[first_center]:
            if first_outer == second_center:
                continue
            for second_outer, _second_order in neighbors[second_center]:
                if second_outer == first_center:
                    continue
                indices = (first_outer, first_center, second_center, second_outer)
                points = [molecule.atoms[index].position for index in indices]
                angle = _dihedral_angle(*points)
                if angle is None:
                    return None, gradient, ("UFF: collinear atoms cannot form a torsion gradient",)
                energy += barrier * (1.0 - cosine_ideal * math.cos(periodicity * angle))
                energy_derivative = barrier * cosine_ideal * periodicity * math.sin(periodicity * angle)
                first_bond = tuple(points[1][axis] - points[0][axis] for axis in range(3))
                center_bond = tuple(points[2][axis] - points[1][axis] for axis in range(3))
                last_bond = tuple(points[3][axis] - points[2][axis] for axis in range(3))
                first_normal = cross(first_bond, center_bond)
                last_normal = cross(center_bond, last_bond)
                first_norm_squared = sum(value * value for value in first_normal)
                last_norm_squared = sum(value * value for value in last_normal)
                center_norm_squared = sum(value * value for value in center_bond)
                center_norm = math.sqrt(center_norm_squared)
                if min(first_norm_squared, last_norm_squared, center_norm_squared) < 1e-20:
                    return None, gradient, ("UFF: degenerate torsion has no stable gradient",)
                first_angle_gradient = [
                    -center_norm * value / first_norm_squared for value in first_normal
                ]
                fourth_angle_gradient = [
                    -center_norm * value / last_norm_squared for value in last_normal
                ]
                first_projection = sum(a * b for a, b in zip(first_bond, center_bond)) / center_norm_squared
                last_projection = sum(a * b for a, b in zip(last_bond, center_bond)) / center_norm_squared
                second_angle_gradient = [
                    (first_projection - 1.0) * first_angle_gradient[axis]
                    - last_projection * fourth_angle_gradient[axis]
                    for axis in range(3)
                ]
                third_angle_gradient = [
                    (last_projection - 1.0) * fourth_angle_gradient[axis]
                    - first_projection * first_angle_gradient[axis]
                    for axis in range(3)
                ]
                for atom_index, angle_gradient in zip(
                    indices,
                    (first_angle_gradient, second_angle_gradient, third_angle_gradient, fourth_angle_gradient),
                ):
                    for axis in range(3):
                        gradient[atom_index][axis] += energy_derivative * angle_gradient[axis]
    return energy, gradient, ()


def _angle_energy_gradient(
    molecule: Molecule, _typing: UffTypingReport | None = None,
    _neighbors: tuple[tuple[tuple[int, int], ...], ...] | None = None,
) -> tuple[float | None, list[list[float]], tuple[str, ...]]:
    """Analytic Cartesian gradient for UFF angle-bend terms."""

    typing = _typing or type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    gradient = [[0.0, 0.0, 0.0] for _atom in molecule.atoms]
    if messages:
        return None, gradient, tuple(messages)
    neighbors = _build_neighbors(molecule, typing) if _neighbors is None else _neighbors
    energy = 0.0
    for center_index, attached in enumerate(neighbors):
        center_type = typing.types[center_index]
        center_parameter = UFF_PARAMETERS[center_type]
        center_position = molecule.atoms[center_index].position
        for first_offset in range(len(attached)):
            first_index, first_order = attached[first_offset]
            first_parameter = UFF_PARAMETERS[typing.types[first_index]]
            first_vector = [
                molecule.atoms[first_index].position[axis] - center_position[axis]
                for axis in range(3)
            ]
            first_length = math.sqrt(sum(component * component for component in first_vector))
            for second_index, second_order in attached[first_offset + 1:]:
                second_parameter = UFF_PARAMETERS[typing.types[second_index]]
                second_vector = [
                    molecule.atoms[second_index].position[axis] - center_position[axis]
                    for axis in range(3)
                ]
                second_length = math.sqrt(sum(component * component for component in second_vector))
                if min(first_length, second_length) < 1e-10:
                    return None, gradient, ("UFF: overlapping atoms cannot form an angle gradient",)
                cosine = sum(a * b for a, b in zip(first_vector, second_vector)) / (
                    first_length * second_length
                )
                observed = math.acos(max(-1.0, min(1.0, cosine)))
                sine_observed = math.sin(observed)
                # A linear (sp) center's energy is K*(1 + cos(theta)), whose
                # theta-derivative -K*sin(theta) is proportional to
                # sin(theta) itself. The ratio the chain rule needs below is
                # therefore exactly -K everywhere, including at this center's
                # own 180 degree equilibrium where sin(theta) underflows to
                # 0 — a removable singularity, not a genuinely degenerate
                # geometry, so it must not raise like a true collinear error.
                is_linear_center = _coordination(center_type) == "1" or len(attached) == 1
                if abs(sine_observed) < 1e-8 and not is_linear_center:
                    return None, gradient, ("UFF: collinear atoms have no stable angle gradient",)
                force_constant = _angle_force_constant(
                    first_parameter, center_parameter, second_parameter, first_order, second_order
                )
                ideal = math.radians(center_parameter.ideal_angle)
                energy += _uff_angle_energy(
                    force_constant, observed, ideal, center_type, len(attached)
                )
                energy_derivative = _uff_angle_derivative(
                    force_constant, observed, ideal, center_type, len(attached)
                )
                derivative_over_sine = (
                    -force_constant if is_linear_center else energy_derivative / sine_observed
                )
                first_gradient = [
                    derivative_over_sine * (
                        cosine * first_vector[axis] / first_length ** 2
                        - second_vector[axis] / (first_length * second_length)
                    )
                    for axis in range(3)
                ]
                second_gradient = [
                    derivative_over_sine * (
                        cosine * second_vector[axis] / second_length ** 2
                        - first_vector[axis] / (first_length * second_length)
                    )
                    for axis in range(3)
                ]
                for axis in range(3):
                    gradient[first_index][axis] += first_gradient[axis]
                    gradient[second_index][axis] += second_gradient[axis]
                    gradient[center_index][axis] -= first_gradient[axis] + second_gradient[axis]
    return energy, gradient, ()


def _bond_vdw_energy_gradient(
    molecule: Molecule, _typing: UffTypingReport | None = None,
    _pairs: tuple[tuple[int, int, float, float], ...] | None = None,
) -> tuple[float | None, list[list[float]], tuple[str, ...]]:
    """Analytic Cartesian gradient for bond stretch and van der Waals terms."""

    typing = _typing or type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    gradient = [[0.0, 0.0, 0.0] for _atom in molecule.atoms]
    if messages:
        return None, gradient, tuple(messages)

    energy = 0.0
    for bond_index, bond in enumerate(molecule.bonds):
        first_parameter = UFF_PARAMETERS[typing.types[bond.atom_a]]
        second_parameter = UFF_PARAMETERS[typing.types[bond.atom_b]]
        equilibrium = _equilibrium_bond_length(
            first_parameter,
            second_parameter,
            _effective_bond_order(first_parameter, second_parameter, typing.bond_order(bond_index, bond.order)),
        )
        force_constant = (
            664.12 * first_parameter.effective_charge * second_parameter.effective_charge
            / equilibrium ** 3
        )
        delta = [
            molecule.atoms[bond.atom_a].position[axis]
            - molecule.atoms[bond.atom_b].position[axis]
            for axis in range(3)
        ]
        distance = math.sqrt(sum(component * component for component in delta))
        if distance < 1e-10:
            return None, gradient, ("UFF: coincident bonded atoms have no defined gradient",)
        displacement = distance - equilibrium
        energy += 0.5 * force_constant * displacement ** 2
        derivative = force_constant * displacement / distance
        for axis in range(3):
            component = derivative * delta[axis]
            gradient[bond.atom_a][axis] += component
            gradient[bond.atom_b][axis] -= component

    vdw_pairs = _build_vdw_pairs(molecule, typing) if _pairs is None else _pairs

    native_result = _native_vdw_energy_gradient(
        [atom.position for atom in molecule.atoms], vdw_pairs
    )
    if native_result is not None:
        native_energy, native_gradient = native_result
        energy += native_energy
        for atom_index in range(len(gradient)):
            for axis in range(3):
                gradient[atom_index][axis] += native_gradient[atom_index][axis]
        return energy, gradient, ()

    for first, second, depth, equilibrium in vdw_pairs:
        delta = [
            molecule.atoms[first].position[axis] - molecule.atoms[second].position[axis]
            for axis in range(3)
        ]
        distance_squared = sum(component * component for component in delta)
        if distance_squared > 100.0:
            continue
        equilibrium_sixth = equilibrium ** 6
        observed_sixth = max(distance_squared ** 3, 1e-6)
        energy += depth * (
            equilibrium_sixth ** 2 / observed_sixth ** 2
            - 2.0 * equilibrium_sixth / observed_sixth
        )
        if distance_squared ** 3 <= 1e-6:
            continue
        distance = math.sqrt(distance_squared)
        derivative = 12.0 * depth * (
            equilibrium_sixth / distance ** 7
            - equilibrium_sixth ** 2 / distance ** 13
        ) / distance
        for axis in range(3):
            component = derivative * delta[axis]
            gradient[first][axis] += component
            gradient[second][axis] -= component
    return energy, gradient, ()


def _with_positions(molecule: Molecule, positions: list[list[float]]) -> Molecule:
    return Molecule(
        molecule.name,
        tuple(
            replace(atom, position=tuple(positions[index]))
            for index, atom in enumerate(molecule.atoms)
        ),
        molecule.bonds,
    )


def _distance_constraint_energy_gradient(
    molecule: Molecule,
    constraints: tuple[tuple[int, int, float, float], ...],
) -> tuple[float, list[list[float]]]:
    """Harmonic distance-restraint energy and analytic Cartesian gradient.

    Each term is ``(atom_a, atom_b, target_angstrom, force_constant)`` and uses
    ``0.5 * k * (r - target)^2`` in kcal/mol. These are user constraints, not
    additional UFF terms.
    """

    gradient = [[0.0, 0.0, 0.0] for _atom in molecule.atoms]
    energy = 0.0
    for first, second, target, force_constant in constraints:
        if first == second or not (0 <= first < len(molecule.atoms)) or not (0 <= second < len(molecule.atoms)):
            raise ValueError("distance constraint references invalid atoms")
        if target <= 0.0 or force_constant <= 0.0:
            raise ValueError("distance constraint target and force constant must be positive")
        delta = [
            molecule.atoms[first].position[axis] - molecule.atoms[second].position[axis]
            for axis in range(3)
        ]
        distance = math.sqrt(sum(component * component for component in delta))
        if distance < 1e-10:
            raise ValueError("distance constraint atoms cannot occupy the same position")
        displacement = distance - target
        energy += 0.5 * force_constant * displacement ** 2
        derivative = force_constant * displacement / distance
        for axis in range(3):
            component = derivative * delta[axis]
            gradient[first][axis] += component
            gradient[second][axis] -= component
    return energy, gradient


def _angle_constraint_energy_gradient(
    molecule: Molecule,
    constraints: tuple[tuple[int, int, int, float, float], ...],
) -> tuple[float, list[list[float]]]:
    """Harmonic angle-restraint energy and analytic gradient.

    Terms are ``(outer_a, center, outer_b, target_radians, force_constant)``.
    """

    gradient = [[0.0, 0.0, 0.0] for _atom in molecule.atoms]
    energy = 0.0
    for first, center, second, target, force_constant in constraints:
        indices = (first, center, second)
        if len(set(indices)) != 3 or any(index < 0 or index >= len(molecule.atoms) for index in indices):
            raise ValueError("angle constraint references invalid atoms")
        if not 0.0 < target < math.pi or force_constant <= 0.0:
            raise ValueError("angle constraint target and force constant must be positive and valid")
        first_vector = [
            molecule.atoms[first].position[axis] - molecule.atoms[center].position[axis]
            for axis in range(3)
        ]
        second_vector = [
            molecule.atoms[second].position[axis] - molecule.atoms[center].position[axis]
            for axis in range(3)
        ]
        first_length = math.sqrt(sum(value * value for value in first_vector))
        second_length = math.sqrt(sum(value * value for value in second_vector))
        if min(first_length, second_length) < 1e-10:
            raise ValueError("angle constraint contains coincident atoms")
        cosine = sum(a * b for a, b in zip(first_vector, second_vector)) / (
            first_length * second_length
        )
        cosine = max(-1.0, min(1.0, cosine))
        observed = math.acos(cosine)
        sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        if sine < 1e-8:
            raise ValueError("angle constraint is collinear and has no stable gradient")
        displacement = observed - target
        energy += 0.5 * force_constant * displacement ** 2
        derivative = force_constant * displacement
        first_gradient = [
            derivative / sine * (
                cosine * first_vector[axis] / first_length ** 2
                - second_vector[axis] / (first_length * second_length)
            )
            for axis in range(3)
        ]
        second_gradient = [
            derivative / sine * (
                cosine * second_vector[axis] / second_length ** 2
                - first_vector[axis] / (first_length * second_length)
            )
            for axis in range(3)
        ]
        for axis in range(3):
            gradient[first][axis] += first_gradient[axis]
            gradient[second][axis] += second_gradient[axis]
            gradient[center][axis] -= first_gradient[axis] + second_gradient[axis]
    return energy, gradient


def _wrapped_angle_difference(observed: float, target: float) -> float:
    return (observed - target + math.pi) % (2.0 * math.pi) - math.pi


def _dihedral_constraint_energy_gradient(
    molecule: Molecule,
    constraints: tuple[tuple[int, int, int, int, float, float], ...],
) -> tuple[float, list[list[float]]]:
    """Periodic four-atom dihedral restraints with a local central gradient.

    The wrapped harmonic energy is smooth except at the physically equivalent
    ±180° boundary. Only the four participating atoms are differentiated.
    """

    def constraint_energy(current: Molecule) -> float:
        total = 0.0
        for first, center_a, center_b, fourth, target, force_constant in constraints:
            indices = (first, center_a, center_b, fourth)
            if len(set(indices)) != 4 or any(
                index < 0 or index >= len(current.atoms) for index in indices
            ):
                raise ValueError("dihedral constraint references invalid atoms")
            if not -math.pi <= target <= math.pi or force_constant <= 0.0:
                raise ValueError("dihedral constraint target or force constant is invalid")
            observed = _dihedral_angle(*(current.atoms[index].position for index in indices))
            if observed is None:
                raise ValueError("dihedral constraint is degenerate")
            difference = _wrapped_angle_difference(observed, target)
            total += 0.5 * force_constant * difference ** 2
        return total

    energy = constraint_energy(molecule)
    gradient = [[0.0, 0.0, 0.0] for _atom in molecule.atoms]
    active = sorted({index for term in constraints for index in term[:4]})
    step = 1e-5
    positions = [list(atom.position) for atom in molecule.atoms]
    for atom_index in active:
        for axis in range(3):
            positions[atom_index][axis] += step
            higher = constraint_energy(_with_positions(molecule, positions))
            positions[atom_index][axis] -= 2.0 * step
            lower = constraint_energy(_with_positions(molecule, positions))
            positions[atom_index][axis] += step
            gradient[atom_index][axis] = (higher - lower) / (2.0 * step)
    return energy, gradient


def partial_objective_energy_gradient(
    molecule: Molecule,
    prepared: PreparedUffContext | None = None,
    fixed_atoms: set[int] | frozenset[int] = frozenset(),
    distance_constraints: tuple[tuple[int, int, float, float], ...] = (),
    angle_constraints: tuple[tuple[int, int, int, float, float], ...] = (),
    dihedral_constraints: tuple[tuple[int, int, int, int, float, float], ...] = (),
    finite_difference: float = 1e-4,
) -> tuple[float | None, tuple[tuple[float, float, float], ...], tuple[str, ...]]:
    """Evaluate the optimization objective and Cartesian gradient together.

    This is the calculation-facing boundary used by the optimizer. UFF and
    user-restraint energies are accumulated with the same gradient components,
    and frozen atoms are projected out before the result is returned.
    """

    context = prepared or prepare_uff_context(molecule)
    if context.topology_key != _uff_topology_key(molecule):
        raise ValueError("prepared UFF context belongs to a different molecular topology")
    fixed = frozenset(fixed_atoms)
    positions = [list(atom.position) for atom in molecule.atoms]
    analytic_energy, gradient, analytic_messages = _bond_vdw_energy_gradient(
        molecule, context.typing, context.vdw_pairs
    )
    angle_energy, angle_gradient, angle_messages = _angle_energy_gradient(
        molecule, context.typing, context.neighbors
    )
    messages = analytic_messages or angle_messages
    if messages:
        return None, tuple(tuple(vector) for vector in gradient), messages
    for atom_index in range(len(gradient)):
        for axis in range(3):
            gradient[atom_index][axis] += angle_gradient[atom_index][axis]

    distance_energy, distance_gradient = _distance_constraint_energy_gradient(
        molecule, distance_constraints
    )
    angle_constraint_energy, angle_constraint_gradient = _angle_constraint_energy_gradient(
        molecule, angle_constraints
    )
    dihedral_energy, dihedral_gradient = _dihedral_constraint_energy_gradient(
        molecule, dihedral_constraints
    )
    for addition in (distance_gradient, angle_constraint_gradient, dihedral_gradient):
        for atom_index in range(len(gradient)):
            for axis in range(3):
                gradient[atom_index][axis] += addition[atom_index][axis]

    native_residual = _native_residual_energy_gradient(
        positions, context.torsion_terms, context.inversion_terms,
        context.numeric_atoms - fixed, finite_difference,
    )
    if native_residual is not None:
        residual_energy, residual_gradient = native_residual
        for atom_index in context.numeric_atoms - fixed:
            for axis in range(3):
                gradient[atom_index][axis] += residual_gradient[atom_index][axis]
    else:
        residual = _residual_numeric_energy(molecule, context.typing)
        if not residual.ready:
            return None, tuple(tuple(vector) for vector in gradient), residual.messages
        residual_energy = residual.energy_kcal_mol
        for atom_index in context.numeric_atoms - fixed:
            for axis in range(3):
                positions[atom_index][axis] += finite_difference
                higher = _residual_numeric_energy(
                    _with_positions(molecule, positions), context.typing
                )
                positions[atom_index][axis] -= finite_difference
                if not higher.ready:
                    return None, tuple(tuple(vector) for vector in gradient), higher.messages
                gradient[atom_index][axis] += (
                    higher.energy_kcal_mol - residual_energy
                ) / finite_difference

    for atom_index in fixed:
        if not 0 <= atom_index < len(gradient):
            raise ValueError("fixed atom index is out of range")
        gradient[atom_index] = [0.0, 0.0, 0.0]
    total_energy = (
        analytic_energy + angle_energy + residual_energy + distance_energy
        + angle_constraint_energy + dihedral_energy
    )
    return total_energy, tuple(tuple(vector) for vector in gradient), ()


def _gradient_step_metrics(
    gradient: list[list[float]], fixed_atoms: set[int] | frozenset[int],
) -> tuple[float, float]:
    """Return max Cartesian component and max per-atom vector norm."""

    movable = [vector for index, vector in enumerate(gradient) if index not in fixed_atoms]
    if not movable:
        return 0.0, 0.0
    max_component = max(abs(component) for vector in movable for component in vector)
    max_atom_norm = max(math.sqrt(sum(component * component for component in vector)) for vector in movable)
    return max_component, max_atom_norm


_FIRE2_ESCAPE_MAX_ITERATIONS = 20


def _attempt_fire2_escape(
    molecule: Molecule, positions: list[list[float]], fixed_atoms: set[int] | frozenset[int],
    max_step: float,
    distance_constraints: tuple[tuple[int, int, float, float], ...],
    angle_constraints: tuple[tuple[int, int, int, float, float], ...],
    dihedral_constraints: tuple[tuple[int, int, int, int, float, float], ...],
    remaining_iterations: int, gradient_tolerance: float, energy_tolerance: float,
    cancelled: Callable[[], bool] | None,
) -> "UffMinimizationResult | None":
    """Recover from a stalled Cartesian line search (LBFGS/STEEPEST).

    UFF's van der Waals term is steep enough near a close contact that the
    line search can fail to accept *any* step even after repeated halving,
    while the gradient is still far above gradient_tolerance -- reproducing
    what this project's ROADMAP already anticipated as the case for falling
    back to FIRE2. FIRE2's damped-dynamics step never needs a
    sufficient-decrease test, so it can usually push through where the line
    search gives up. Only adopted when it actually makes progress; otherwise
    the caller reports the stall exactly as before.
    """
    if remaining_iterations <= 0:
        return None
    stalled_molecule = _with_positions(molecule, positions)
    escape = minimize_partial_energy(
        stalled_molecule, min(remaining_iterations, _FIRE2_ESCAPE_MAX_ITERATIONS), max_step,
        fixed_atoms, distance_constraints, angle_constraints, dihedral_constraints,
        gradient_tolerance=gradient_tolerance, cancelled=cancelled,
        energy_tolerance=energy_tolerance, algorithm="FIRE2",
    )
    if escape.ready and escape.iterations > 0:
        return escape
    return None


def minimize_partial_energy(
    molecule: Molecule, iterations: int, max_step: float, fixed_atoms: set[int] | None = None,
    distance_constraints: tuple[tuple[int, int, float, float], ...] = (),
    angle_constraints: tuple[tuple[int, int, int, float, float], ...] = (),
    dihedral_constraints: tuple[tuple[int, int, int, int, float, float], ...] = (),
    prepared: PreparedUffContext | None = None,
    gradient_tolerance: float = 1e-5,
    cancelled: Callable[[], bool] | None = None,
    energy_tolerance: float = 1e-6,
    algorithm: str = "STEEPEST",
    optimizer_state: UffOptimizerState | None = None,
) -> UffMinimizationResult:
    """Minimize the implemented UFF subset with selectable Cartesian optimizers.

    Bond stretch, angle bend and van der Waals use analytic gradients. Torsion
    and inversion use forward differences, evaluated as one parallel native
    batch when the Rust backend is active or by the reference Python fallback.
    """

    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0.0:
        raise ValueError("gradient tolerance must be positive and finite")
    if not math.isfinite(energy_tolerance) or energy_tolerance <= 0.0:
        raise ValueError("energy tolerance must be positive and finite")
    algorithm = algorithm.upper()
    if algorithm not in {"STEEPEST", "FIRE2", "LBFGS"}:
        raise ValueError("optimizer algorithm must be STEEPEST, FIRE2, or LBFGS")
    fixed = fixed_atoms or set()
    context = prepared or prepare_uff_context(molecule)
    if context.topology_key != _uff_topology_key(molecule):
        raise ValueError("prepared UFF context belongs to a different molecular topology")
    initial = partial_energy(molecule, context.typing, context)
    if not initial.ready:
        return UffMinimizationResult(None, None, None, 0, "not ready", initial.messages)
    positions = [list(atom.position) for atom in molecule.atoms]
    constraint_energy, _constraint_gradient = _distance_constraint_energy_gradient(
        molecule, distance_constraints
    )
    angle_constraint_energy, _angle_constraint_gradient = _angle_constraint_energy_gradient(
        molecule, angle_constraints
    )
    dihedral_constraint_energy, _dihedral_constraint_gradient = (
        _dihedral_constraint_energy_gradient(molecule, dihedral_constraints)
    )
    current_objective = (
        initial.energy_kcal_mol + constraint_energy + angle_constraint_energy
        + dihedral_constraint_energy
    )
    completed = 0
    finite_difference = 1e-4
    termination = "iteration limit"

    if algorithm == "LBFGS":
        atom_count = len(molecule.atoms)
        vector_size = atom_count * 3
        state_is_reusable = (
            optimizer_state is not None
            and optimizer_state.algorithm == "LBFGS"
            and len(optimizer_state.history_s) == len(optimizer_state.history_y)
            and all(len(vector) == vector_size for vector in optimizer_state.history_s)
            and all(len(vector) == vector_size for vector in optimizer_state.history_y)
        )
        history_s = list(optimizer_state.history_s[-7:]) if state_is_reusable else []
        history_y = list(optimizer_state.history_y[-7:]) if state_is_reusable else []
        current_molecule = _with_positions(molecule, positions)
        current_objective, evaluated_gradient, evaluation_messages = (
            partial_objective_energy_gradient(
                current_molecule, context, fixed, distance_constraints,
                angle_constraints, dihedral_constraints, finite_difference,
            )
        )
        if evaluation_messages:
            return UffMinimizationResult(
                None, initial.energy_kcal_mol, None, completed, "evaluation failed",
                evaluation_messages,
            )
        gradient = [component for vector in evaluated_gradient for component in vector]
        energy_tolerance_streak = 0
        for _ in range(iterations):
            if cancelled is not None and cancelled():
                termination = "cancelled"
                break
            gradient_vectors = [gradient[index:index + 3] for index in range(0, vector_size, 3)]
            max_gradient_component, _maximum_norm = _gradient_step_metrics(gradient_vectors, fixed)
            if max_gradient_component < gradient_tolerance:
                termination = "gradient tolerance"
                break

            work = gradient[:]
            coefficients = []
            valid_pairs = []
            for step_vector, gradient_delta in reversed(list(zip(history_s, history_y))):
                curvature = sum(s * y for s, y in zip(step_vector, gradient_delta))
                if curvature <= 1e-12:
                    continue
                coefficient = sum(s * q for s, q in zip(step_vector, work)) / curvature
                coefficients.append(coefficient)
                valid_pairs.append((step_vector, gradient_delta, curvature))
                work = [q - coefficient * y for q, y in zip(work, gradient_delta)]
            if valid_pairs:
                newest_s, newest_y, _curvature = valid_pairs[0]
                yy = sum(value * value for value in newest_y)
                scale = sum(s * y for s, y in zip(newest_s, newest_y)) / yy if yy > 0.0 else 1.0
            else:
                scale = 1.0
            direction = [scale * value for value in work]
            for coefficient, (step_vector, gradient_delta, curvature) in zip(
                reversed(coefficients), reversed(valid_pairs)
            ):
                beta = sum(y * r for y, r in zip(gradient_delta, direction)) / curvature
                direction = [r + s * (coefficient - beta) for r, s in zip(direction, step_vector)]
            direction = [-value for value in direction]
            directional_derivative = sum(g * d for g, d in zip(gradient, direction))
            if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
                direction = [-value for value in gradient]
                directional_derivative = -sum(value * value for value in gradient)

            direction_vectors = [direction[index:index + 3] for index in range(0, vector_size, 3)]
            for atom_index in fixed:
                if not 0 <= atom_index < atom_count:
                    raise ValueError("fixed atom index is out of range")
                direction_vectors[atom_index] = [0.0, 0.0, 0.0]
            direction = [component for vector in direction_vectors for component in vector]
            directional_derivative = sum(g * d for g, d in zip(gradient, direction))
            _component, maximum_direction = _gradient_step_metrics(direction_vectors, fixed)
            base_scale = min(1.0, max_step / maximum_direction) if maximum_direction > 0.0 else 1.0
            accepted = False
            line_scale = base_scale
            candidate_gradient = None
            candidate_objective = None
            candidate_positions = None
            for _line_iteration in range(8):
                if cancelled is not None and cancelled():
                    termination = "cancelled"
                    break
                trial = [position[:] for position in positions]
                for atom_index in range(atom_count):
                    if atom_index in fixed:
                        continue
                    for axis in range(3):
                        trial[atom_index][axis] += line_scale * direction[atom_index * 3 + axis]
                candidate = _with_positions(molecule, trial)
                candidate_objective, candidate_evaluated_gradient, candidate_messages = (
                    partial_objective_energy_gradient(
                        candidate, context, fixed, distance_constraints,
                        angle_constraints, dihedral_constraints, finite_difference,
                    )
                )
                if candidate_messages:
                    return UffMinimizationResult(
                        None, initial.energy_kcal_mol, None, completed, "evaluation failed",
                        candidate_messages,
                    )
                if candidate_objective <= current_objective + 1e-4 * line_scale * directional_derivative:
                    candidate_positions = trial
                    candidate_gradient = [
                        component for vector in candidate_evaluated_gradient for component in vector
                    ]
                    accepted = True
                    break
                line_scale *= 0.5
            if termination == "cancelled":
                break
            if not accepted:
                escape = _attempt_fire2_escape(
                    molecule, positions, fixed, max_step, distance_constraints,
                    angle_constraints, dihedral_constraints, iterations - completed,
                    gradient_tolerance, energy_tolerance, cancelled,
                )
                if escape is not None:
                    return UffMinimizationResult(
                        escape.molecule, initial.energy_kcal_mol, escape.final_energy_kcal_mol,
                        completed + escape.iterations, escape.termination, (),
                        escape.optimizer_state,
                    )
                termination = "line search stalled"
                break
            step_vector = tuple(
                candidate_positions[index // 3][index % 3] - positions[index // 3][index % 3]
                for index in range(vector_size)
            )
            gradient_delta = tuple(
                candidate_gradient[index] - gradient[index] for index in range(vector_size)
            )
            if sum(s * y for s, y in zip(step_vector, gradient_delta)) > 1e-12:
                history_s.append(step_vector)
                history_y.append(gradient_delta)
                history_s = history_s[-7:]
                history_y = history_y[-7:]
            energy_change = current_objective - candidate_objective
            positions = candidate_positions
            gradient = candidate_gradient
            current_objective = candidate_objective
            completed += 1
            if energy_change <= energy_tolerance:
                energy_tolerance_streak += 1
                if energy_tolerance_streak >= _ENERGY_TOLERANCE_STREAK:
                    termination = "energy tolerance"
                    break
            else:
                energy_tolerance_streak = 0

        final_molecule = _with_positions(molecule, positions)
        final_energy = partial_energy(final_molecule, context.typing, context)
        state = UffOptimizerState(
            "LBFGS", initialized=True,
            history_s=tuple(history_s), history_y=tuple(history_y),
        )
        return UffMinimizationResult(
            final_molecule, initial.energy_kcal_mol, final_energy.energy_kcal_mol,
            completed, termination, (), state,
        )

    if algorithm == "FIRE2":
        atom_count = len(molecule.atoms)
        state_is_reusable = (
            optimizer_state is not None
            and optimizer_state.algorithm == "FIRE2"
            and len(optimizer_state.velocity) == atom_count
            and math.isfinite(optimizer_state.time_step)
            and optimizer_state.time_step > 0.0
            and math.isfinite(optimizer_state.alpha)
            and 0.0 <= optimizer_state.alpha <= 1.0
        )
        if state_is_reusable:
            velocity = [list(vector) for vector in optimizer_state.velocity]
            time_step = optimizer_state.time_step
            alpha = optimizer_state.alpha
            positive_steps = optimizer_state.positive_steps
        else:
            velocity = [[0.0, 0.0, 0.0] for _atom in molecule.atoms]
            time_step = 0.05
            alpha = 0.25
            positive_steps = 0
        for atom_index in fixed:
            if not 0 <= atom_index < atom_count:
                raise ValueError("fixed atom index is out of range")
            velocity[atom_index] = [0.0, 0.0, 0.0]

        current_molecule = _with_positions(molecule, positions)
        current_objective, evaluated_gradient, evaluation_messages = (
            partial_objective_energy_gradient(
                current_molecule, context, fixed, distance_constraints,
                angle_constraints, dihedral_constraints, finite_difference,
            )
        )
        if evaluation_messages:
            return UffMinimizationResult(
                None, initial.energy_kcal_mol, None, completed, "evaluation failed",
                evaluation_messages,
            )
        gradient = [list(vector) for vector in evaluated_gradient]
        energy_tolerance_streak = 0
        for _ in range(iterations):
            if cancelled is not None and cancelled():
                termination = "cancelled"
                break
            max_gradient_component, _max_atom_gradient = _gradient_step_metrics(gradient, fixed)
            if max_gradient_component < gradient_tolerance:
                termination = "gradient tolerance"
                break

            power = -sum(
                gradient[index][axis] * velocity[index][axis]
                for index in range(atom_count) if index not in fixed
                for axis in range(3)
            )
            if power > 0.0:
                positive_steps += 1
                if positive_steps > 5:
                    time_step = min(time_step * 1.1, 0.5)
                    alpha *= 0.99
            elif power < 0.0:
                positive_steps = 0
                time_step *= 0.5
                for atom_index in range(atom_count):
                    if atom_index in fixed:
                        continue
                    for axis in range(3):
                        positions[atom_index][axis] -= 0.5 * time_step * velocity[atom_index][axis]
                        velocity[atom_index][axis] = 0.0
                alpha = 0.25

            for atom_index in range(atom_count):
                if atom_index in fixed:
                    velocity[atom_index] = [0.0, 0.0, 0.0]
                    continue
                for axis in range(3):
                    velocity[atom_index][axis] -= time_step * gradient[atom_index][axis]

            velocity_norm = math.sqrt(sum(
                component * component for index, vector in enumerate(velocity)
                if index not in fixed for component in vector
            ))
            gradient_norm = math.sqrt(sum(
                component * component for index, vector in enumerate(gradient)
                if index not in fixed for component in vector
            ))
            if velocity_norm > 0.0 and gradient_norm > 0.0:
                mixing = alpha * velocity_norm / gradient_norm
                for atom_index in range(atom_count):
                    if atom_index in fixed:
                        continue
                    for axis in range(3):
                        velocity[atom_index][axis] = (
                            (1.0 - alpha) * velocity[atom_index][axis]
                            - mixing * gradient[atom_index][axis]
                        )

            displacement = [
                [time_step * component for component in vector] for vector in velocity
            ]
            _component, max_displacement = _gradient_step_metrics(displacement, fixed)
            displacement_scale = min(1.0, max_step / max_displacement) if max_displacement > 0.0 else 1.0
            for atom_index in range(atom_count):
                if atom_index in fixed:
                    continue
                for axis in range(3):
                    positions[atom_index][axis] += displacement_scale * displacement[atom_index][axis]
            completed += 1

            candidate = _with_positions(molecule, positions)
            candidate_objective, evaluated_gradient, evaluation_messages = (
                partial_objective_energy_gradient(
                    candidate, context, fixed, distance_constraints,
                    angle_constraints, dihedral_constraints, finite_difference,
                )
            )
            if evaluation_messages or candidate_objective is None or not math.isfinite(candidate_objective):
                return UffMinimizationResult(
                    None, initial.energy_kcal_mol, None, completed, "evaluation failed",
                    evaluation_messages or ("UFF optimizer produced a non-finite objective",),
                )
            energy_change = current_objective - candidate_objective
            current_objective = candidate_objective
            gradient = [list(vector) for vector in evaluated_gradient]
            if 0.0 <= energy_change <= energy_tolerance:
                energy_tolerance_streak += 1
                if energy_tolerance_streak >= _ENERGY_TOLERANCE_STREAK:
                    termination = "energy tolerance"
                    break
            else:
                energy_tolerance_streak = 0

        final_molecule = _with_positions(molecule, positions)
        final_energy = partial_energy(final_molecule, context.typing, context)
        state = UffOptimizerState(
            "FIRE2", tuple(tuple(vector) for vector in velocity), time_step,
            alpha, positive_steps, True,
        )
        return UffMinimizationResult(
            final_molecule, initial.energy_kcal_mol, final_energy.energy_kcal_mol,
            completed, termination, (), state,
        )

    energy_tolerance_streak = 0
    for _ in range(iterations):
        if cancelled is not None and cancelled():
            termination = "cancelled"
            break
        current_molecule = _with_positions(molecule, positions)
        _objective_energy, evaluated_gradient, evaluation_messages = (
            partial_objective_energy_gradient(
                current_molecule, context, fixed, distance_constraints,
                angle_constraints, dihedral_constraints, finite_difference,
            )
        )
        if evaluation_messages:
            return UffMinimizationResult(
                None, initial.energy_kcal_mol, None, completed, "evaluation failed",
                evaluation_messages,
            )
        gradient = [list(vector) for vector in evaluated_gradient]
        max_gradient_component, max_atom_gradient = _gradient_step_metrics(gradient, fixed)
        if max_gradient_component < gradient_tolerance:
            termination = "gradient tolerance"
            break
        accepted = False
        step = max_step
        while step >= max_step / 128.0:
            if cancelled is not None and cancelled():
                termination = "cancelled"
                break
            candidate_positions = [position[:] for position in positions]
            for atom_index, vector in enumerate(gradient):
                if atom_index in fixed:
                    continue
                for axis in range(3):
                    candidate_positions[atom_index][axis] -= step * vector[axis] / max_atom_gradient
            candidate = _with_positions(molecule, candidate_positions)
            evaluation = partial_energy(candidate, context.typing, context)
            candidate_constraint_energy, _candidate_constraint_gradient = (
                _distance_constraint_energy_gradient(candidate, distance_constraints)
            )
            candidate_angle_energy, _candidate_angle_gradient = (
                _angle_constraint_energy_gradient(candidate, angle_constraints)
            )
            candidate_dihedral_energy, _candidate_dihedral_gradient = (
                _dihedral_constraint_energy_gradient(candidate, dihedral_constraints)
            )
            candidate_objective = (
                evaluation.energy_kcal_mol + candidate_constraint_energy + candidate_angle_energy
                + candidate_dihedral_energy
            )
            if evaluation.ready and candidate_objective < current_objective:
                energy_change = current_objective - candidate_objective
                positions = candidate_positions
                current_objective = candidate_objective
                accepted = True
                completed += 1
                if energy_change <= energy_tolerance:
                    energy_tolerance_streak += 1
                    if energy_tolerance_streak >= _ENERGY_TOLERANCE_STREAK:
                        termination = "energy tolerance"
                else:
                    energy_tolerance_streak = 0
                break
            step *= 0.5
        if termination == "cancelled":
            break
        if termination == "energy tolerance":
            break
        if not accepted:
            escape = _attempt_fire2_escape(
                molecule, positions, fixed, max_step, distance_constraints,
                angle_constraints, dihedral_constraints, iterations - completed,
                gradient_tolerance, energy_tolerance, cancelled,
            )
            if escape is not None:
                return UffMinimizationResult(
                    escape.molecule, initial.energy_kcal_mol, escape.final_energy_kcal_mol,
                    completed + escape.iterations, escape.termination, (),
                    escape.optimizer_state,
                )
            termination = "line search stalled"
            break
    final_molecule = _with_positions(molecule, positions)
    final_energy = partial_energy(final_molecule, context.typing, context)
    return UffMinimizationResult(
        final_molecule, initial.energy_kcal_mol, final_energy.energy_kcal_mol,
        completed, termination, (),
    )
