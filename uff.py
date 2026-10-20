"""First, dependency-free UFF atom typing support for Blend Mol.

This module deliberately contains no Blender API calls.  It is the boundary
between the editable molecular graph and a future energy/minimisation engine.
Only the well-defined organic subset is enabled at this stage; unknown or
ambiguous atoms are reported rather than silently receiving a generic type.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

try:
    from .molfile import Molecule
except ImportError:  # Enables the small module to be tested outside Blender.
    from molfile import Molecule


@dataclass(frozen=True)
class UffParameter:
    """UFF per-atom parameters in the published UFF ordering and units."""

    name: str
    bond_radius: float
    ideal_angle: float
    vdw_distance: float
    vdw_well_depth: float
    effective_charge: float
    electronegativity: float


# Organic subset transcribed from the UFF parameter set.  The remaining UFF
# columns are added only when their associated energy terms are implemented.
UFF_PARAMETERS = {
    "H_": UffParameter("H_", 0.354, 180.0, 2.886, 0.044, 0.712, 4.528),
    "H_b": UffParameter("H_b", 0.460, 83.5, 2.886, 0.044, 0.712, 4.528),
    "C_3": UffParameter("C_3", 0.757, 109.47, 3.851, 0.105, 1.912, 5.343),
    "C_2": UffParameter("C_2", 0.732, 120.0, 3.851, 0.105, 1.912, 5.343),
    "C_1": UffParameter("C_1", 0.706, 180.0, 3.851, 0.105, 1.912, 5.343),
    "C_R": UffParameter("C_R", 0.729, 120.0, 3.851, 0.105, 1.912, 5.343),
    "N_3": UffParameter("N_3", 0.700, 106.7, 3.660, 0.069, 2.544, 6.899),
    "N_2": UffParameter("N_2", 0.685, 111.2, 3.660, 0.069, 2.544, 6.899),
    "N_1": UffParameter("N_1", 0.656, 180.0, 3.660, 0.069, 2.544, 6.899),
    "N_R": UffParameter("N_R", 0.699, 120.0, 3.660, 0.069, 2.544, 6.899),
    "O_3": UffParameter("O_3", 0.658, 104.51, 3.500, 0.060, 2.300, 8.741),
    "O_2": UffParameter("O_2", 0.634, 120.0, 3.500, 0.060, 2.300, 8.741),
    "O_1": UffParameter("O_1", 0.639, 180.0, 3.500, 0.060, 2.300, 8.741),
    "O_R": UffParameter("O_R", 0.680, 110.0, 3.500, 0.060, 2.300, 8.741),
    "F_": UffParameter("F_", 0.668, 180.0, 3.364, 0.050, 1.735, 10.874),
}


@dataclass(frozen=True)
class UffTypingReport:
    types: tuple[str | None, ...]
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.messages and all(atom_type is not None for atom_type in self.types)


@dataclass(frozen=True)
class UffBondStretchReport:
    """Partial UFF energy from all explicit bonds, in kcal mol⁻¹."""

    energy_kcal_mol: float | None
    terms: int
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.energy_kcal_mol is not None and not self.messages


def _type_from_max_bond(element: str, max_order: int) -> str:
    if max_order >= 3:
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

    neighbors: list[list[int]] = [[] for _ in molecule.atoms]
    messages: list[str] = []
    for bond in molecule.bonds:
        if not 0 <= bond.atom_a < len(molecule.atoms) or not 0 <= bond.atom_b < len(molecule.atoms):
            messages.append("UFF: a bond references an atom that does not exist")
            continue
        neighbors[bond.atom_a].append(bond.order)
        neighbors[bond.atom_b].append(bond.order)

    types: list[str | None] = []
    for index, atom in enumerate(molecule.atoms):
        bond_orders = neighbors[index]
        degree = len(bond_orders)
        max_order = max(bond_orders, default=1)
        atom_type: str | None = None
        if atom.element == "H":
            if degree <= 1:
                atom_type = "H_"
            elif degree == 2:
                atom_type = "H_b"
        elif atom.element == "C" and atom.formal_charge == 0:
            atom_type = _type_from_max_bond("C", max_order)
        elif atom.element == "N" and -1 <= atom.formal_charge <= 1:
            atom_type = _type_from_max_bond("N", max_order)
        elif atom.element == "O" and -1 <= atom.formal_charge <= 1:
            atom_type = _type_from_max_bond("O", max_order)
        elif atom.element == "F" and atom.formal_charge == 0 and degree <= 1:
            atom_type = "F_"

        if atom_type not in UFF_PARAMETERS:
            types.append(None)
            messages.append(
                f"UFF: atom {index + 1} ({atom.element}, charge {atom.formal_charge:+d}) is not supported"
            )
        else:
            types.append(atom_type)
    return UffTypingReport(tuple(types), tuple(messages))


def _implicit_hydrogen_count(molecule: Molecule) -> int:
    """Return virtual hydrogens that would make a bond-only UFF energy partial."""

    valences = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}
    bond_sums = [0] * len(molecule.atoms)
    for bond in molecule.bonds:
        if 0 <= bond.atom_a < len(molecule.atoms) and 0 <= bond.atom_b < len(molecule.atoms):
            bond_sums[bond.atom_a] += bond.order
            bond_sums[bond.atom_b] += bond.order
    count = 0
    for index, atom in enumerate(molecule.atoms):
        target = valences.get(atom.element, 0)
        if atom.element == "N" and atom.formal_charge > 0:
            target = 4
        if atom.element == "O" and atom.formal_charge < 0:
            target = 1
        count += max(0, target - bond_sums[index])
    return count


def _equilibrium_bond_length(first: UffParameter, second: UffParameter, order: int) -> float:
    """UFF r_ij with bond-order and electronegativity corrections, in Å."""

    radii = first.bond_radius + second.bond_radius
    bond_order_correction = -0.1332 * radii * math.log(order)
    electronegativity_correction = (
        first.bond_radius * second.bond_radius
        * (math.sqrt(first.electronegativity) - math.sqrt(second.electronegativity)) ** 2
        / (first.electronegativity * first.bond_radius + second.electronegativity * second.bond_radius)
    )
    return radii + bond_order_correction + electronegativity_correction


def bond_stretch_energy(molecule: Molecule) -> UffBondStretchReport:
    """Evaluate only UFF bond-stretch energy for a fully explicit structure.

    UFF's harmonic term is 1/2 k(r-r0)^2.  Other UFF terms (angles, torsions,
    inversion and non-bonded interactions) are intentionally excluded here.
    """

    typing = type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffBondStretchReport(None, 0, tuple(messages))

    energy = 0.0
    for bond in molecule.bonds:
        first_type = typing.types[bond.atom_a]
        second_type = typing.types[bond.atom_b]
        first = UFF_PARAMETERS[first_type]
        second = UFF_PARAMETERS[second_type]
        equilibrium = _equilibrium_bond_length(first, second, bond.order)
        force_constant = 664.12 * first.effective_charge * second.effective_charge / equilibrium ** 3
        first_position = molecule.atoms[bond.atom_a].position
        second_position = molecule.atoms[bond.atom_b].position
        distance = math.dist(first_position, second_position)
        energy += 0.5 * force_constant * (distance - equilibrium) ** 2
    return UffBondStretchReport(energy, len(molecule.bonds), ())
