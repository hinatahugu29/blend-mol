"""First, dependency-free UFF atom typing support for Blend Mol.

This module deliberately contains no Blender API calls.  It is the boundary
between the editable molecular graph and a future energy/minimisation engine.
Only the well-defined organic subset is enabled at this stage; unknown or
ambiguous atoms are reported rather than silently receiving a generic type.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


_KJ_TO_KCAL = 1.0 / 4.184

try:
    from .molfile import Atom, Molecule
    from .native_backend import (
        residual_energy_gradient as _native_residual_energy_gradient,
        vdw_energy_gradient as _native_vdw_energy_gradient,
    )
except ImportError:  # Enables the small module to be tested outside Blender.
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
    "S_3+2": UffParameter("S_3+2", 1.064, 92.1, 4.035, 0.274, 2.703, 0.484, 1.25, 6.928),
    "Cl": UffParameter("Cl", 1.044, 180.0, 3.947, 0.227, 2.348, 0.0, 1.25, 4.946),
    "Br": UffParameter("Br", 1.192, 180.0, 4.189, 0.251, 2.519, 0.0, 0.70, 4.425),
    "I_": UffParameter("I_", 1.382, 180.0, 4.500, 0.339, 2.650, 0.0, 0.20, 3.762),
}


@dataclass(frozen=True)
class UffTypingReport:
    types: tuple[str | None, ...]
    messages: tuple[str, ...]

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
            atom_type = _type_from_max_bond("C", max_order)
        elif atom.element == "N" and -1 <= atom.formal_charge <= 1:
            atom_type = _type_from_max_bond("N", max_order)
        elif atom.element == "O" and -1 <= atom.formal_charge <= 1:
            atom_type = _type_from_max_bond("O", max_order)
        elif atom.element == "F" and atom.formal_charge == 0 and degree <= 1:
            atom_type = "F_"
        elif atom.element == "P" and atom.formal_charge == 0 and degree <= 3 and max_order == 1:
            atom_type = "P_3+3"
        elif atom.element == "S" and atom.formal_charge == 0 and degree <= 2 and max_order == 1:
            atom_type = "S_3+2"
        elif atom.element in {"Cl", "Br"} and atom.formal_charge == 0 and degree <= 1:
            atom_type = atom.element
        elif atom.element == "I" and atom.formal_charge == 0 and degree <= 1:
            atom_type = "I_"

        if atom_type not in UFF_PARAMETERS:
            types.append(None)
            messages.append(
                f"UFF: atom {index + 1} ({atom.element}, charge {atom.formal_charge:+d}) is not supported"
            )
        else:
            types.append(atom_type)
    # UFF has separate resonant types. This mirrors the conservative resonance
    # pass used in Avogadro: only an sp2-like C/N/O adjacent to another sp2 or
    # resonant atom is upgraded; carbonyl-like terminal oxygen is left alone.
    resonant_types = {"C_2": "C_R", "N_2": "N_R", "N_3": "N_R", "O_2": "O_R", "O_3": "O_R"}
    for index, atom in enumerate(molecule.atoms):
        current = types[index]
        if current not in resonant_types:
            continue
        if atom.element == "O" and len(neighbors[index]) == 1:
            continue
        if any(types[neighbor] and types[neighbor].endswith(("_2", "_R")) for neighbor, _order in neighbors[index]):
            types[index] = resonant_types[current]
    return UffTypingReport(tuple(types), tuple(messages))


def _implicit_hydrogen_count(molecule: Molecule) -> int:
    """Return virtual hydrogens that would make a bond-only UFF energy partial."""

    valences = {"H": 1, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
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


def _effective_bond_order(first: UffParameter, second: UffParameter, stored_order: int) -> float:
    """UFF uses bond order 1.5 between two resonance-typed atoms."""

    if first.name.endswith("_R") and second.name.endswith("_R"):
        return 1.5
    return float(stored_order)


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
    for bond in molecule.bonds:
        first_type = typing.types[bond.atom_a]
        second_type = typing.types[bond.atom_b]
        first = UFF_PARAMETERS[first_type]
        second = UFF_PARAMETERS[second_type]
        equilibrium = _equilibrium_bond_length(
            first, second, _effective_bond_order(first, second, bond.order)
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

    coordination = center_type[-1]
    if coordination == "1" or degree == 1:
        return force_constant * (1.0 + math.cos(observed))
    if coordination in {"2", "R"} and degree == 3:
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

    coordination = center_type[-1]
    if coordination == "1" or degree == 1:
        return -force_constant * math.sin(observed)
    if coordination in {"2", "R"} and degree == 3:
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


def _build_neighbors(molecule: Molecule) -> tuple[tuple[tuple[int, int], ...], ...]:
    neighbors: list[list[tuple[int, int]]] = [[] for _atom in molecule.atoms]
    for bond in molecule.bonds:
        neighbors[bond.atom_a].append((bond.atom_b, bond.order))
        neighbors[bond.atom_b].append((bond.atom_a, bond.order))
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

    neighbors = _build_neighbors(molecule) if _neighbors is None else _neighbors

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


def _torsion_parameters(first: UffParameter, second: UffParameter) -> tuple[int, float, float] | None:
    """Return periodicity, cos(n*phi0), and UFF torsional barrier."""

    first_coordination = first.name[-1]
    second_coordination = second.name[-1]
    if "3" in {first_coordination, second_coordination}:
        periodicity = 3
        ideal = math.radians(60.0)
        first_barrier = first.torsion_barrier
        second_barrier = second.torsion_barrier
        if first.name.startswith("O_"):
            periodicity, ideal, first_barrier = 2, math.radians(90.0), 2.0
        if second.name.startswith("O_"):
            periodicity, ideal, second_barrier = 2, math.radians(90.0), 2.0
        return periodicity, math.cos(periodicity * ideal), 0.5 * math.sqrt(first_barrier * second_barrier)
    if first_coordination == "R" and second_coordination == "R":
        aromatic_order = 1.5
        barrier = 5.0 * math.sqrt(first.sp2_torsion_barrier * second.sp2_torsion_barrier)
        barrier *= 0.5 * (1.0 + 4.18 * math.log(aromatic_order))
        return 2, 1.0, barrier
    if {first_coordination, second_coordination} == {"2", "3"}:
        return 6, 1.0, 0.5
    return None


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
    for bond in molecule.bonds:
        if bond.order != 1:
            continue
        first_center, second_center = bond.atom_a, bond.atom_b
        parameters = _torsion_parameters(
            UFF_PARAMETERS[typing.types[first_center]], UFF_PARAMETERS[typing.types[second_center]]
        )
        if parameters is None:
            continue
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
    for first, second, depth, distance in pairs:
        first_position = molecule.atoms[first].position
        second_position = molecule.atoms[second].position
        distance_squared = sum(
            (first_position[axis] - second_position[axis]) ** 2 for axis in range(3)
        )
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

    neighbors: list[list[int]] = [[] for _ in molecule.atoms]
    for bond in molecule.bonds:
        neighbors[bond.atom_a].append(bond.atom_b)
        neighbors[bond.atom_b].append(bond.atom_a)
    energy = 0.0
    terms = 0
    for center_index, attached in enumerate(neighbors):
        if len(attached) != 3:
            continue
        center_type = typing.types[center_index]
        if center_type in {"N_3", "N_R", "N_2", "O_R", "O_2", "C_R", "C_2"}:
            barrier = 6.0
            if center_type in {"C_R", "C_2"} and any(
                typing.types[index] == "O_2" for index in attached
            ):
                barrier = 50.0
        else:
            continue
        for outer_offset in range(3):
            plane = [attached[offset] for offset in range(3) if offset != outer_offset]
            angle = _out_of_plane_angle(
                molecule.atoms[center_index].position,
                molecule.atoms[plane[0]].position,
                molecule.atoms[plane[1]].position,
                molecule.atoms[attached[outer_offset]].position,
            )
            if angle is None:
                return UffOutOfPlaneReport(None, terms, ("UFF: collinear neighbors cannot form an inversion plane",))
            energy += barrier / 3.0 * (1.0 - math.cos(angle))
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
    for bond in molecule.bonds:
        if bond.order != 1:
            continue
        first_center, second_center = bond.atom_a, bond.atom_b
        if _torsion_parameters(
            UFF_PARAMETERS[typing.types[first_center]], UFF_PARAMETERS[typing.types[second_center]]
        ) is None:
            continue
        first_outer = [index for index, _order in neighbors[first_center] if index != second_center]
        second_outer = [index for index, _order in neighbors[second_center] if index != first_center]
        if first_outer and second_outer:
            active.update((first_center, second_center))
            active.update(first_outer)
            active.update(second_outer)
    inversion_types = {"N_3", "N_R", "N_2", "O_R", "O_2", "C_R", "C_2"}
    for center_index, attached in enumerate(neighbors):
        if len(attached) == 3 and typing.types[center_index] in inversion_types:
            active.add(center_index)
            active.update(index for index, _order in attached)
    return active


def _residual_terms(
    molecule: Molecule, _typing: UffTypingReport | None = None,
) -> tuple[
    list[tuple[int, int, int, int, int, float, float]],
    list[tuple[int, int, int, int, float]],
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
    for bond in molecule.bonds:
        if bond.order != 1:
            continue
        first_center, second_center = bond.atom_a, bond.atom_b
        parameters = _torsion_parameters(
            UFF_PARAMETERS[typing.types[first_center]], UFF_PARAMETERS[typing.types[second_center]]
        )
        if parameters is None:
            continue
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
    inversion_types = {"N_3", "N_R", "N_2", "O_R", "O_2", "C_R", "C_2"}
    inversions = []
    for center_index, attached in enumerate(neighbors):
        if len(attached) == 3 and typing.types[center_index] in inversion_types:
            barrier = 6.0
            if typing.types[center_index] in {"C_R", "C_2"} and any(
                typing.types[index] == "O_2" for index, _order in attached
            ):
                barrier = 50.0
            attached_indices = [index for index, _order in attached]
            for outer_offset in range(3):
                plane = [attached_indices[offset] for offset in range(3) if offset != outer_offset]
                inversions.append((
                    center_index, plane[0], plane[1], attached_indices[outer_offset], barrier / 3.0,
                ))
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
    neighbors = _build_neighbors(molecule)
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
    for bond in molecule.bonds:
        if bond.order != 1:
            continue
        first_center, second_center = bond.atom_a, bond.atom_b
        parameters = _torsion_parameters(
            UFF_PARAMETERS[typing.types[first_center]], UFF_PARAMETERS[typing.types[second_center]]
        )
        if parameters is None:
            continue
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
    neighbors = _build_neighbors(molecule) if _neighbors is None else _neighbors
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
                if abs(sine_observed) < 1e-8:
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
                first_gradient = [
                    energy_derivative / sine_observed * (
                        cosine * first_vector[axis] / first_length ** 2
                        - second_vector[axis] / (first_length * second_length)
                    )
                    for axis in range(3)
                ]
                second_gradient = [
                    energy_derivative / sine_observed * (
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
    for bond in molecule.bonds:
        first_parameter = UFF_PARAMETERS[typing.types[bond.atom_a]]
        second_parameter = UFF_PARAMETERS[typing.types[bond.atom_b]]
        equilibrium = _equilibrium_bond_length(
            first_parameter,
            second_parameter,
            _effective_bond_order(first_parameter, second_parameter, bond.order),
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
            Atom(atom.element, tuple(positions[index]), atom.formal_charge, atom.chirality)
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
    """Minimize the implemented UFF subset with steepest descent or FIRE2.

    Bond stretch, angle bend and van der Waals use analytic gradients. Torsion
    and inversion use forward differences, evaluated as one parallel native
    batch when the Rust backend is active or by the reference Python fallback.
    """

    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0.0:
        raise ValueError("gradient tolerance must be positive and finite")
    if not math.isfinite(energy_tolerance) or energy_tolerance <= 0.0:
        raise ValueError("energy tolerance must be positive and finite")
    algorithm = algorithm.upper()
    if algorithm not in {"STEEPEST", "FIRE2"}:
        raise ValueError("optimizer algorithm must be STEEPEST or FIRE2")
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
                termination = "energy tolerance"
                break

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
                    termination = "energy tolerance"
                break
            step *= 0.5
        if termination == "cancelled":
            break
        if termination == "energy tolerance":
            break
        if not accepted:
            termination = "line search stalled"
            break
    final_molecule = _with_positions(molecule, positions)
    final_energy = partial_energy(final_molecule, context.typing, context)
    return UffMinimizationResult(
        final_molecule, initial.energy_kcal_mol, final_energy.energy_kcal_mol,
        completed, termination, (),
    )
