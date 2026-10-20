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
    from .molfile import Atom, Molecule
except ImportError:  # Enables the small module to be tested outside Blender.
    from molfile import Atom, Molecule


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
class UffMinimizationResult:
    molecule: Molecule | None
    initial_energy_kcal_mol: float | None
    final_energy_kcal_mol: float | None
    iterations: int
    termination: str
    messages: tuple[str, ...]

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
    return radii + bond_order_correction - electronegativity_correction


def bond_stretch_energy(molecule: Molecule) -> UffBondStretchReport:
    """Evaluate only UFF bond-stretch energy for a fully explicit structure.

    Other UFF terms (angles, torsions, inversion and non-bonded interactions)
    are intentionally excluded here.
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
        energy += force_constant * (distance - equilibrium) ** 2
    return UffBondStretchReport(energy, len(molecule.bonds), ())


def _angle_force_constant(
    outer_first: UffParameter, center: UffParameter, outer_second: UffParameter,
    first_bond_order: int, second_bond_order: int,
) -> float:
    """UFF angle constant for i-j-k, with j as the central atom."""

    first_length = _equilibrium_bond_length(outer_first, center, first_bond_order)
    second_length = _equilibrium_bond_length(center, outer_second, second_bond_order)
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
        return (
            force_constant / 9.0 * (1.0 - math.cos(3.0 * observed))
            + math.exp(-20.0 * (observed - ideal + 0.25))
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


def angle_bend_energy(molecule: Molecule) -> UffAngleBendReport:
    """Evaluate only UFF angle-bend energy for a fully explicit structure."""

    typing = type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffAngleBendReport(None, 0, tuple(messages))

    neighbors: list[list[tuple[int, int]]] = [[] for _ in molecule.atoms]
    for bond in molecule.bonds:
        neighbors[bond.atom_a].append((bond.atom_b, bond.order))
        neighbors[bond.atom_b].append((bond.atom_a, bond.order))

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


def torsion_energy(molecule: Molecule) -> UffTorsionReport:
    """Evaluate UFF proper torsions around explicit, single central bonds."""

    typing = type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
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


def van_der_waals_energy(molecule: Molecule) -> UffVanDerWaalsReport:
    """Evaluate UFF's non-bonded 12-6 term for non-1-2/non-1-3 pairs.

    The current prototype is for isolated molecules: it uses a 10 Å cutoff and
    has no periodic-cell or electrostatic treatment.
    """

    typing = type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
    if missing_hydrogens:
        messages.append(f"UFF: {missing_hydrogens} implicit hydrogen(s) must be explicit for energy evaluation")
    if messages:
        return UffVanDerWaalsReport(None, 0, tuple(messages))

    excluded: set[tuple[int, int]] = set()
    neighbors: list[set[int]] = [set() for _ in molecule.atoms]
    for bond in molecule.bonds:
        first, second = sorted((bond.atom_a, bond.atom_b))
        excluded.add((first, second))
        neighbors[bond.atom_a].add(bond.atom_b)
        neighbors[bond.atom_b].add(bond.atom_a)
    for center, attached in enumerate(neighbors):
        for first in attached:
            for second in attached:
                if first < second:
                    excluded.add((first, second))

    energy = 0.0
    terms = 0
    cutoff_squared = 100.0
    for first in range(len(molecule.atoms)):
        first_parameter = UFF_PARAMETERS[typing.types[first]]
        for second in range(first + 1, len(molecule.atoms)):
            if (first, second) in excluded:
                continue
            first_position = molecule.atoms[first].position
            second_position = molecule.atoms[second].position
            distance_squared = sum(
                (first_position[axis] - second_position[axis]) ** 2 for axis in range(3)
            )
            if distance_squared > cutoff_squared:
                continue
            second_parameter = UFF_PARAMETERS[typing.types[second]]
            depth = math.sqrt(first_parameter.vdw_well_depth * second_parameter.vdw_well_depth)
            distance = math.sqrt(first_parameter.vdw_distance * second_parameter.vdw_distance)
            distance_sixth = distance ** 6
            observed_sixth = max(distance_squared ** 3, 1e-6)
            energy += depth * (distance_sixth ** 2 / observed_sixth ** 2 - 2.0 * distance_sixth / observed_sixth)
            terms += 1
    return UffVanDerWaalsReport(energy, terms, ())


def _out_of_plane_angle(center, first, second, third) -> float | None:
    """Angle of the central atom out of the plane formed by three neighbors."""

    first_to_center = tuple(center[axis] - first[axis] for axis in range(3))
    first_to_second = tuple(second[axis] - first[axis] for axis in range(3))
    first_to_third = tuple(third[axis] - first[axis] for axis in range(3))
    normal = (
        first_to_second[1] * first_to_third[2] - first_to_second[2] * first_to_third[1],
        first_to_second[2] * first_to_third[0] - first_to_second[0] * first_to_third[2],
        first_to_second[0] * first_to_third[1] - first_to_second[1] * first_to_third[0],
    )
    normal_length = math.sqrt(sum(component * component for component in normal))
    center_length = math.sqrt(sum(component * component for component in first_to_center))
    if min(normal_length, center_length) < 1e-10:
        return None
    sine = sum(a * b for a, b in zip(first_to_center, normal)) / (normal_length * center_length)
    return math.asin(max(-1.0, min(1.0, sine)))


def out_of_plane_energy(molecule: Molecule) -> UffOutOfPlaneReport:
    """Evaluate UFF inversion energy for supported, three-coordinate centers."""

    typing = type_molecule(molecule)
    messages = list(typing.messages)
    missing_hydrogens = _implicit_hydrogen_count(molecule)
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
        if center_type in {"N_R", "N_2", "O_R", "O_2", "C_R", "C_2"}:
            barrier = 6.0
        else:
            continue
        angle = _out_of_plane_angle(
            molecule.atoms[center_index].position,
            molecule.atoms[attached[0]].position,
            molecule.atoms[attached[1]].position,
            molecule.atoms[attached[2]].position,
        )
        if angle is None:
            return UffOutOfPlaneReport(None, terms, ("UFF: collinear neighbors cannot form an inversion plane",))
        energy += barrier * (1.0 - math.cos(angle))
        terms += 1
    return UffOutOfPlaneReport(energy, terms, ())


def partial_energy(molecule: Molecule) -> UffPartialEnergyReport:
    """Sum the implemented UFF terms; this is not a complete UFF energy yet."""

    reports = (
        bond_stretch_energy(molecule),
        angle_bend_energy(molecule),
        torsion_energy(molecule),
        out_of_plane_energy(molecule),
        van_der_waals_energy(molecule),
    )
    messages = tuple(message for report in reports for message in report.messages)
    if messages:
        return UffPartialEnergyReport(None, messages)
    return UffPartialEnergyReport(sum(report.energy_kcal_mol for report in reports), ())


def _with_positions(molecule: Molecule, positions: list[list[float]]) -> Molecule:
    return Molecule(
        molecule.name,
        tuple(
            Atom(atom.element, tuple(positions[index]), atom.formal_charge)
            for index, atom in enumerate(molecule.atoms)
        ),
        molecule.bonds,
    )


def minimize_partial_energy(
    molecule: Molecule, iterations: int, max_step: float, fixed_atoms: set[int] | None = None,
) -> UffMinimizationResult:
    """Minimize the implemented UFF subset using finite differences.

    This intentionally uses a bounded, backtracking steepest-descent step: it
    is slower than analytical gradients but keeps the first Blender-native
    optimizer auditable and makes it straightforward to honour fixed atoms.
    """

    fixed = fixed_atoms or set()
    initial = partial_energy(molecule)
    if not initial.ready:
        return UffMinimizationResult(None, None, None, 0, "not ready", initial.messages)
    positions = [list(atom.position) for atom in molecule.atoms]
    current_energy = initial.energy_kcal_mol
    completed = 0
    finite_difference = 1e-4
    termination = "iteration limit"
    for _ in range(iterations):
        gradient = [[0.0, 0.0, 0.0] for _atom in positions]
        for atom_index in range(len(positions)):
            if atom_index in fixed:
                continue
            for axis in range(3):
                positions[atom_index][axis] += finite_difference
                higher = partial_energy(_with_positions(molecule, positions))
                positions[atom_index][axis] -= 2.0 * finite_difference
                lower = partial_energy(_with_positions(molecule, positions))
                positions[atom_index][axis] += finite_difference
                if not higher.ready or not lower.ready:
                    messages = higher.messages or lower.messages
                    return UffMinimizationResult(
                        None, initial.energy_kcal_mol, None, completed, "evaluation failed", messages
                    )
                gradient[atom_index][axis] = (
                    higher.energy_kcal_mol - lower.energy_kcal_mol
                ) / (2.0 * finite_difference)
        norm = math.sqrt(sum(component * component for vector in gradient for component in vector))
        if norm < 1e-5:
            termination = "gradient tolerance"
            break
        accepted = False
        step = max_step
        while step >= max_step / 128.0:
            candidate_positions = [position[:] for position in positions]
            for atom_index, vector in enumerate(gradient):
                if atom_index in fixed:
                    continue
                for axis in range(3):
                    candidate_positions[atom_index][axis] -= step * vector[axis] / norm
            candidate = _with_positions(molecule, candidate_positions)
            evaluation = partial_energy(candidate)
            if evaluation.ready and evaluation.energy_kcal_mol < current_energy:
                positions = candidate_positions
                current_energy = evaluation.energy_kcal_mol
                accepted = True
                completed += 1
                break
            step *= 0.5
        if not accepted:
            termination = "line search stalled"
            break
    return UffMinimizationResult(
        _with_positions(molecule, positions), initial.energy_kcal_mol, current_energy, completed, termination, ()
    )
