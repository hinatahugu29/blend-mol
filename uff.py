"""First, dependency-free UFF atom typing support for Blend Mol.

This module deliberately contains no Blender API calls.  It is the boundary
between the editable molecular graph and a future energy/minimisation engine.
Only the well-defined organic subset is enabled at this stage; unknown or
ambiguous atoms are reported rather than silently receiving a generic type.
"""

from __future__ import annotations

from dataclasses import dataclass

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


# Organic subset transcribed from the UFF parameter set.  The remaining UFF
# columns are added only when their associated energy terms are implemented.
UFF_PARAMETERS = {
    "H_": UffParameter("H_", 0.354, 180.0, 2.886, 0.044, 0.712),
    "H_b": UffParameter("H_b", 0.460, 83.5, 2.886, 0.044, 0.712),
    "C_3": UffParameter("C_3", 0.757, 109.47, 3.851, 0.105, 1.912),
    "C_2": UffParameter("C_2", 0.732, 120.0, 3.851, 0.105, 1.912),
    "C_1": UffParameter("C_1", 0.706, 180.0, 3.851, 0.105, 1.912),
    "C_R": UffParameter("C_R", 0.729, 120.0, 3.851, 0.105, 1.912),
    "N_3": UffParameter("N_3", 0.700, 106.7, 3.660, 0.069, 2.544),
    "N_2": UffParameter("N_2", 0.685, 111.2, 3.660, 0.069, 2.544),
    "N_1": UffParameter("N_1", 0.656, 180.0, 3.660, 0.069, 2.544),
    "N_R": UffParameter("N_R", 0.699, 120.0, 3.660, 0.069, 2.544),
    "O_3": UffParameter("O_3", 0.658, 104.51, 3.500, 0.060, 2.300),
    "O_2": UffParameter("O_2", 0.634, 120.0, 3.500, 0.060, 2.300),
    "O_1": UffParameter("O_1", 0.639, 180.0, 3.500, 0.060, 2.300),
    "O_R": UffParameter("O_R", 0.680, 110.0, 3.500, 0.060, 2.300),
    "F_": UffParameter("F_", 0.668, 180.0, 3.364, 0.050, 1.735),
}


@dataclass(frozen=True)
class UffTypingReport:
    types: tuple[str | None, ...]
    messages: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.messages and all(atom_type is not None for atom_type in self.types)


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
