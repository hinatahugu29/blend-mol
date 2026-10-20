"""Small, dependency-free reader for MDL Molfile V2000 structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Atom:
    element: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class Bond:
    atom_a: int
    atom_b: int
    order: int


@dataclass(frozen=True)
class Molecule:
    name: str
    atoms: tuple[Atom, ...]
    bonds: tuple[Bond, ...]


class MolfileError(ValueError):
    """Raised when a file is not a supported V2000 Molfile."""


def _parse_counts(line: str) -> tuple[int, int]:
    try:
        return int(line[0:3]), int(line[3:6])
    except ValueError as exc:
        raise MolfileError("MOL counts line is invalid.") from exc


def _parse_atom(line: str, line_number: int) -> Atom:
    try:
        position = (float(line[0:10]), float(line[10:20]), float(line[20:30]))
        element = line[31:34].strip()
    except ValueError as exc:
        raise MolfileError(f"Atom record on line {line_number} is invalid.") from exc
    if not element:
        raise MolfileError(f"Atom record on line {line_number} has no element.")
    return Atom(element=element, position=position)


def _parse_bond(line: str, line_number: int, atom_count: int) -> Bond:
    try:
        atom_a = int(line[0:3]) - 1
        atom_b = int(line[3:6]) - 1
        order = int(line[6:9])
    except ValueError as exc:
        raise MolfileError(f"Bond record on line {line_number} is invalid.") from exc
    if not 0 <= atom_a < atom_count or not 0 <= atom_b < atom_count:
        raise MolfileError(f"Bond record on line {line_number} references an unknown atom.")
    if order not in {1, 2, 3}:
        raise MolfileError(
            f"Bond record on line {line_number} has unsupported order {order}."
        )
    return Bond(atom_a=atom_a, atom_b=atom_b, order=order)


def read_v2000(path: str | Path) -> Molecule:
    """Read the atom coordinates and single/double/triple bonds from a V2000 MOL."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = source.read_text(encoding="latin-1").splitlines()
    if len(lines) < 4 or "V2000" not in lines[3]:
        raise MolfileError("Only MDL Molfile V2000 files are supported.")

    atom_count, bond_count = _parse_counts(lines[3])
    expected_lines = 4 + atom_count + bond_count
    if len(lines) < expected_lines:
        raise MolfileError("MOL file ends before all atom and bond records are present.")

    atoms = tuple(
        _parse_atom(lines[4 + index], 5 + index) for index in range(atom_count)
    )
    bonds = tuple(
        _parse_bond(lines[4 + atom_count + index], 5 + atom_count + index, atom_count)
        for index in range(bond_count)
    )
    return Molecule(name=lines[0].strip() or source.stem, atoms=atoms, bonds=bonds)
