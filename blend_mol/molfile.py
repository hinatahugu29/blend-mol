"""Small, dependency-free MDL Molfile V2000 reader and writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re


@dataclass(frozen=True)
class Atom:
    element: str
    position: tuple[float, float, float]
    formal_charge: int = 0
    chirality: str | None = None


@dataclass(frozen=True)
class Bond:
    atom_a: int
    atom_b: int
    order: int
    stereo: str | None = None


@dataclass(frozen=True)
class Molecule:
    name: str
    atoms: tuple[Atom, ...]
    bonds: tuple[Bond, ...]


TYPICAL_VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}


def _target_valence(atom: Atom) -> int | None:
    """Typical, neutral-state valence used for implicit hydrogen counts.

    Deliberately not the ceiling used to flag an over-valence error: sulfur
    and phosphorus routinely exceed this in legitimate hypervalent groups
    (sulfoxide/sulfone/sulfonamide, phosphate/phosphonate), but adding
    implicit hydrogens there would be wrong, not just conservative, so
    callers that place hydrogens keep using this value (its excess is
    already clamped to zero, never negative). Use _max_legitimate_valence
    for anything that reports or rejects an apparent valence violation.
    """
    if atom.element == "N" and atom.formal_charge > 0:
        return 4
    if atom.element == "O" and atom.formal_charge < 0:
        return 1
    return TYPICAL_VALENCE.get(atom.element)


# Highest bond-order sum sulfur/phosphorus can legitimately reach in common
# neutral hypervalent groups (sulfoxide/sulfone/sulfonamide for S=4/6;
# phosphate/phosphonate/phosphine oxide for P=5), via expanded-octet bonding
# that _target_valence's typical/neutral value does not represent.
_HYPERVALENT_CEILING = {"S": 6, "P": 5}


def _max_legitimate_valence(atom: Atom) -> int | None:
    """Upper bound for flagging a genuine over-valence error.

    Matches _target_valence everywhere except sulfur and phosphorus, where
    it allows common hypervalent oxidation states instead of flagging them
    as errors.
    """
    typical = _target_valence(atom)
    if typical is None:
        return None
    return max(typical, _HYPERVALENT_CEILING.get(atom.element, typical))


class MolfileError(ValueError):
    """Raised when a file is not a supported V2000 Molfile."""


_SMILES_ELEMENTS = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
_SMILES_AROMATIC = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S"}


def _relax_xy(
    positions: list[list[float]], atom_count: int,
    bonded: set[tuple[int, int]], iterations: int = 80,
) -> None:
    """Push bonded pairs toward 1.45 Å and non-bonded close pairs apart, in place.

    The force magnitude is computed from the full 3D distance (so an
    existing z offset, e.g. a tetrahedral stereocenter's out-of-plane
    component, is correctly accounted for instead of double-counted), but
    only positions[i][0]/[1] are moved. A caller that has already set a
    meaningful z can call this afterwards to repair bond lengths without
    disturbing it; for an all-z=0 starting layout this reduces to a plain
    2D relaxation, since dz is always 0 there.
    """
    for _iteration in range(iterations):
        forces = [[0.0, 0.0] for _index in positions]
        for first in range(atom_count):
            for second in range(first + 1, atom_count):
                dx = positions[second][0] - positions[first][0]
                dy = positions[second][1] - positions[first][1]
                dz = positions[second][2] - positions[first][2]
                planar_distance = max(math.hypot(dx, dy), 0.001)
                distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.001)
                ux, uy = dx / planar_distance, dy / planar_distance
                if (first, second) in bonded:
                    magnitude = (distance - 1.45) * 0.16
                elif distance < 2.0:
                    magnitude = -0.035 / (distance * distance)
                else:
                    continue
                forces[first][0] += magnitude * ux
                forces[first][1] += magnitude * uy
                forces[second][0] -= magnitude * ux
                forces[second][1] -= magnitude * uy
        for index, position in enumerate(positions):
            position[0] += max(-0.10, min(0.10, forces[index][0]))
            position[1] += max(-0.10, min(0.10, forces[index][1]))


def _snap_bond_lengths_xy(
    positions: list[list[float]], bonds: list[Bond], target: float = 1.45, sweeps: int = 30,
) -> None:
    """Directly pull each bonded pair's xy separation toward ``target``.

    The force-directed relaxation in _relax_xy can settle into a stable
    local equilibrium where a bond stays stretched because the pull toward
    its ideal length is balanced by repulsion from nearby non-bonded atoms
    (observed on a pyranose sugar SMILES with several ring-adjacent
    stereocenters: two bonds stayed stuck around 2.5 Å even after 1200
    iterations). This applies a direct geometric correction instead of more
    force iterations: each sweep moves both ends of every bond halfway
    toward the target 3D distance (via an xy-only adjustment, since z holds
    tetrahedral stereo information), with later sweeps re-correcting bonds
    disturbed by an earlier bond sharing one of their atoms. It intentionally
    does not resolve non-bonded overlaps this can introduce -- a real UFF
    minimization is far better suited to that than another ad hoc force term
    here, and this only runs when relaxation left a bond significantly
    wrong, not as the general-purpose layout step.
    """
    for _sweep in range(sweeps):
        moved = False
        for bond in bonds:
            first, second = positions[bond.atom_a], positions[bond.atom_b]
            dx, dy, dz = second[0] - first[0], second[1] - first[1], second[2] - first[2]
            distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.001)
            if abs(distance - target) < 0.05:
                continue
            moved = True
            planar_distance = max(math.hypot(dx, dy), 0.001)
            # Solve for the xy-only separation that, combined with the
            # existing (fixed) z gap, reaches the target 3D distance.
            target_planar_squared = max(target * target - dz * dz, 0.0)
            target_planar = math.sqrt(target_planar_squared)
            scale = target_planar / planar_distance
            correction_x = dx * (scale - 1.0) * 0.5
            correction_y = dy * (scale - 1.0) * 0.5
            first[0] -= correction_x
            first[1] -= correction_y
            second[0] += correction_x
            second[1] += correction_y
        if not moved:
            break


def _smiles_layout(
    atom_count: int, bonds: list[Bond], chiralities: list[str | None],
) -> list[tuple[float, float, float]]:
    """Create a modest 2D starting layout; it is intentionally not a conformer generator."""
    if atom_count == 1:
        return [(0.0, 0.0, 0.0)]
    positions = [
        [1.45 * index, 0.35 * math.sin(index * 2.4), 0.0]
        for index in range(atom_count)
    ]
    bonded = {tuple(sorted((bond.atom_a, bond.atom_b))) for bond in bonds}
    _relax_xy(positions, atom_count, bonded)
    center_x = sum(position[0] for position in positions) / atom_count
    center_y = sum(position[1] for position in positions) / atom_count
    centered = [[position[0] - center_x, position[1] - center_y, 0.0] for position in positions]
    neighbors: list[list[int]] = [[] for _index in range(atom_count)]
    for bond in bonds:
        neighbors[bond.atom_a].append(bond.atom_b)
        neighbors[bond.atom_b].append(bond.atom_a)
    tetrahedron = (
        (1.0, 1.0, 1.0), (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0),
    )
    scale = 1.45 / math.sqrt(3.0)
    # Read every center's own position from a frozen pre-loop snapshot, not
    # from `centered` as the loop mutates it. Reading live would let a later
    # stereocenter's placement build on an *earlier* stereocenter's already
    # offset output whenever they're adjacent (the common case in a ring of
    # stereocenters, e.g. a pyranose sugar), compounding a small per-center
    # offset into a multi-Angstrom bond stretch after a few links in the
    # chain -- confirmed by reproducing exactly that on
    # "OC[C@@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O" before this fix.
    reference = [tuple(position) for position in centered]
    # Two bonded stereocenters are mutually a "neighbour to place" for each
    # other, but each is also the fixed anchor the *other* substituents it
    # placed were positioned relative to. Letting one center's placement
    # overwrite the other's own position, after already using that other
    # center's original position to place a third atom, breaks the very
    # relationship that third atom's placement depended on -- a real and
    # distinct conflict from two centers merely sharing one ordinary
    # neighbour. Confirmed directly on beta-glucose
    # (OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O): atoms 4 and 6 are both
    # stereocenters bonded to each other; atom 6 overwrote atom 4's position
    # after atom 3 had already been placed relative to atom 4's *original*
    # spot, landing atom 3 and atom 6 almost exactly on top of each other.
    # Never repositioning a stereocenter via another center's placement
    # avoids this; each center's own position always stays whatever the
    # initial relaxation (or, later, the xy-only repair pass) gave it.
    #
    # An *ordinary* (non-stereocenter) neighbour can still legitimately be
    # claimed by two different adjacent centers -- the common case for a
    # ring or chain oxygen between two stereocenters -- so those proposals
    # are collected and averaged together, rather than letting whichever
    # center is processed last overwrite the other's equally valid
    # placement (and rather than a running pairwise average, which would
    # silently over-weight later entries if three or more centers ever
    # shared one neighbour).
    proposals: list[list[list[float]]] = [[] for _ in range(atom_count)]
    for center_index, chirality in enumerate(chiralities):
        if chirality is None or len(neighbors[center_index]) < 3:
            continue
        ordered_neighbors = sorted(neighbors[center_index])[:4]
        vectors = list(tetrahedron)
        if chirality == "@@":
            vectors[0], vectors[1] = vectors[1], vectors[0]
        center = reference[center_index]
        for neighbor_index, vector in zip(ordered_neighbors, vectors):
            if chiralities[neighbor_index] is not None:
                continue
            proposals[neighbor_index].append(
                [center[axis] + vector[axis] * scale for axis in range(3)]
            )
    for neighbor_index, placements in enumerate(proposals):
        if not placements:
            continue
        centered[neighbor_index] = [
            sum(placement[axis] for placement in placements) / len(placements)
            for axis in range(3)
        ]
    # Directional markers on the two single bonds around a double bond encode
    # whether its substituents begin on opposite or matching sides.
    for double_bond in (bond for bond in bonds if bond.order == 2):
        first_side = next(
            (bond for bond in bonds if bond is not double_bond and bond.stereo
             and double_bond.atom_a in {bond.atom_a, bond.atom_b}), None,
        )
        second_side = next(
            (bond for bond in bonds if bond is not double_bond and bond.stereo
             and double_bond.atom_b in {bond.atom_a, bond.atom_b}), None,
        )
        if first_side is None or second_side is None:
            continue
        first_substituent = first_side.atom_b if first_side.atom_a == double_bond.atom_a else first_side.atom_a
        second_substituent = second_side.atom_b if second_side.atom_a == double_bond.atom_b else second_side.atom_a
        first_center, second_center = centered[double_bond.atom_a], centered[double_bond.atom_b]
        dx, dy = second_center[0] - first_center[0], second_center[1] - first_center[1]
        length = max(math.hypot(dx, dy), 0.001)
        axis = (dx / length, dy / length)
        perpendicular = (-axis[1], axis[0])
        second_sign = -1.0 if first_side.stereo == second_side.stereo else 1.0
        centered[first_substituent] = [
            first_center[0] - axis[0] * 0.70 + perpendicular[0] * 1.20,
            first_center[1] - axis[1] * 0.70 + perpendicular[1] * 1.20,
            first_center[2],
        ]
        centered[second_substituent] = [
            second_center[0] + axis[0] * 0.70 + perpendicular[0] * 1.20 * second_sign,
            second_center[1] + axis[1] * 0.70 + perpendicular[1] * 1.20 * second_sign,
            second_center[2],
        ]
    # Placing a stereocenter's neighbours on a fixed tetrahedron (and a
    # double bond's substituents on fixed offsets) overwrites whatever x/y
    # position the initial relaxation gave those atoms, without regard to
    # positions a *different*, already-processed stereocenter assigned to a
    # shared ring neighbour. Left alone, that can leave some bonds several
    # Angstroms long in a ring with adjacent stereocenters (e.g. a pyranose
    # sugar) -- not just a cosmetically poor layout, but one bad enough that
    # UFF minimization starting from it can hit a genuine collinear-angle
    # singularity. A second xy-only relaxation pass repairs bond lengths
    # without touching z, which is where tetrahedral handedness lives.
    _relax_xy(centered, atom_count, bonded, iterations=200)
    _snap_bond_lengths_xy(centered, bonds)
    # Two different stereocenters can still both claim a shared ring
    # neighbour (the last one processed wins); if that neighbour's offset
    # happens to coincide with an unrelated atom's -- observed exactly on
    # the same pyranose sugar SMILES above, two atoms 4+ bonds apart ended
    # up at identical coordinates -- a UFF angle/vdW evaluation at that atom
    # would face an outright zero-distance degeneracy, worse than a merely
    # stretched bond. Nudge any atom within 0.05 Angstrom of an earlier one
    # apart deterministically so no two atoms are ever exactly coincident.
    for index in range(1, atom_count):
        for earlier in range(index):
            dx = centered[index][0] - centered[earlier][0]
            dy = centered[index][1] - centered[earlier][1]
            dz = centered[index][2] - centered[earlier][2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < 0.05:
                angle = index * 2.399963
                centered[index][0] += 0.3 * math.cos(angle)
                centered[index][1] += 0.3 * math.sin(angle)
                break
    return [tuple(position) for position in centered]


def read_smiles(smiles: str, name: str = "SMILES") -> Molecule:
    """Read a small, dependency-free organic SMILES subset into an editable graph.

    The reader supports common organic atoms, bracket formal charges, branches,
    ring closures, disconnected fragments, single/double/triple bonds, and
    common aromatic atom notation. Aromatic bonds are converted to an editable
    alternating single/double representation. Tetrahedral and directional
    double-bond stereochemistry are retained in the editable graph and layout.
    """
    text = smiles.strip()
    if not text:
        raise MolfileError("SMILES is empty.")
    atoms: list[Atom] = []
    aromatic_atoms: list[bool] = []
    bonds: list[Bond] = []
    aromatic_bond_indices: list[int] = []
    current: int | None = None
    branches: list[int] = []
    rings: dict[str, tuple[int, int, bool, bool, str | None]] = {}
    pending_order = 1
    pending_explicit = False
    pending_aromatic = False
    pending_stereo: str | None = None
    index = 0

    def reset_pending_bond() -> None:
        nonlocal pending_order, pending_explicit, pending_aromatic, pending_stereo
        pending_order = 1
        pending_explicit = False
        pending_aromatic = False
        pending_stereo = None

    def add_atom(
        element: str, charge: int, aromatic: bool = False, chirality: str | None = None,
    ) -> None:
        nonlocal current
        atom_index = len(atoms)
        atoms.append(Atom(element, (0.0, 0.0, 0.0), charge, chirality))
        aromatic_atoms.append(aromatic)
        if current is not None:
            bonds.append(Bond(current, atom_index, pending_order, pending_stereo))
            if pending_aromatic or (aromatic_atoms[current] and aromatic and not pending_explicit):
                aromatic_bond_indices.append(len(bonds) - 1)
        current = atom_index
        reset_pending_bond()

    while index < len(text):
        character = text[index]
        if character in "-=#:/\\":
            if current is None or pending_explicit:
                raise MolfileError(f"SMILES bond symbol at character {index + 1} is misplaced.")
            pending_order = {"-": 1, "=": 2, "#": 3, ":": 1, "/": 1, "\\": 1}[character]
            pending_explicit = True
            pending_aromatic = character == ":"
            pending_stereo = character if character in "/\\" else None
            index += 1
        elif character == "(":
            if current is None:
                raise MolfileError("SMILES branch has no preceding atom.")
            branches.append(current)
            index += 1
        elif character == ")":
            if not branches:
                raise MolfileError("SMILES has an unmatched closing branch.")
            current = branches.pop()
            index += 1
        elif character == ".":
            if pending_explicit:
                raise MolfileError("SMILES bond cannot end at a fragment separator.")
            current = None
            index += 1
        elif character.isdigit() or character == "%":
            if current is None:
                raise MolfileError("SMILES ring closure has no atom.")
            if character == "%":
                label = text[index + 1:index + 3]
                if len(label) != 2 or not label.isdigit():
                    raise MolfileError("SMILES percent ring closure must use two digits.")
                index += 3
            else:
                label = character
                index += 1
            if label in rings:
                other, opening_order, opening_explicit, opening_aromatic, opening_stereo = rings.pop(label)
                if pending_explicit and opening_explicit and pending_order != opening_order:
                    raise MolfileError(f"SMILES ring {label} has conflicting bond orders.")
                order = pending_order if pending_explicit else opening_order
                stereo = pending_stereo if pending_explicit else opening_stereo
                bonds.append(Bond(other, current, order, stereo))
                if pending_aromatic or opening_aromatic or (
                    not pending_explicit and not opening_explicit
                    and aromatic_atoms[other] and aromatic_atoms[current]
                ):
                    aromatic_bond_indices.append(len(bonds) - 1)
            else:
                rings[label] = (current, pending_order, pending_explicit, pending_aromatic, pending_stereo)
            reset_pending_bond()
        elif character == "[":
            closing = text.find("]", index + 1)
            if closing < 0:
                raise MolfileError("SMILES bracket atom is not closed.")
            token = text[index + 1:closing]
            match = re.fullmatch(
                r"(?:\d*)?([A-Z][a-z]?|[bcnops])(@@?)?(H(\d*)?)?([+-](?:\d+)?|\d+[+-])?", token
            )
            raw_element = match.group(1) if match else ""
            aromatic = raw_element in _SMILES_AROMATIC
            element = _SMILES_AROMATIC.get(raw_element, raw_element)
            if not match or element not in _SMILES_ELEMENTS:
                raise MolfileError(f"SMILES bracket atom [{token}] is not supported.")
            charge_text = match.group(5) or ""
            if charge_text.endswith("+") or charge_text.endswith("-"):
                sign = 1 if charge_text.endswith("+") else -1
                digits = charge_text[:-1]
            else:
                sign = 1 if charge_text.startswith("+") else -1
                digits = charge_text[1:]
            charge = sign * int(digits or "1") if charge_text else 0
            add_atom(element, charge, aromatic, match.group(2))
            if aromatic and match.group(3):
                hydrogen_count = int(match.group(4) or "1")
                if hydrogen_count != 1:
                    raise MolfileError("Aromatic bracket atoms support one hydrogen at most.")
                parent = current
                atoms.append(Atom("H", (0.0, 0.0, 0.0)))
                aromatic_atoms.append(False)
                bonds.append(Bond(parent, len(atoms) - 1, 1))
            index = closing + 1
        elif character.isalpha():
            if character in _SMILES_AROMATIC:
                add_atom(_SMILES_AROMATIC[character], 0, True)
                index += 1
                continue
            token = text[index:index + 2] if text[index:index + 2] in {"Cl", "Br"} else character
            if token not in _SMILES_ELEMENTS:
                raise MolfileError(f"SMILES element at character {index + 1} is not supported.")
            add_atom(token, 0)
            index += len(token)
        else:
            raise MolfileError(f"SMILES character {character!r} at position {index + 1} is not supported.")
    if branches:
        raise MolfileError("SMILES has an unclosed branch.")
    if rings:
        raise MolfileError("SMILES has an unclosed ring label.")
    if pending_explicit:
        raise MolfileError("SMILES ends with a bond symbol.")
    if not atoms:
        raise MolfileError("SMILES contains no atoms.")
    double_bonded_atoms = {
        atom_index
        for bond in bonds if bond.order == 2
        for atom_index in (bond.atom_a, bond.atom_b)
    }
    # Every aromatic bond still holds its unpromoted order (1) here, so this
    # sum is each atom's current sigma-bond count before any of them gets a
    # Kekule double bond.
    bond_order_sum = [0] * len(atoms)
    for bond in bonds:
        bond_order_sum[bond.atom_a] += bond.order
        bond_order_sum[bond.atom_b] += bond.order
    for bond_index in aromatic_bond_indices:
        bond = bonds[bond_index]
        if bond.atom_a in double_bonded_atoms or bond.atom_b in double_bonded_atoms:
            continue
        # A 3-substituted aromatic atom (e.g. a pyrrole/purine-style nitrogen
        # bearing a ring-exocyclic substituent) donates its lone pair into
        # the ring and must keep both ring bonds single in any valid Kekule
        # structure -- promoting either one here would push it a bond order
        # over its valence (e.g. a trisubstituted neutral N already at 3).
        target_a = _max_legitimate_valence(atoms[bond.atom_a])
        target_b = _max_legitimate_valence(atoms[bond.atom_b])
        if (
            (target_a is not None and bond_order_sum[bond.atom_a] + 1 > target_a)
            or (target_b is not None and bond_order_sum[bond.atom_b] + 1 > target_b)
        ):
            continue
        bonds[bond_index] = Bond(bond.atom_a, bond.atom_b, 2, bond.stereo)
        double_bonded_atoms.update((bond.atom_a, bond.atom_b))
        bond_order_sum[bond.atom_a] += 1
        bond_order_sum[bond.atom_b] += 1
    positions = _smiles_layout(len(atoms), bonds, [atom.chirality for atom in atoms])
    return Molecule(
        name,
        tuple(
            Atom(atom.element, positions[index], atom.formal_charge, atom.chirality)
            for index, atom in enumerate(atoms)
        ),
        tuple(bonds),
    )


def _parse_counts(line: str) -> tuple[int, int]:
    try:
        return int(line[0:3]), int(line[3:6])
    except ValueError as exc:
        raise MolfileError("MOL counts line is invalid.") from exc


def _parse_atom(line: str, line_number: int) -> Atom:
    try:
        position = (float(line[0:10]), float(line[10:20]), float(line[20:30]))
        element = line[31:34].strip()
        charge_code = int(line[36:39] or "0")
    except ValueError as exc:
        raise MolfileError(f"Atom record on line {line_number} is invalid.") from exc
    if not element:
        raise MolfileError(f"Atom record on line {line_number} has no element.")
    charge = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}.get(charge_code)
    if charge is None:
        raise MolfileError(f"Atom record on line {line_number} has unsupported charge code.")
    return Atom(element=element, position=position, formal_charge=charge)


def _apply_m_chg(atoms: tuple[Atom, ...], lines: list[str]) -> tuple[Atom, ...]:
    updated = list(atoms)
    for line in lines:
        if not line.startswith("M  CHG"):
            continue
        fields = line.split()
        try:
            count = int(fields[2])
            values = fields[3:]
            if len(values) != count * 2:
                raise ValueError
            for offset in range(0, len(values), 2):
                atom_index = int(values[offset]) - 1
                formal_charge = int(values[offset + 1])
                atom = updated[atom_index]
                updated[atom_index] = Atom(atom.element, atom.position, formal_charge, atom.chirality)
        except (IndexError, ValueError) as exc:
            raise MolfileError("M CHG record is invalid.") from exc
    return tuple(updated)


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


def _read_v2000_lines(lines: list[str], fallback_name: str) -> Molecule:
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
    atoms = _apply_m_chg(atoms, lines[expected_lines:])
    return Molecule(name=lines[0].strip() or fallback_name, atoms=atoms, bonds=bonds)


def read_v2000(path: str | Path) -> Molecule:
    """Read the atom coordinates and single/double/triple bonds from a V2000 MOL."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = source.read_text(encoding="latin-1").splitlines()
    return _read_v2000_lines(lines, source.stem)


def read_sdf(path: str | Path) -> Molecule:
    """Read the first V2000 structure record from an SD file."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = source.read_text(encoding="latin-1")
    first_record = text.split("$$$$", 1)[0].splitlines()
    return _read_v2000_lines(first_record, source.stem)


def _render_v2000(molecule: Molecule) -> str:
    """Build a V2000 structure block, shared by MOL and SDF writing."""
    atom_count = len(molecule.atoms)
    bond_count = len(molecule.bonds)
    if atom_count > 999 or bond_count > 999:
        raise MolfileError("V2000 writing supports at most 999 atoms and 999 bonds.")
    for index, atom in enumerate(molecule.atoms):
        if not atom.element or len(atom.element) > 3 or any(char.isspace() for char in atom.element):
            raise MolfileError(f"Atom {index + 1} has an invalid element symbol.")
    for index, bond in enumerate(molecule.bonds):
        if not 0 <= bond.atom_a < atom_count or not 0 <= bond.atom_b < atom_count:
            raise MolfileError(f"Bond {index + 1} references an unknown atom.")
        if bond.order not in {1, 2, 3}:
            raise MolfileError(f"Bond {index + 1} has unsupported order {bond.order}.")

    lines = [
        molecule.name or "Blend Mol molecule",
        "Blend Mol",
        "",
        f"{atom_count:>3}{bond_count:>3}  0  0  0  0  0  0  0  0999 V2000",
    ]
    lines.extend(
        f"{atom.position[0]:10.4f}{atom.position[1]:10.4f}{atom.position[2]:10.4f} "
        f"{atom.element:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        for atom in molecule.atoms
    )
    lines.extend(
        f"{bond.atom_a + 1:>3}{bond.atom_b + 1:>3}{bond.order:>3}  0  0  0  0"
        for bond in molecule.bonds
    )
    charges = [(index + 1, atom.formal_charge) for index, atom in enumerate(molecule.atoms) if atom.formal_charge]
    for offset in range(0, len(charges), 8):
        group = charges[offset:offset + 8]
        lines.append(f"M  CHG{len(group):>3}" + "".join(f"{index:>4}{charge:>4}" for index, charge in group))
    lines.append("M  END")
    return "\n".join(lines) + "\n"


def write_v2000(molecule: Molecule, path: str | Path) -> None:
    """Write atoms, single/double/triple bonds, and formal charges as V2000."""
    Path(path).write_text(_render_v2000(molecule), encoding="utf-8")


def write_sdf(molecule: Molecule, path: str | Path) -> None:
    """Write a single-record SD file: a V2000 block followed by ``$$$$``.

    Matches what read_sdf already reads back (the first V2000 record up to
    the ``$$$$`` delimiter); no per-atom/per-molecule SDF data tags are
    written.
    """
    Path(path).write_text(_render_v2000(molecule) + "$$$$\n", encoding="utf-8")


def write_xyz(molecule: Molecule, path: str | Path) -> None:
    """Write atom elements and Cartesian coordinates in the standard XYZ format."""
    for index, atom in enumerate(molecule.atoms):
        if not atom.element or any(char.isspace() for char in atom.element):
            raise MolfileError(f"Atom {index + 1} has an invalid element symbol.")
    lines = [str(len(molecule.atoms)), molecule.name or "Blend Mol molecule"]
    lines.extend(
        f"{atom.element} {atom.position[0]:.8f} {atom.position[1]:.8f} {atom.position[2]:.8f}"
        for atom in molecule.atoms
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pdb(molecule: Molecule, path: str | Path) -> None:
    """Write atom coordinates as HETATM records with CONECT connectivity.

    PDB CONECT records do not encode bond order, matching how read_pdb
    already reads every CONECT entry back as a single bond; re-importing a
    molecule written here loses any double/triple bond order and formal
    charges, same as it does for any other PDB file.
    """

    atom_count = len(molecule.atoms)
    if atom_count > 99999:
        raise MolfileError("PDB writing supports at most 99999 atoms.")
    for index, atom in enumerate(molecule.atoms):
        if not atom.element or len(atom.element) > 2 or any(char.isspace() for char in atom.element):
            raise MolfileError(f"Atom {index + 1} has an invalid element symbol.")

    lines = []
    for index, atom in enumerate(molecule.atoms):
        serial = index + 1
        name = atom.element.upper().ljust(4)
        # Fixed-width PDB columns (0-indexed slices): [6:11] serial,
        # [12:16] name, [17:20] resName, [21] chain, [22:26] resSeq,
        # [30:38]/[38:46]/[46:54] x/y/z, [54:60]/[60:66] occupancy/temp,
        # [76:78] element -- matching exactly what read_pdb slices back out.
        lines.append(
            f"HETATM{serial:>5} {name}{' '}MOL A{1:>4}    "
            f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}          {atom.element.upper():>2}"
        )
    neighbors: list[list[int]] = [[] for _atom in molecule.atoms]
    for bond in molecule.bonds:
        if not 0 <= bond.atom_a < atom_count or not 0 <= bond.atom_b < atom_count:
            raise MolfileError("A bond references an unknown atom.")
        neighbors[bond.atom_a].append(bond.atom_b)
        neighbors[bond.atom_b].append(bond.atom_a)
    for index, attached in enumerate(neighbors):
        if not attached:
            continue
        serial = index + 1
        attached_serials = sorted(other + 1 for other in attached)
        for offset in range(0, len(attached_serials), 4):
            group = attached_serials[offset:offset + 4]
            lines.append(f"CONECT{serial:>5}" + "".join(f"{other:>5}" for other in group))
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_xyz(path: str | Path) -> Molecule:
    """Read atom elements and Cartesian coordinates from a standard XYZ file."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = source.read_text(encoding="latin-1").splitlines()
    if len(lines) < 2:
        raise MolfileError("XYZ file must contain an atom count and comment line.")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise MolfileError("XYZ atom count is invalid.") from exc
    if atom_count < 0 or len(lines) < atom_count + 2:
        raise MolfileError("XYZ file ends before all atom coordinate records.")
    atoms = []
    for index, line in enumerate(lines[2:atom_count + 2], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise MolfileError(f"XYZ atom record on line {index} is invalid.")
        try:
            position = (float(fields[1]), float(fields[2]), float(fields[3]))
        except ValueError as exc:
            raise MolfileError(f"XYZ coordinates on line {index} are invalid.") from exc
        element = fields[0]
        if not element or len(element) > 3:
            raise MolfileError(f"XYZ atom record on line {index} has an invalid element symbol.")
        atoms.append(Atom(element, position))
    return Molecule(lines[1].strip() or source.stem, tuple(atoms), ())


def read_pdb(path: str | Path) -> Molecule:
    """Read PDB ATOM/HETATM coordinates and single-bond CONECT records."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = source.read_text(encoding="latin-1").splitlines()
    atoms = []
    serial_to_index: dict[int, int] = {}
    bonds: set[tuple[int, int]] = set()
    seen_atoms = False
    for line_number, line in enumerate(lines, start=1):
        record = line[0:6].strip().upper()
        if record == "ENDMDL" and seen_atoms:
            break
        if record in {"ATOM", "HETATM"}:
            seen_atoms = True
            if len(line) < 54:
                raise MolfileError(f"PDB atom record on line {line_number} is too short.")
            alternate_location = line[16:17]
            if alternate_location not in {" ", "A", "1"}:
                continue
            try:
                serial = int(line[6:11])
                position = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError as exc:
                raise MolfileError(f"PDB atom record on line {line_number} is invalid.") from exc
            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                atom_name = "".join(character for character in line[12:16] if character.isalpha())
                element = atom_name[:1]
            element = element[:1].upper() + element[1:].lower()
            if not element:
                raise MolfileError(f"PDB atom record on line {line_number} has no element symbol.")
            if serial not in serial_to_index:
                serial_to_index[serial] = len(atoms)
                atoms.append(Atom(element, position))
        elif record == "CONECT":
            fields = line[6:].split()
            try:
                serials = [int(field) for field in fields]
            except ValueError as exc:
                raise MolfileError(f"PDB CONECT record on line {line_number} is invalid.") from exc
            if len(serials) >= 2 and serials[0] in serial_to_index:
                first = serial_to_index[serials[0]]
                for serial in serials[1:]:
                    if serial in serial_to_index and serial_to_index[serial] != first:
                        bonds.add(tuple(sorted((first, serial_to_index[serial]))))
    if not atoms:
        raise MolfileError("PDB file contains no ATOM or HETATM coordinate records.")
    return Molecule(source.stem, tuple(atoms), tuple(Bond(first, second, 1) for first, second in sorted(bonds)))
