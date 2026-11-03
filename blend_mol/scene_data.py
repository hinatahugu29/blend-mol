"""Molecule storage on the root object, selection lookup, and stored constraints.

Nothing here creates or removes Blender meshes."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime
from itertools import permutations

import bpy

from .molfile import _max_legitimate_valence, Atom, Bond, Molecule


def _serialize_atom(atom: Atom) -> dict:
    data = {"element": atom.element, "position": atom.position, "formal_charge": atom.formal_charge,
            "chirality": atom.chirality}
    # Written only when set, so molecules without isotopes keep the same payload.
    if atom.isotope is not None:
        data["isotope"] = atom.isotope
    return data


def _serialize_molecule(molecule: Molecule) -> str:
    return json.dumps({
        "name": molecule.name,
        "atoms": [
            _serialize_atom(atom)
            for atom in molecule.atoms
        ],
        "bonds": [
            {"atom_a": bond.atom_a, "atom_b": bond.atom_b, "order": bond.order, "stereo": bond.stereo}
            for bond in molecule.bonds
        ],
    }, separators=(",", ":"))


def _molecule_revision(root: bpy.types.Object) -> int:
    return int(root.get("blend_mol_revision", 1))


def _store_topology_change(root: bpy.types.Object, molecule: Molecule) -> None:
    """Store a structural edit and invalidate calculations using older topology."""
    root["blend_mol_data"] = _serialize_molecule(molecule)
    root["blend_mol_revision"] = _molecule_revision(root) + 1


def _topology_signature(molecule: Molecule) -> tuple:
    """Undo-safe identity for all data that changes UFF terms or parameters."""
    return (
        tuple((atom.element, atom.formal_charge, atom.chirality) for atom in molecule.atoms),
        tuple((bond.atom_a, bond.atom_b, bond.order, bond.stereo) for bond in molecule.bonds),
    )


def _store_live_uff_session(
    root: bpy.types.Object, initial_positions: str, topology: tuple,
    held_atom: int, applied_updates: int,
) -> bool:
    """Save a completed interactive session when its topology is still valid."""
    final_molecule = _deserialize_molecule(root["blend_mol_data"])
    if _topology_signature(final_molecule) != topology:
        return False
    root["blend_mol_uff_before_positions"] = initial_positions
    root["blend_mol_uff_after_positions"] = _serialize_positions(final_molecule)
    root["blend_mol_live_uff_session"] = json.dumps({
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "held_atom": held_atom,
        "applied_updates": applied_updates,
        "final_phase": "stopped",
    }, separators=(",", ":"))
    return True


def _deserialize_molecule(payload: str) -> Molecule:
    data = json.loads(payload)
    return Molecule(
        name=data["name"],
        atoms=tuple(
            Atom(atom["element"], tuple(atom["position"]), atom.get("formal_charge", 0), atom.get("chirality"),
                 atom.get("isotope"))
            for atom in data["atoms"]
        ),
        bonds=tuple(
            Bond(bond["atom_a"], bond["atom_b"], bond["order"], bond.get("stereo"))
            for bond in data["bonds"]
        ),
    )


def _molecule_collection(root: bpy.types.Object) -> bpy.types.Collection | None:
    collection_name = root.get("blend_mol_collection")
    return bpy.data.collections.get(collection_name) if collection_name else None


def _selected_molecule_root(context) -> bpy.types.Object | None:
    obj = context.active_object
    while obj:
        if obj.get("blend_mol_managed"):
            return obj
        obj = obj.parent
    return None


def _selected_molecule_roots(context) -> list[bpy.types.Object]:
    """Every molecule root owning a selected object, active molecule first, without repeats."""
    roots: list[bpy.types.Object] = []
    candidates = [context.active_object] + list(context.selected_objects)
    for obj in candidates:
        while obj:
            if obj.get("blend_mol_managed"):
                if obj not in roots:
                    roots.append(obj)
                break
            obj = obj.parent
    return roots


def _selected_explicit_atom(context) -> tuple[bpy.types.Object, bpy.types.Collection, int] | None:
    obj = context.active_object
    if obj is None or obj.get("blend_mol_kind") != "atom" or obj.get("blend_mol_implicit"):
        return None
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    atom_index = obj.get("blend_mol_atom_index")
    if root is None or collection is None or atom_index is None:
        return None
    return root, collection, atom_index


def _activate_explicit_atom(context, root: bpy.types.Object, atom_index: int) -> None:
    """Make an atom the active selection after creating or rebuilding a molecule."""
    collection = _molecule_collection(root)
    if collection is None:
        return
    atom = next(
        (obj for obj in collection.objects if obj.get("blend_mol_atom_index") == atom_index),
        None,
    )
    bpy.ops.object.select_all(action="DESELECT")
    if atom is not None:
        atom.select_set(True)
        context.view_layer.objects.active = atom
    else:
        root.select_set(True)
        context.view_layer.objects.active = root


def _selected_explicit_atoms(context) -> tuple[bpy.types.Object, bpy.types.Collection, tuple[int, int]] | None:
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    if root is None or collection is None:
        return None
    indices = []
    for obj in context.selected_objects:
        if obj.parent != root or obj.get("blend_mol_kind") != "atom" or obj.get("blend_mol_implicit"):
            continue
        atom_index = obj.get("blend_mol_atom_index")
        if atom_index is not None:
            indices.append(atom_index)
    indices = tuple(sorted(set(indices)))
    if len(indices) != 2:
        return None
    return root, collection, indices


def _selected_explicit_atoms_any(context) -> tuple[bpy.types.Object, bpy.types.Collection, tuple[int, ...]] | None:
    """Like _selected_explicit_atoms, but any non-zero count of atoms, for
    operations (e.g. delete) that apply the same way to one atom or many.
    """
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    if root is None or collection is None:
        return None
    indices = []
    for obj in context.selected_objects:
        if obj.parent != root or obj.get("blend_mol_kind") != "atom" or obj.get("blend_mol_implicit"):
            continue
        atom_index = obj.get("blend_mol_atom_index")
        if atom_index is not None:
            indices.append(atom_index)
    indices = tuple(sorted(set(indices)))
    if not indices:
        return None
    return root, collection, indices


def _selected_angle_atoms(
    context,
) -> tuple[bpy.types.Object, bpy.types.Collection, tuple[int, int, int]] | None:
    """Return outer-center-outer indices, using the active atom as center."""
    active = _selected_explicit_atom(context)
    if active is None:
        return None
    root, collection, center = active
    indices = sorted({
        int(obj["blend_mol_atom_index"])
        for obj in context.selected_objects
        if obj.parent == root
        and obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and "blend_mol_atom_index" in obj
    })
    if len(indices) != 3 or center not in indices:
        return None
    outer = [index for index in indices if index != center]
    return root, collection, (outer[0], center, outer[1])


def _selected_dihedral_atoms(
    context,
) -> tuple[bpy.types.Object, bpy.types.Collection, tuple[int, int, int, int]] | None:
    """Infer one unambiguous bonded four-atom path from the selection."""
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    if root is None or collection is None:
        return None
    indices = sorted({
        int(obj["blend_mol_atom_index"])
        for obj in context.selected_objects
        if obj.parent == root
        and obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and "blend_mol_atom_index" in obj
    })
    if len(indices) != 4:
        return None
    molecule = _deserialize_molecule(root["blend_mol_data"])
    bonds = {frozenset((bond.atom_a, bond.atom_b)) for bond in molecule.bonds}
    paths = set()
    for candidate in permutations(indices):
        if all(frozenset(candidate[offset:offset + 2]) in bonds for offset in range(3)):
            reverse = tuple(reversed(candidate))
            paths.add(min(candidate, reverse))
    if len(paths) != 1:
        return None
    return root, collection, next(iter(paths))


def _selected_explicit_bond(context) -> tuple[bpy.types.Object, bpy.types.Collection, int] | None:
    obj = context.active_object
    if obj is None or obj.get("blend_mol_kind") != "bond" or obj.get("blend_mol_implicit"):
        return None
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    bond_index = obj.get("blend_mol_bond_index")
    if root is None or collection is None or bond_index is None:
        return None
    return root, collection, bond_index


def _sync_atom_positions(root: bpy.types.Object, collection: bpy.types.Collection) -> Molecule:
    """Copy user-edited displayed atom positions back into the stored molecule graph."""
    molecule = _deserialize_molecule(root["blend_mol_data"])
    positions: dict[int, tuple[float, float, float]] = {}
    for obj in collection.objects:
        if obj.get("blend_mol_kind") != "atom" or obj.get("blend_mol_implicit"):
            continue
        atom_index = obj.get("blend_mol_atom_index")
        if atom_index is None:
            continue
        positions[atom_index] = tuple(obj.matrix_local.translation)
    expected = set(range(len(molecule.atoms)))
    if set(positions) != expected:
        raise ValueError("Displayed atoms no longer match the stored molecule data")
    updated_atoms = tuple(
        replace(atom, position=positions[index])
        for index, atom in enumerate(molecule.atoms)
    )
    updated = Molecule(molecule.name, updated_atoms, molecule.bonds)
    root["blend_mol_data"] = _serialize_molecule(updated)
    return updated


def _molecule_validation_messages(molecule: Molecule) -> list[str]:
    bond_order_sum = [0] * len(molecule.atoms)
    messages = []
    for bond in molecule.bonds:
        if bond.atom_a == bond.atom_b:
            messages.append(f"Bond connects atom {bond.atom_a + 1} to itself")
            continue
        if not 0 <= bond.atom_a < len(molecule.atoms) or not 0 <= bond.atom_b < len(molecule.atoms):
            messages.append("Bond references an atom that does not exist")
            continue
        bond_order_sum[bond.atom_a] += bond.order
        bond_order_sum[bond.atom_b] += bond.order
    for index, atom in enumerate(molecule.atoms):
        valence = _max_legitimate_valence(atom)
        if valence is None:
            messages.append(f"Atom {index + 1} ({atom.element}) has no valence rule")
        elif bond_order_sum[index] > valence:
            messages.append(
                f"Atom {index + 1} ({atom.element}) has bond order {bond_order_sum[index]} > {valence}"
            )
    return messages


def _bond_edit_valence_error(original: Molecule, updated: Molecule, indices) -> str | None:
    """Reject edits that introduce or worsen a known valence excess.

    Allow an existing invalid structure to be repaired one bond order at a time.
    These are the same conservative, charge-aware rules used by the panel.
    """
    def bond_order_sum(molecule, index):
        return sum(bond.order for bond in molecule.bonds
                   if index in (bond.atom_a, bond.atom_b))

    for index in indices:
        atom = updated.atoms[index]
        target = _max_legitimate_valence(atom)
        total = bond_order_sum(updated, index)
        previous = bond_order_sum(original, index) if index < len(original.atoms) else 0
        if target is not None and total > target and total > previous:
            return (
                f"Cannot add this bond: atom {index + 1} ({atom.element}) would have "
                f"bond order {total}, above its valence {target}. "
                "Remove an existing bond or explicit hydrogen first. "
                "To replace a hydrogen, select it and use Change Selected Atom Element."
            )
    return None


def _store_position_snapshot(root: bpy.types.Object, molecule: Molecule) -> None:
    root["blend_mol_position_snapshot"] = _serialize_positions(molecule)


def _serialize_positions(molecule: Molecule) -> str:
    return json.dumps([atom.position for atom in molecule.atoms], separators=(",", ":"))


def _apply_serialized_positions(molecule: Molecule, serialized: str) -> Molecule:
    positions = json.loads(serialized)
    if len(positions) != len(molecule.atoms):
        raise ValueError("The saved coordinates belong to a different atom count")
    atoms = tuple(
        replace(atom, position=tuple(positions[index]))
        for index, atom in enumerate(molecule.atoms)
    )
    return Molecule(molecule.name, atoms, molecule.bonds)


def _fixed_atom_indices(root: bpy.types.Object) -> set[int]:
    try:
        return set(json.loads(root.get("blend_mol_fixed_atoms", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def _store_fixed_atom_indices(root: bpy.types.Object, indices: set[int]) -> None:
    root["blend_mol_fixed_atoms"] = json.dumps(sorted(indices))


def _distance_constraints(root: bpy.types.Object) -> tuple[tuple[int, int, float, float], ...]:
    data = json.loads(root.get("blend_mol_distance_constraints", "[]"))
    constraints = []
    for item in data:
        first, second = sorted((int(item["atom_a"]), int(item["atom_b"])))
        constraints.append((first, second, float(item["target"]), float(item["strength"])))
    return tuple(constraints)


def _store_distance_constraints(
    root: bpy.types.Object, constraints: tuple[tuple[int, int, float, float], ...]
) -> None:
    root["blend_mol_distance_constraints"] = json.dumps([
        {"atom_a": first, "atom_b": second, "target": target, "strength": strength}
        for first, second, target, strength in constraints
    ], separators=(",", ":"))
    root["blend_mol_revision"] = _molecule_revision(root) + 1


def _angle_constraints(root: bpy.types.Object) -> tuple[tuple[int, int, int, float, float], ...]:
    data = json.loads(root.get("blend_mol_angle_constraints", "[]"))
    constraints = []
    for item in data:
        first, second = sorted((int(item["outer_a"]), int(item["outer_b"])))
        constraints.append((
            first, int(item["center"]), second,
            math.radians(float(item["target_degrees"])), float(item["strength"]),
        ))
    return tuple(constraints)


def _store_angle_constraints(
    root: bpy.types.Object, constraints: tuple[tuple[int, int, int, float, float], ...]
) -> None:
    root["blend_mol_angle_constraints"] = json.dumps([
        {"outer_a": first, "center": center, "outer_b": second,
         "target_degrees": math.degrees(target), "strength": strength}
        for first, center, second, target, strength in constraints
    ], separators=(",", ":"))
    root["blend_mol_revision"] = _molecule_revision(root) + 1


def _dihedral_constraints(
    root: bpy.types.Object,
) -> tuple[tuple[int, int, int, int, float, float], ...]:
    data = json.loads(root.get("blend_mol_dihedral_constraints", "[]"))
    return tuple((
        int(item["atom_a"]), int(item["atom_b"]), int(item["atom_c"]), int(item["atom_d"]),
        math.radians(float(item["target_degrees"])), float(item["strength"]),
    ) for item in data)


def _store_dihedral_constraints(
    root: bpy.types.Object,
    constraints: tuple[tuple[int, int, int, int, float, float], ...],
) -> None:
    root["blend_mol_dihedral_constraints"] = json.dumps([
        {"atom_a": first, "atom_b": center_a, "atom_c": center_b, "atom_d": fourth,
         "target_degrees": math.degrees(target), "strength": strength}
        for first, center_a, center_b, fourth, target, strength in constraints
    ], separators=(",", ":"))
    root["blend_mol_revision"] = _molecule_revision(root) + 1


def _restore_position_snapshot(root: bpy.types.Object, molecule: Molecule) -> Molecule:
    return _apply_serialized_positions(molecule, root["blend_mol_position_snapshot"])
