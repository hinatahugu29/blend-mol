"""Display relaxation, position snapshot, colour and subdivision operators."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import bpy
from bpy.types import Operator
from mathutils import Vector

from .display import (
    _clear_molecule_display, _managed_atom_elements, _palette_item, _render_molecule,
)
from .elements import COVALENT_RADII, ELEMENT_COLORS, UNKNOWN_COLOR, UNKNOWN_COVALENT_RADIUS
from .molfile import Atom, Molecule
from .scene_data import (
    _deserialize_molecule, _fixed_atom_indices, _molecule_collection,
    _restore_position_snapshot, _selected_molecule_root, _serialize_molecule,
    _store_position_snapshot,
)


def _covalent_radius(element: str) -> float:
    return COVALENT_RADII.get(element, UNKNOWN_COVALENT_RADIUS)


def _target_bond_length(atom_a: Atom, atom_b: Atom, order: int) -> float:
    """Return a display-oriented bond length; this is not a UFF parameterization."""
    return (_covalent_radius(atom_a.element) + _covalent_radius(atom_b.element)) * (
        1.0 - 0.12 * (order - 1)
    )


def _relax_display_geometry(molecule: Molecule, iterations: int, fixed_atoms: set[int]) -> Molecule:
    """Lightweight bond-length and overlap relaxation for editing, not force-field optimization."""
    positions = [Vector(atom.position) for atom in molecule.atoms]
    bonded_pairs = {tuple(sorted((bond.atom_a, bond.atom_b))) for bond in molecule.bonds}
    strength = 0.30
    for _ in range(iterations):
        for bond in molecule.bonds:
            start, end = positions[bond.atom_a], positions[bond.atom_b]
            delta = end - start
            distance = delta.length
            direction = delta.normalized() if distance > 1e-8 else Vector((1.0, 0.0, 0.0))
            target = _target_bond_length(molecule.atoms[bond.atom_a], molecule.atoms[bond.atom_b], bond.order)
            correction = direction * ((distance - target) * strength)
            movable_a = bond.atom_a not in fixed_atoms
            movable_b = bond.atom_b not in fixed_atoms
            if movable_a and movable_b:
                positions[bond.atom_a] += correction * 0.5
                positions[bond.atom_b] -= correction * 0.5
            elif movable_a:
                positions[bond.atom_a] += correction
            elif movable_b:
                positions[bond.atom_b] -= correction
        for atom_a in range(len(positions)):
            for atom_b in range(atom_a + 1, len(positions)):
                if (atom_a, atom_b) in bonded_pairs:
                    continue
                delta = positions[atom_b] - positions[atom_a]
                distance = delta.length
                minimum = 0.65 * (
                    _covalent_radius(molecule.atoms[atom_a].element)
                    + _covalent_radius(molecule.atoms[atom_b].element)
                )
                if distance >= minimum:
                    continue
                direction = delta.normalized() if distance > 1e-8 else Vector((1.0, 0.0, 0.0))
                correction = direction * ((minimum - distance) * strength)
                movable_a = atom_a not in fixed_atoms
                movable_b = atom_b not in fixed_atoms
                if movable_a and movable_b:
                    positions[atom_a] -= correction * 0.5
                    positions[atom_b] += correction * 0.5
                elif movable_a:
                    positions[atom_a] -= correction
                elif movable_b:
                    positions[atom_b] += correction
    atoms = tuple(
        replace(atom, position=tuple(positions[index]))
        for index, atom in enumerate(molecule.atoms)
    )
    return Molecule(molecule.name, atoms, molecule.bonds)


class BLENDMOL_OT_rebuild_display(Operator):
    bl_idname = "blendmol.rebuild_display"
    bl_label = "Rebuild Selected Molecule"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        collection = _molecule_collection(root)
        if collection is None or "blend_mol_data" not in root:
            self.report({"ERROR"}, "This molecule does not contain rebuildable Blend Mol data")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, molecule, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Rebuilt {molecule.name}")
        return {"FINISHED"}


class BLENDMOL_OT_relax_display_geometry(Operator):
    bl_idname = "blendmol.relax_display_geometry"
    bl_label = "Relax Display Geometry"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        _store_position_snapshot(root, molecule)
        updated = _relax_display_geometry(
            molecule, context.scene.blend_mol_settings.relax_iterations, _fixed_atom_indices(root)
        )
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, "Relaxed display geometry (not UFF)")
        return {"FINISHED"}


class BLENDMOL_OT_jitter_atom_positions(Operator):
    bl_idname = "blendmol.jitter_atom_positions"
    bl_label = "Jitter All Atom Positions"
    bl_description = (
        "Nudge every non-fixed atom in the selected molecule by a small random "
        "amount; an emergency escape from an exactly-degenerate starting geometry "
        "(e.g. a collinear-angle UFF error) that keeps happening even after moving "
        "an atom by hand"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        magnitude = settings.jitter_magnitude
        fixed_atoms = _fixed_atom_indices(root)
        _store_position_snapshot(root, molecule)
        jittered_count = 0
        atoms = []
        for index, atom in enumerate(molecule.atoms):
            if index in fixed_atoms:
                atoms.append(atom)
                continue
            jittered_count += 1
            atoms.append(replace(atom, position=tuple(
                coordinate + random.uniform(-magnitude, magnitude)
                for coordinate in atom.position
            )))
        updated = Molecule(molecule.name, tuple(atoms), molecule.bonds)
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, settings)
        self.report(
            {"INFO"},
            f"Jittered {jittered_count} atom(s) by up to ±{magnitude:.3f} Å "
            f"(Restore Position Snapshot to undo)",
        )
        return {"FINISHED"}


class BLENDMOL_OT_save_position_snapshot(Operator):
    bl_idname = "blendmol.save_position_snapshot"
    bl_label = "Save Position Snapshot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        _store_position_snapshot(root, molecule)
        self.report({"INFO"}, f"Saved positions for {len(molecule.atoms)} atoms")
        return {"FINISHED"}


class BLENDMOL_OT_restore_position_snapshot(Operator):
    bl_idname = "blendmol.restore_position_snapshot"
    bl_label = "Restore Position Snapshot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None or "blend_mol_position_snapshot" not in root:
            self.report({"WARNING"}, "Select a molecule with a saved position snapshot first")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
            updated = _restore_position_snapshot(root, molecule)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot restore position snapshot: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Restored positions for {len(updated.atoms)} atoms")
        return {"FINISHED"}


class BLENDMOL_OT_reset_element_colors(Operator):
    bl_idname = "blendmol.reset_element_colors"
    bl_label = "Reset to CPK Colors"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.blend_mol_settings
        elements = _managed_atom_elements()
        for element in elements:
            _palette_item(settings, element).color = ELEMENT_COLORS.get(element, UNKNOWN_COLOR)
        self.report({"INFO"}, f"Reset colors for {sum(elements.values())} atoms")
        return {"FINISHED"}


class BLENDMOL_OT_add_subdivision(Operator):
    bl_idname = "blendmol.add_subdivision"
    bl_label = "Add Subdivision Surface"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        levels = context.scene.blend_mol_settings.subdivision_levels
        updated = 0
        for obj in bpy.data.objects:
            if obj.get("blend_mol_kind") not in {"atom", "bond"}:
                continue
            modifier = obj.modifiers.get("Blend Mol Subdivision")
            if modifier is None:
                modifier = obj.modifiers.new("Blend Mol Subdivision", "SUBSURF")
            modifier.subdivision_type = "CATMULL_CLARK"
            modifier.levels = levels
            modifier.render_levels = levels
            updated += 1
        self.report({"INFO"}, f"Added subdivision level {levels} to {updated} objects")
        return {"FINISHED"}
