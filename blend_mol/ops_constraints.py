"""Fixed-atom and distance/angle/dihedral constraint operators."""

from __future__ import annotations

import json
import math

from bpy.props import BoolProperty
from bpy.types import Operator
from mathutils import Vector

from .scene_data import (
    _angle_constraints, _dihedral_constraints, _distance_constraints, _fixed_atom_indices,
    _selected_angle_atoms, _selected_dihedral_atoms, _selected_explicit_atom,
    _selected_explicit_atoms, _selected_molecule_root, _store_angle_constraints,
    _store_dihedral_constraints, _store_distance_constraints, _store_fixed_atom_indices,
    _sync_atom_positions,
)
from .uff import _dihedral_angle


class BLENDMOL_OT_toggle_fixed_atom(Operator):
    bl_idname = "blendmol.toggle_fixed_atom"
    bl_label = "Toggle Fixed Selected Atom"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select a non-implicit Blend Mol atom first")
            return {"CANCELLED"}
        root, _collection, atom_index = selected
        fixed_atoms = _fixed_atom_indices(root)
        if atom_index in fixed_atoms:
            fixed_atoms.remove(atom_index)
            state = "unfixed"
        else:
            fixed_atoms.add(atom_index)
            state = "fixed"
        _store_fixed_atom_indices(root, fixed_atoms)
        self.report({"INFO"}, f"Atom {atom_index + 1} is now {state}")
        return {"FINISHED"}


class BLENDMOL_OT_set_distance_constraint(Operator):
    bl_idname = "blendmol.set_distance_constraint"
    bl_label = "Set Selected Distance Constraint"
    bl_description = "Add or update a harmonic distance restraint between two selected atoms"
    bl_options = {"REGISTER", "UNDO"}

    use_current: BoolProperty(name="Use Current Distance", default=False)

    def execute(self, context):
        selected = _selected_explicit_atoms(context)
        if selected is None:
            self.report({"WARNING"}, "Select exactly two non-implicit atoms in one molecule")
            return {"CANCELLED"}
        root, collection, (first, second) = selected
        try:
            molecule = _sync_atom_positions(root, collection)
            constraints = list(_distance_constraints(root))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot create distance constraint: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        if self.use_current:
            delta = tuple(
                molecule.atoms[first].position[axis] - molecule.atoms[second].position[axis]
                for axis in range(3)
            )
            target = math.sqrt(sum(component * component for component in delta))
            settings.distance_constraint_target = target
        else:
            target = settings.distance_constraint_target
        term = (first, second, target, settings.distance_constraint_strength)
        constraints = [item for item in constraints if item[:2] != term[:2]]
        constraints.append(term)
        _store_distance_constraints(root, tuple(sorted(constraints)))
        self.report({"INFO"}, f"Constrained atoms {first + 1}-{second + 1} to {target:.3f} Å")
        return {"FINISHED"}


class BLENDMOL_OT_remove_distance_constraint(Operator):
    bl_idname = "blendmol.remove_distance_constraint"
    bl_label = "Remove Distance Constraint"
    bl_options = {"REGISTER", "UNDO"}

    remove_all: BoolProperty(name="Remove All", default=False)

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol molecule first")
            return {"CANCELLED"}
        try:
            constraints = list(_distance_constraints(root))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot read distance constraints: {exc}")
            return {"CANCELLED"}
        if self.remove_all:
            removed = len(constraints)
            constraints = []
        else:
            selected = _selected_explicit_atoms(context)
            if selected is None or selected[0] != root:
                self.report({"WARNING"}, "Select the two constrained atoms first")
                return {"CANCELLED"}
            pair = selected[2]
            remaining = [item for item in constraints if item[:2] != pair]
            removed = len(constraints) - len(remaining)
            constraints = remaining
        if removed == 0:
            self.report({"INFO"}, "No matching distance constraint")
            return {"CANCELLED"}
        _store_distance_constraints(root, tuple(constraints))
        self.report({"INFO"}, f"Removed {removed} distance constraint(s)")
        return {"FINISHED"}


class BLENDMOL_OT_set_angle_constraint(Operator):
    bl_idname = "blendmol.set_angle_constraint"
    bl_label = "Set Selected Angle Constraint"
    bl_description = "Use the active atom as center of a three-atom angle restraint"
    bl_options = {"REGISTER", "UNDO"}

    use_current: BoolProperty(name="Use Current Angle", default=False)

    def execute(self, context):
        selected = _selected_angle_atoms(context)
        if selected is None:
            self.report({"WARNING"}, "Select three atoms; make the center atom active")
            return {"CANCELLED"}
        root, collection, (first, center, second) = selected
        try:
            molecule = _sync_atom_positions(root, collection)
            constraints = list(_angle_constraints(root))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot create angle constraint: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        if self.use_current:
            first_vector = Vector(molecule.atoms[first].position) - Vector(molecule.atoms[center].position)
            second_vector = Vector(molecule.atoms[second].position) - Vector(molecule.atoms[center].position)
            if min(first_vector.length, second_vector.length) < 1e-10:
                self.report({"ERROR"}, "Angle atoms cannot occupy the same position")
                return {"CANCELLED"}
            target = first_vector.angle(second_vector)
            settings.angle_constraint_target = math.degrees(target)
        else:
            target = math.radians(settings.angle_constraint_target)
        key = (first, center, second)
        term = key + (target, settings.angle_constraint_strength)
        constraints = [item for item in constraints if item[:3] != key]
        constraints.append(term)
        _store_angle_constraints(root, tuple(sorted(constraints)))
        self.report(
            {"INFO"},
            f"Constrained angle {first + 1}-{center + 1}-{second + 1} to {math.degrees(target):.2f}°",
        )
        return {"FINISHED"}


class BLENDMOL_OT_remove_angle_constraint(Operator):
    bl_idname = "blendmol.remove_angle_constraint"
    bl_label = "Remove Angle Constraint"
    bl_options = {"REGISTER", "UNDO"}

    remove_all: BoolProperty(name="Remove All", default=False)

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol molecule first")
            return {"CANCELLED"}
        try:
            constraints = list(_angle_constraints(root))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot read angle constraints: {exc}")
            return {"CANCELLED"}
        if self.remove_all:
            removed = len(constraints)
            constraints = []
        else:
            selected = _selected_angle_atoms(context)
            if selected is None or selected[0] != root:
                self.report({"WARNING"}, "Select three atoms; make the center atom active")
                return {"CANCELLED"}
            key = selected[2]
            remaining = [item for item in constraints if item[:3] != key]
            removed = len(constraints) - len(remaining)
            constraints = remaining
        if removed == 0:
            self.report({"INFO"}, "No matching angle constraint")
            return {"CANCELLED"}
        _store_angle_constraints(root, tuple(constraints))
        self.report({"INFO"}, f"Removed {removed} angle constraint(s)")
        return {"FINISHED"}


class BLENDMOL_OT_set_dihedral_constraint(Operator):
    bl_idname = "blendmol.set_dihedral_constraint"
    bl_label = "Set Selected Dihedral Constraint"
    bl_description = "Infer a bonded four-atom path and add or update its dihedral restraint"
    bl_options = {"REGISTER", "UNDO"}

    use_current: BoolProperty(name="Use Current Dihedral", default=False)

    def execute(self, context):
        selected = _selected_dihedral_atoms(context)
        if selected is None:
            self.report({"WARNING"}, "Select four atoms forming one unambiguous bonded path")
            return {"CANCELLED"}
        root, collection, path = selected
        try:
            molecule = _sync_atom_positions(root, collection)
            constraints = list(_dihedral_constraints(root))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot create dihedral constraint: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        if self.use_current:
            target = _dihedral_angle(*(molecule.atoms[index].position for index in path))
            if target is None:
                self.report({"ERROR"}, "Selected path has a degenerate dihedral")
                return {"CANCELLED"}
            settings.dihedral_constraint_target = math.degrees(target)
        else:
            target = math.radians(settings.dihedral_constraint_target)
        term = path + (target, settings.dihedral_constraint_strength)
        constraints = [item for item in constraints if item[:4] != path]
        constraints.append(term)
        _store_dihedral_constraints(root, tuple(sorted(constraints)))
        self.report(
            {"INFO"},
            f"Constrained dihedral {'-'.join(str(index + 1) for index in path)} "
            f"to {math.degrees(target):.2f}°",
        )
        return {"FINISHED"}


class BLENDMOL_OT_remove_dihedral_constraint(Operator):
    bl_idname = "blendmol.remove_dihedral_constraint"
    bl_label = "Remove Dihedral Constraint"
    bl_options = {"REGISTER", "UNDO"}

    remove_all: BoolProperty(name="Remove All", default=False)

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol molecule first")
            return {"CANCELLED"}
        constraints = list(_dihedral_constraints(root))
        if self.remove_all:
            removed = len(constraints)
            constraints = []
        else:
            selected = _selected_dihedral_atoms(context)
            if selected is None or selected[0] != root:
                self.report({"WARNING"}, "Select four atoms forming one unambiguous bonded path")
                return {"CANCELLED"}
            path = selected[2]
            remaining = [item for item in constraints if item[:4] != path]
            removed = len(constraints) - len(remaining)
            constraints = remaining
        if removed == 0:
            self.report({"INFO"}, "No matching dihedral constraint")
            return {"CANCELLED"}
        _store_dihedral_constraints(root, tuple(constraints))
        self.report({"INFO"}, f"Removed {removed} dihedral constraint(s)")
        return {"FINISHED"}
