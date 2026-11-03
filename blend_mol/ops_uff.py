"""Partial UFF minimization, local relaxation and benchmark operators."""

from __future__ import annotations

import json
from datetime import datetime
from time import perf_counter

from bpy.props import EnumProperty
from bpy.types import Operator

from .display import _clear_molecule_display, _render_molecule, _update_explicit_display_geometry
from .native_backend import backend_status
from .ops_edit import _ensure_hydrogens_explicit_for_uff
from .scene_data import (
    _angle_constraints, _apply_serialized_positions, _deserialize_molecule,
    _dihedral_constraints, _distance_constraints, _fixed_atom_indices, _molecule_collection,
    _selected_explicit_atom, _selected_molecule_root, _serialize_molecule, _serialize_positions,
    _store_position_snapshot, _sync_atom_positions,
)
from .uff import minimize_partial_energy, numeric_gradient_atom_indices


class BLENDMOL_OT_minimize_partial_uff(Operator):
    bl_idname = "blendmol.minimize_partial_uff"
    bl_label = "Minimize Partial UFF"
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
        molecule = _ensure_hydrogens_explicit_for_uff(root, collection, molecule, settings)
        try:
            result = minimize_partial_energy(
                molecule, settings.uff_iterations, settings.uff_max_step,
                _fixed_atom_indices(root), _distance_constraints(root), _angle_constraints(root),
                _dihedral_constraints(root),
                gradient_tolerance=settings.uff_gradient_tolerance,
                energy_tolerance=settings.uff_energy_tolerance,
                algorithm=settings.uff_optimizer,
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, f"UFF minimization stopped; original coordinates kept: {exc}")
            return {"CANCELLED"}
        if not result.ready:
            message = result.messages[0] if result.messages else "UFF minimization failed"
            self.report({"ERROR"}, f"{message}; original coordinates kept")
            return {"CANCELLED"}
        _store_position_snapshot(root, molecule)
        root["blend_mol_uff_before_positions"] = _serialize_positions(molecule)
        root["blend_mol_uff_after_positions"] = _serialize_positions(result.molecule)
        root["blend_mol_data"] = _serialize_molecule(result.molecule)
        root["blend_mol_uff_last_run"] = json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "initial_energy_kcal_mol": result.initial_energy_kcal_mol,
            "final_energy_kcal_mol": result.final_energy_kcal_mol,
            "iterations": result.iterations,
            "termination": result.termination,
            "fixed_atoms": len(_fixed_atom_indices(root)),
            "distance_constraints": len(_distance_constraints(root)),
            "angle_constraints": len(_angle_constraints(root)),
            "dihedral_constraints": len(_dihedral_constraints(root)),
            "gradient_tolerance": settings.uff_gradient_tolerance,
            "energy_tolerance": settings.uff_energy_tolerance,
            "optimizer": settings.uff_optimizer,
        }, separators=(",", ":"))
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, result.molecule, settings)
        self.report(
            {"INFO"},
            f"Partial UFF: {result.initial_energy_kcal_mol:.3f} → {result.final_energy_kcal_mol:.3f} kcal/mol"
        )
        return {"FINISHED"}


class BLENDMOL_OT_relax_around_selected_atom(Operator):
    bl_idname = "blendmol.relax_around_selected_atom"
    bl_label = "Relax Around Selected Atom"
    bl_description = "Keep the selected moved atom fixed and run a few partial UFF steps on its surroundings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select one moved, non-implicit atom first")
            return {"CANCELLED"}
        root, collection, atom_index = selected
        try:
            molecule = _sync_atom_positions(root, collection)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot synchronize moved atom: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        temporary_fixed = _fixed_atom_indices(root) | {atom_index}
        started = perf_counter()
        try:
            result = minimize_partial_energy(
                molecule, settings.uff_live_steps, settings.uff_max_step,
                temporary_fixed, _distance_constraints(root), _angle_constraints(root),
                _dihedral_constraints(root),
                gradient_tolerance=settings.uff_gradient_tolerance,
                energy_tolerance=settings.uff_energy_tolerance,
                algorithm=settings.uff_optimizer,
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, f"Interactive UFF step stopped; moved coordinates kept: {exc}")
            return {"CANCELLED"}
        if not result.ready:
            message = result.messages[0] if result.messages else "UFF step failed"
            self.report({"ERROR"}, f"{message}; moved coordinates kept")
            return {"CANCELLED"}
        elapsed_ms = (perf_counter() - started) * 1000.0
        root["blend_mol_data"] = _serialize_molecule(result.molecule)
        root["blend_mol_uff_live_last"] = json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "initial_energy_kcal_mol": result.initial_energy_kcal_mol,
            "final_energy_kcal_mol": result.final_energy_kcal_mol,
            "steps": result.iterations,
            "held_atom": atom_index,
            "elapsed_ms": elapsed_ms,
            "gradient_tolerance": settings.uff_gradient_tolerance,
            "energy_tolerance": settings.uff_energy_tolerance,
            "optimizer": settings.uff_optimizer,
        }, separators=(",", ":"))
        display_started = perf_counter()
        if not _update_explicit_display_geometry(root, collection, result.molecule, settings):
            _clear_molecule_display(root, collection)
            _render_molecule(root, collection, result.molecule, settings)
            replacement = next(
                (obj for obj in collection.objects
                 if obj.get("blend_mol_kind") == "atom"
                 and obj.get("blend_mol_atom_index") == atom_index),
                None,
            )
            if replacement is not None:
                replacement.select_set(True)
                context.view_layer.objects.active = replacement
        display_elapsed_ms = (perf_counter() - display_started) * 1000.0
        live_log = json.loads(root["blend_mol_uff_live_last"])
        live_log["display_update_ms"] = display_elapsed_ms
        live_log["total_step_ms"] = elapsed_ms + display_elapsed_ms
        root["blend_mol_uff_live_last"] = json.dumps(live_log, separators=(",", ":"))
        self.report(
            {"INFO"},
            f"Held atom {atom_index + 1}; partial UFF {result.initial_energy_kcal_mol:.3f} → "
            f"{result.final_energy_kcal_mol:.3f} kcal/mol",
        )
        return {"FINISHED"}


class BLENDMOL_OT_benchmark_uff_step(Operator):
    bl_idname = "blendmol.benchmark_uff_step"
    bl_label = "Benchmark 1 UFF Step"
    bl_description = "Measure one partial UFF step without changing the molecule coordinates"
    bl_options = {"REGISTER"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot prepare UFF benchmark: {exc}")
            return {"CANCELLED"}
        fixed_atoms = _fixed_atom_indices(root)
        started = perf_counter()
        try:
            result = minimize_partial_energy(
                molecule, 1, context.scene.blend_mol_settings.uff_max_step,
                fixed_atoms, _distance_constraints(root), _angle_constraints(root),
                _dihedral_constraints(root),
                gradient_tolerance=context.scene.blend_mol_settings.uff_gradient_tolerance,
                energy_tolerance=context.scene.blend_mol_settings.uff_energy_tolerance,
                algorithm=context.scene.blend_mol_settings.uff_optimizer,
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, f"UFF benchmark failed: {exc}")
            return {"CANCELLED"}
        elapsed_ms = (perf_counter() - started) * 1000.0
        if not result.ready:
            message = result.messages[0] if result.messages else "UFF benchmark failed"
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        moving_atoms = len(molecule.atoms) - len(fixed_atoms)
        numeric_atoms = len(numeric_gradient_atom_indices(molecule) - fixed_atoms)
        compute_backend = backend_status()
        root["blend_mol_uff_benchmark"] = json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "atom_count": len(molecule.atoms),
            "moving_atoms": moving_atoms,
            "elapsed_ms": elapsed_ms,
            "estimated_hz": 1000.0 / elapsed_ms if elapsed_ms > 0.0 else None,
            "energy_evaluations": numeric_atoms * 3 + 1,
            "numeric_gradient_atoms": numeric_atoms,
            "gradient_method": (
                "analytic bond/angle/vdw + Rust parallel forward torsion/inversion"
                if compute_backend["active"] == "rust"
                else "analytic bond/angle/vdw + Python forward torsion/inversion"
            ),
            "compute_backend": compute_backend["active"],
            "native_available": compute_backend["native_available"],
            "gradient_tolerance": context.scene.blend_mol_settings.uff_gradient_tolerance,
            "energy_tolerance": context.scene.blend_mol_settings.uff_energy_tolerance,
            "optimizer": context.scene.blend_mol_settings.uff_optimizer,
        }, separators=(",", ":"))
        self.report({"INFO"}, f"One partial UFF step: {elapsed_ms:.1f} ms ({len(molecule.atoms)} atoms)")
        return {"FINISHED"}


class BLENDMOL_OT_restore_uff_coordinates(Operator):
    bl_idname = "blendmol.restore_uff_coordinates"
    bl_label = "Restore UFF Coordinates"
    bl_description = "Restore coordinates from immediately before or after the last UFF minimization"
    bl_options = {"REGISTER", "UNDO"}

    state: EnumProperty(
        name="State",
        items=(("BEFORE", "Before", "Before the last minimization"),
               ("AFTER", "After", "Result of the last minimization")),
        default="BEFORE",
    )

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        property_name = (
            "blend_mol_uff_before_positions" if self.state == "BEFORE"
            else "blend_mol_uff_after_positions"
        )
        if root is None or collection is None or property_name not in root:
            self.report({"WARNING"}, "Select a molecule with a saved UFF comparison first")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
            updated = _apply_serialized_positions(molecule, root[property_name])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot restore UFF coordinates: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        label = "before" if self.state == "BEFORE" else "after"
        self.report({"INFO"}, f"Restored coordinates from {label} the last UFF minimization")
        return {"FINISHED"}
