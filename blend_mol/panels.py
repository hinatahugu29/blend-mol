"""N-panel UI."""

from __future__ import annotations

import json
import math

from bpy.types import Panel

from .display import _managed_atom_elements, _palette_item
from .live_uff import BLENDMOL_OT_toggle_live_uff, STATE
from .native_backend import backend_status
from .ops_constraints import (
    BLENDMOL_OT_remove_angle_constraint, BLENDMOL_OT_remove_dihedral_constraint,
    BLENDMOL_OT_remove_distance_constraint, BLENDMOL_OT_set_angle_constraint,
    BLENDMOL_OT_set_dihedral_constraint, BLENDMOL_OT_set_distance_constraint,
    BLENDMOL_OT_toggle_fixed_atom,
)
from .ops_display import (
    BLENDMOL_OT_add_subdivision, BLENDMOL_OT_jitter_atom_positions, BLENDMOL_OT_rebuild_display,
    BLENDMOL_OT_relax_display_geometry, BLENDMOL_OT_reset_element_colors,
    BLENDMOL_OT_restore_position_snapshot, BLENDMOL_OT_save_position_snapshot,
)
from .ops_edit import (
    BLENDMOL_OT_add_atom, BLENDMOL_OT_add_bonded_atom, BLENDMOL_OT_change_atom_element,
    BLENDMOL_OT_delete_atom, BLENDMOL_OT_delete_bond, BLENDMOL_OT_make_hydrogens_explicit,
    BLENDMOL_OT_new_molecule, BLENDMOL_OT_set_bond, BLENDMOL_OT_set_formal_charge,
    BLENDMOL_OT_sync_atom_positions,
)
from .ops_io import (
    BLENDMOL_OT_export_pdb, BLENDMOL_OT_export_sdf, BLENDMOL_OT_export_uff_report,
    BLENDMOL_OT_export_v2000, BLENDMOL_OT_export_xyz, BLENDMOL_OT_import_pdb,
    BLENDMOL_OT_import_sdf, BLENDMOL_OT_import_smiles, BLENDMOL_OT_import_v2000,
    BLENDMOL_OT_import_xyz,
)
from .ops_uff import (
    BLENDMOL_OT_benchmark_uff_step, BLENDMOL_OT_minimize_partial_uff,
    BLENDMOL_OT_relax_around_selected_atom, BLENDMOL_OT_restore_uff_coordinates,
)
from .scene_data import (
    _angle_constraints, _deserialize_molecule, _dihedral_constraints, _distance_constraints,
    _fixed_atom_indices, _molecule_validation_messages, _selected_angle_atoms,
    _selected_dihedral_atoms, _selected_explicit_atom, _selected_explicit_atoms,
    _selected_explicit_bond, _selected_molecule_root,
)
from .uff import (
    angle_bend_energy, bond_stretch_energy, out_of_plane_energy, partial_energy, torsion_energy,
    type_molecule, van_der_waals_energy,
)


class BLENDMOL_PT_panel(Panel):
    """Root panel: the one always-visible entry point and current-selection status.

    Everything else lives in collapsible sub-panels below, grouped by
    workflow stage (file I/O, editing, constraints, UFF, display), so the
    N-panel isn't one long flat list of equally-weighted buttons. No
    operator or property was removed in this split -- every control from
    the previous single-panel layout is still reachable, just regrouped.
    """

    bl_label = "Blend Mol"
    bl_idname = "BLENDMOL_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        new_row = layout.row()
        new_row.scale_y = 1.5
        new_row.operator(BLENDMOL_OT_new_molecule.bl_idname, icon="FILE_NEW")
        root = _selected_molecule_root(context)
        if root:
            layout.label(text=f"Editing: {root.name}", icon="OBJECT_DATA")
        else:
            layout.label(text="Select an atom/bond/root, or create or import a molecule", icon="INFO")


class BLENDMOL_PT_file_io(Panel):
    bl_label = "Import / Export"
    bl_idname = "BLENDMOL_PT_file_io"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        layout.label(text="Import")
        import_row = layout.row(align=True)
        import_row.operator(BLENDMOL_OT_import_v2000.bl_idname, text="MOL", icon="IMPORT")
        import_row.operator(BLENDMOL_OT_import_xyz.bl_idname, text="XYZ", icon="IMPORT")
        import_row.operator(BLENDMOL_OT_import_pdb.bl_idname, text="PDB", icon="IMPORT")
        import_row.operator(BLENDMOL_OT_import_sdf.bl_idname, text="SDF", icon="IMPORT")
        layout.operator(BLENDMOL_OT_import_smiles.bl_idname, icon="MODIFIER")
        layout.separator()
        layout.label(text="Export selected molecule")
        export_row = layout.row(align=True)
        export_row.operator(BLENDMOL_OT_export_v2000.bl_idname, text="MOL", icon="EXPORT")
        export_row.operator(BLENDMOL_OT_export_xyz.bl_idname, text="XYZ", icon="EXPORT")
        export_row.operator(BLENDMOL_OT_export_pdb.bl_idname, text="PDB", icon="EXPORT")
        export_row.operator(BLENDMOL_OT_export_sdf.bl_idname, text="SDF", icon="EXPORT")
        layout.operator(BLENDMOL_OT_export_uff_report.bl_idname, icon="FILE_TEXT")


class BLENDMOL_PT_edit(Panel):
    bl_label = "Edit Atoms & Bonds"
    bl_idname = "BLENDMOL_PT_edit"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_order = 2

    @classmethod
    def poll(cls, context):
        return _selected_molecule_root(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        root = _selected_molecule_root(context)
        layout.label(text="Move atom objects, then synchronize")
        layout.operator(BLENDMOL_OT_sync_atom_positions.bl_idname, icon="CON_TRANSFORM")
        layout.separator()
        layout.prop(settings, "edit_element")
        layout.prop(settings, "formal_charge")
        layout.operator(BLENDMOL_OT_add_atom.bl_idname, icon="ADD")
        layout.operator(BLENDMOL_OT_make_hydrogens_explicit.bl_idname, icon="MODIFIER")
        selected_atom = _selected_explicit_atom(context)
        if selected_atom:
            atom_box = layout.box()
            atom_box.prop(settings, "bond_order")
            atom_box.operator(BLENDMOL_OT_add_bonded_atom.bl_idname, icon="CON_TRACKTO")
            fixed_atoms = _fixed_atom_indices(root)
            state = "fixed" if selected_atom[2] in fixed_atoms else "movable"
            molecule = _deserialize_molecule(root["blend_mol_data"])
            charge = molecule.atoms[selected_atom[2]].formal_charge
            atom_box.label(text=f"Selected atom: {selected_atom[2] + 1} ({state}, {charge:+d})")
            atom_box.operator(BLENDMOL_OT_change_atom_element.bl_idname, icon="MODIFIER")
            atom_box.operator(BLENDMOL_OT_set_formal_charge.bl_idname, icon="ADD")
            atom_box.operator(BLENDMOL_OT_toggle_fixed_atom.bl_idname, icon="PINNED")
            atom_box.operator(BLENDMOL_OT_delete_atom.bl_idname, icon="TRASH")
        layout.separator()
        layout.label(text="Select two atoms to create or change a bond")
        layout.operator(BLENDMOL_OT_set_bond.bl_idname, icon="CON_TRACKTO")
        selected_bond = _selected_explicit_bond(context)
        if selected_bond:
            layout.label(text=f"Selected bond: {selected_bond[2] + 1}")
            layout.operator(BLENDMOL_OT_delete_bond.bl_idname, icon="TRASH")
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
            validation_messages = _molecule_validation_messages(molecule)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            validation_messages = ["Stored molecule data cannot be validated"]
        validation_box = layout.box()
        if validation_messages:
            validation_box.alert = True
            validation_box.label(text="Valence warnings", icon="ERROR")
            for message in validation_messages[:4]:
                validation_box.label(text=message)
            if len(validation_messages) > 4:
                validation_box.label(text=f"… and {len(validation_messages) - 4} more")
            validation_box.alert = False
        else:
            validation_box.label(text="Valence: no over-valence warnings", icon="CHECKMARK")


class BLENDMOL_PT_constraints(Panel):
    bl_label = "UFF Constraints"
    bl_idname = "BLENDMOL_PT_constraints"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    @classmethod
    def poll(cls, context):
        return _selected_molecule_root(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        root = _selected_molecule_root(context)

        constraint_box = layout.box()
        constraint_box.label(text="Distance: select two atoms")
        constraint_box.prop(settings, "distance_constraint_target")
        constraint_box.prop(settings, "distance_constraint_strength")
        selected_pair = _selected_explicit_atoms(context)
        row = constraint_box.row(align=True)
        set_target = row.operator(
            BLENDMOL_OT_set_distance_constraint.bl_idname, text="Set Target"
        )
        set_target.use_current = False
        set_current = row.operator(
            BLENDMOL_OT_set_distance_constraint.bl_idname, text="Use Current"
        )
        set_current.use_current = True
        if selected_pair:
            pair = selected_pair[2]
            matching = next(
                (item for item in _distance_constraints(root) if item[:2] == pair), None
            )
            if matching:
                constraint_box.label(
                    text=f"Atoms {pair[0] + 1}-{pair[1] + 1}: {matching[2]:.3f} Å"
                )
                remove = constraint_box.operator(
                    BLENDMOL_OT_remove_distance_constraint.bl_idname,
                    text="Remove Selected Constraint", icon="X",
                )
                remove.remove_all = False
        constraints = _distance_constraints(root)
        constraint_box.label(text=f"Stored constraints: {len(constraints)}")
        if constraints:
            remove_all = constraint_box.operator(
                BLENDMOL_OT_remove_distance_constraint.bl_idname,
                text="Remove All Constraints", icon="TRASH",
            )
            remove_all.remove_all = True

        angle_box = layout.box()
        angle_box.label(text="Angle: select 3 atoms; active atom is the center")
        angle_box.prop(settings, "angle_constraint_target")
        angle_box.prop(settings, "angle_constraint_strength")
        row = angle_box.row(align=True)
        set_angle = row.operator(
            BLENDMOL_OT_set_angle_constraint.bl_idname, text="Set Target"
        )
        set_angle.use_current = False
        current_angle = row.operator(
            BLENDMOL_OT_set_angle_constraint.bl_idname, text="Use Current"
        )
        current_angle.use_current = True
        selected_angle = _selected_angle_atoms(context)
        if selected_angle:
            key = selected_angle[2]
            matching_angle = next(
                (item for item in _angle_constraints(root) if item[:3] == key), None
            )
            if matching_angle:
                angle_box.label(
                    text=(
                        f"Atoms {key[0] + 1}-{key[1] + 1}-{key[2] + 1}: "
                        f"{math.degrees(matching_angle[3]):.2f}°"
                    )
                )
                remove_angle = angle_box.operator(
                    BLENDMOL_OT_remove_angle_constraint.bl_idname,
                    text="Remove Selected Constraint", icon="X",
                )
                remove_angle.remove_all = False
        angle_constraints = _angle_constraints(root)
        angle_box.label(text=f"Stored constraints: {len(angle_constraints)}")
        if angle_constraints:
            remove_all_angles = angle_box.operator(
                BLENDMOL_OT_remove_angle_constraint.bl_idname,
                text="Remove All Constraints", icon="TRASH",
            )
            remove_all_angles.remove_all = True

        dihedral_box = layout.box()
        dihedral_box.label(text="Dihedral: select 4 atoms forming one bonded path")
        dihedral_box.prop(settings, "dihedral_constraint_target")
        dihedral_box.prop(settings, "dihedral_constraint_strength")
        row = dihedral_box.row(align=True)
        set_dihedral = row.operator(
            BLENDMOL_OT_set_dihedral_constraint.bl_idname, text="Set Target"
        )
        set_dihedral.use_current = False
        current_dihedral = row.operator(
            BLENDMOL_OT_set_dihedral_constraint.bl_idname, text="Use Current"
        )
        current_dihedral.use_current = True
        selected_dihedral = _selected_dihedral_atoms(context)
        if selected_dihedral:
            path = selected_dihedral[2]
            matching_dihedral = next(
                (item for item in _dihedral_constraints(root) if item[:4] == path), None
            )
            if matching_dihedral:
                dihedral_box.label(
                    text=(
                        f"Atoms {'-'.join(str(index + 1) for index in path)}: "
                        f"{math.degrees(matching_dihedral[4]):.2f}°"
                    )
                )
                remove_dihedral = dihedral_box.operator(
                    BLENDMOL_OT_remove_dihedral_constraint.bl_idname,
                    text="Remove Selected Constraint", icon="X",
                )
                remove_dihedral.remove_all = False
        dihedral_constraints = _dihedral_constraints(root)
        dihedral_box.label(text=f"Stored constraints: {len(dihedral_constraints)}")
        if dihedral_constraints:
            remove_all_dihedrals = dihedral_box.operator(
                BLENDMOL_OT_remove_dihedral_constraint.bl_idname,
                text="Remove All Constraints", icon="TRASH",
            )
            remove_all_dihedrals.remove_all = True


class BLENDMOL_PT_uff(Panel):
    bl_label = "Partial UFF"
    bl_idname = "BLENDMOL_PT_uff"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_order = 4

    @classmethod
    def poll(cls, context):
        return _selected_molecule_root(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        root = _selected_molecule_root(context)
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            layout.alert = True
            layout.label(text="Stored molecule data cannot be read", icon="ERROR")
            return
        try:
            uff_report = type_molecule(molecule)
        except (IndexError, TypeError, ValueError):
            uff_report = None
        uff_box = layout
        if uff_report is None:
            uff_box.alert = True
            uff_box.label(text="Cannot inspect stored molecule", icon="ERROR")
            uff_box.alert = False
        elif uff_report.ready:
            counts: dict[str, int] = {}
            for atom_type in uff_report.types:
                counts[atom_type] = counts.get(atom_type, 0) + 1
            summary = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
            uff_box.label(text="Atom typing: ready", icon="CHECKMARK")
            uff_box.label(text=summary)
            uff_box.prop(settings, "show_uff_energy_breakdown", toggle=True, icon="ALIGN_JUSTIFY")
            if settings.show_uff_energy_breakdown:
                breakdown_box = uff_box.box()
                stretch = bond_stretch_energy(molecule)
                if stretch.ready:
                    breakdown_box.label(
                        text=f"Bond stretch: {stretch.energy_kcal_mol:.3f} kcal/mol ({stretch.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="Bond stretch: needs explicit hydrogens", icon="INFO")
                angle = angle_bend_energy(molecule)
                if angle.ready:
                    breakdown_box.label(
                        text=f"Angle bend: {angle.energy_kcal_mol:.3f} kcal/mol ({angle.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="Angle bend: needs explicit hydrogens", icon="INFO")
                torsion = torsion_energy(molecule)
                if torsion.ready:
                    breakdown_box.label(
                        text=f"Torsion: {torsion.energy_kcal_mol:.3f} kcal/mol ({torsion.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="Torsion: needs explicit hydrogens", icon="INFO")
                inversion = out_of_plane_energy(molecule)
                if inversion.ready:
                    breakdown_box.label(
                        text=f"Out of plane: {inversion.energy_kcal_mol:.3f} kcal/mol ({inversion.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="Out of plane: needs explicit hydrogens", icon="INFO")
                vdw = van_der_waals_energy(molecule)
                if vdw.ready:
                    breakdown_box.label(
                        text=f"van der Waals: {vdw.energy_kcal_mol:.3f} kcal/mol ({vdw.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="van der Waals: needs explicit hydrogens", icon="INFO")
            total = partial_energy(molecule)
            if total.ready:
                uff_box.label(text=f"Partial UFF total: {total.energy_kcal_mol:.3f} kcal/mol")
                settings_box = uff_box.box()
                settings_box.prop(settings, "uff_iterations")
                settings_box.prop(settings, "uff_max_step")
                settings_box.prop(settings, "uff_gradient_tolerance")
                settings_box.prop(settings, "uff_energy_tolerance")
                settings_box.prop(settings, "uff_optimizer")
                settings_box.prop(settings, "uff_backend")
                compute_status = backend_status()
                settings_box.label(
                    text=f"Compute backend: {compute_status['active'].title()}",
                    icon="CHECKMARK" if compute_status["native_available"] else "INFO",
                )
                minimize_row = uff_box.row()
                minimize_row.scale_y = 1.3
                minimize_row.operator(BLENDMOL_OT_minimize_partial_uff.bl_idname, icon="MOD_SMOOTH")
                live_box = uff_box.box()
                live_box.label(text="Interactive UFF relaxation", icon="TRACKING")
                live_running = bool(
                    STATE.operator is not None
                    and STATE.operator._root_name == root.name
                )
                live_row = live_box.row()
                live_row.scale_y = 1.5
                live_row.operator(
                    BLENDMOL_OT_toggle_live_uff.bl_idname,
                    text="Stop Live UFF" if live_running else "Start Live UFF",
                    icon="PAUSE" if live_running else "PLAY",
                )
                if live_running:
                    phase = root.get("blend_mol_live_uff_phase", "ready")
                    live_box.label(text=f"Live state: {phase.title()}")
                if "blend_mol_live_uff_error" in root:
                    # Live UFF failures (e.g. the collinear-angle error) are
                    # also reported via self.report when they happen, but
                    # that status-bar message is easy to miss mid-drag; keep
                    # it visible here until the next Start Live UFF clears it.
                    live_box.label(text=root["blend_mol_live_uff_error"], icon="ERROR")
                    error_row = live_box.row()
                    error_row.prop(settings, "jitter_magnitude")
                    error_row.operator(BLENDMOL_OT_jitter_atom_positions.bl_idname, icon="MOD_NOISE")
                live_box.prop(settings, "uff_live_auto_start")
                if "blend_mol_uff_live_last" in root:
                    try:
                        live_run = json.loads(root["blend_mol_uff_live_last"])
                        live_box.label(
                            text=(
                                f"Atom {live_run['held_atom'] + 1}: "
                                f"{live_run['initial_energy_kcal_mol']:.3f} → "
                                f"{live_run['final_energy_kcal_mol']:.3f} kcal/mol"
                            )
                        )
                        if "display_update_ms" in live_run:
                            live_box.label(
                                text=(
                                    f"Compute {live_run['elapsed_ms']:.1f} ms + display "
                                    f"{live_run['display_update_ms']:.1f} ms = "
                                    f"{live_run['total_step_ms']:.1f} ms"
                                )
                            )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        live_box.label(text="Last interactive step log is unreadable", icon="ERROR")
                live_box.operator(BLENDMOL_OT_relax_around_selected_atom.bl_idname, icon="TRACKING")
                live_settings_box = live_box.box()
                live_settings_box.label(text="Live UFF settings")
                live_settings_box.prop(settings, "uff_live_steps")
                live_settings_box.prop(settings, "uff_live_target_ms")
                live_settings_box.prop(settings, "uff_live_settle_updates")
                live_settings_box.prop(settings, "uff_live_interval")
                live_settings_box.separator()
                live_settings_box.operator(BLENDMOL_OT_benchmark_uff_step.bl_idname, icon="TIME")
                if "blend_mol_uff_benchmark" in root:
                    try:
                        benchmark = json.loads(root["blend_mol_uff_benchmark"])
                        live_settings_box.label(
                            text=(
                                f"{benchmark['atom_count']} atoms / {benchmark['moving_atoms']} moving: "
                                f"{benchmark['elapsed_ms']:.1f} ms"
                            )
                        )
                        live_settings_box.label(
                            text=f"Estimated compute rate: {benchmark['estimated_hz']:.1f} Hz"
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        live_settings_box.label(text="UFF benchmark log is unreadable", icon="ERROR")
            if "blend_mol_uff_last_run" in root:
                try:
                    last_run = json.loads(root["blend_mol_uff_last_run"])
                    log_box = uff_box.box()
                    log_box.label(text="Last partial UFF minimization")
                    log_box.label(text=last_run["timestamp"])
                    log_box.label(
                        text=(
                            f"{last_run['initial_energy_kcal_mol']:.3f} → "
                            f"{last_run['final_energy_kcal_mol']:.3f} kcal/mol"
                        )
                    )
                    if (
                        "blend_mol_uff_before_positions" in root
                        and "blend_mol_uff_after_positions" in root
                    ):
                        row = log_box.row(align=True)
                        before = row.operator(
                            BLENDMOL_OT_restore_uff_coordinates.bl_idname,
                            text="Show Before", icon="LOOP_BACK",
                        )
                        before.state = "BEFORE"
                        after = row.operator(
                            BLENDMOL_OT_restore_uff_coordinates.bl_idname,
                            text="Show After", icon="CHECKMARK",
                        )
                        after.state = "AFTER"
                    log_box.label(
                        text=(
                            f"{last_run['iterations']} accepted steps; "
                            f"{last_run['termination']}; {last_run['fixed_atoms']} fixed"
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    uff_box.label(text="Last UFF log is unreadable", icon="ERROR")
            uff_box.label(text="No electrostatic terms; minimizer uses implemented terms only")
        else:
            uff_box.alert = True
            uff_box.label(text="Not ready for UFF", icon="ERROR")
            for message in uff_report.messages[:3]:
                uff_box.label(text=message)
            if len(uff_report.messages) > 3:
                uff_box.label(text=f"… and {len(uff_report.messages) - 3} more")
            uff_box.alert = False


class BLENDMOL_PT_relax(Panel):
    bl_label = "Display Relaxation & Snapshot"
    bl_idname = "BLENDMOL_PT_relax"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    @classmethod
    def poll(cls, context):
        return _selected_molecule_root(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        root = _selected_molecule_root(context)
        layout.label(text="Simple display relaxation (not UFF)")
        layout.prop(settings, "relax_iterations")
        layout.operator(BLENDMOL_OT_relax_display_geometry.bl_idname, icon="MOD_SMOOTH")
        layout.operator(BLENDMOL_OT_save_position_snapshot.bl_idname, icon="BOOKMARKS")
        if "blend_mol_position_snapshot" in root:
            layout.operator(BLENDMOL_OT_restore_position_snapshot.bl_idname, icon="LOOP_BACK")
        layout.separator()
        layout.label(text="Stuck on an exactly-degenerate geometry (e.g. collinear)?")
        jitter_row = layout.row()
        jitter_row.prop(settings, "jitter_magnitude")
        jitter_row.operator(BLENDMOL_OT_jitter_atom_positions.bl_idname, icon="MOD_NOISE")


class BLENDMOL_PT_display(Panel):
    bl_label = "Display & Mesh Settings"
    bl_idname = "BLENDMOL_PT_display"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 6

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        elements = _managed_atom_elements()
        if elements:
            element_box = layout.box()
            element_box.label(text="Elements in Blend Mol")
            for element, count in sorted(elements.items()):
                item = _palette_item(settings, element)
                element_box.prop(item, "color", text=f"{element} ({count})")
            element_box.operator(BLENDMOL_OT_reset_element_colors.bl_idname, icon="LOOP_BACK")
            layout.separator()
        layout.label(text="Display defaults for next import")
        layout.prop(settings, "atom_scale")
        layout.prop(settings, "bond_radius")
        layout.prop(settings, "show_implicit_hydrogens")
        layout.operator(BLENDMOL_OT_rebuild_display.bl_idname, icon="FILE_REFRESH")
        layout.separator()
        layout.label(text="Material (applies immediately)")
        layout.prop(settings, "roughness")
        layout.prop(settings, "metallic")
        layout.separator()
        mesh_box = layout.box()
        mesh_box.label(text="Mesh detail")
        mesh_box.label(text="Defaults for next import")
        mesh_box.prop(settings, "atom_segments")
        mesh_box.prop(settings, "atom_rings")
        mesh_box.prop(settings, "bond_sides")
        mesh_box.prop(settings, "bond_length_segments")
        mesh_box.separator()
        mesh_box.prop(settings, "subdivision_levels")
        mesh_box.operator(BLENDMOL_OT_add_subdivision.bl_idname, icon="MOD_SUBSURF")
