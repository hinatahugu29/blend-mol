"""Blend Mol: molecular editing, visualization, and UFF optimization for Blender."""

from __future__ import annotations

import bpy
from bpy.props import PointerProperty

# Every submodule is imported so it is reachable as blend_mol.<module>.
from . import properties, scene_data, display, ops_io, ops_edit
from . import ops_display, ops_uff, ops_constraints, live_uff, panels
from .live_uff import (
    _invalidate_live_uff_before_history_change, _queue_auto_live_uff_from_transform,
    _start_auto_live_uff, BLENDMOL_OT_toggle_live_uff, STATE,
)
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
from .panels import (
    BLENDMOL_PT_constraints, BLENDMOL_PT_display, BLENDMOL_PT_edit, BLENDMOL_PT_file_io,
    BLENDMOL_PT_panel, BLENDMOL_PT_relax, BLENDMOL_PT_uff,
)
from .properties import BlendMolElementColor, BlendMolSettings


CLASSES = (
    BlendMolElementColor,
    BlendMolSettings,
    BLENDMOL_OT_import_v2000,
    BLENDMOL_OT_import_xyz,
    BLENDMOL_OT_import_pdb,
    BLENDMOL_OT_import_sdf,
    BLENDMOL_OT_import_smiles,
    BLENDMOL_OT_export_v2000,
    BLENDMOL_OT_export_xyz,
    BLENDMOL_OT_export_pdb,
    BLENDMOL_OT_export_sdf,
    BLENDMOL_OT_export_uff_report,
    BLENDMOL_OT_new_molecule,
    BLENDMOL_OT_rebuild_display,
    BLENDMOL_OT_sync_atom_positions,
    BLENDMOL_OT_add_atom,
    BLENDMOL_OT_add_bonded_atom,
    BLENDMOL_OT_make_hydrogens_explicit,
    BLENDMOL_OT_change_atom_element,
    BLENDMOL_OT_set_formal_charge,
    BLENDMOL_OT_delete_atom,
    BLENDMOL_OT_set_bond,
    BLENDMOL_OT_delete_bond,
    BLENDMOL_OT_relax_display_geometry,
    BLENDMOL_OT_jitter_atom_positions,
    BLENDMOL_OT_minimize_partial_uff,
    BLENDMOL_OT_relax_around_selected_atom,
    BLENDMOL_OT_benchmark_uff_step,
    BLENDMOL_OT_toggle_live_uff,
    BLENDMOL_OT_restore_uff_coordinates,
    BLENDMOL_OT_toggle_fixed_atom,
    BLENDMOL_OT_set_distance_constraint,
    BLENDMOL_OT_remove_distance_constraint,
    BLENDMOL_OT_set_angle_constraint,
    BLENDMOL_OT_remove_angle_constraint,
    BLENDMOL_OT_set_dihedral_constraint,
    BLENDMOL_OT_remove_dihedral_constraint,
    BLENDMOL_OT_save_position_snapshot,
    BLENDMOL_OT_restore_position_snapshot,
    BLENDMOL_OT_reset_element_colors,
    BLENDMOL_OT_add_subdivision,
    BLENDMOL_PT_panel,
    BLENDMOL_PT_file_io,
    BLENDMOL_PT_edit,
    BLENDMOL_PT_constraints,
    BLENDMOL_PT_uff,
    BLENDMOL_PT_relax,
    BLENDMOL_PT_display,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.blend_mol_settings = PointerProperty(type=BlendMolSettings)
    if _queue_auto_live_uff_from_transform not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_queue_auto_live_uff_from_transform)
    for handlers in (bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre, bpy.app.handlers.load_pre):
        if _invalidate_live_uff_before_history_change not in handlers:
            handlers.append(_invalidate_live_uff_before_history_change)


def unregister():
    if STATE.operator is not None:
        STATE.operator._finish(bpy.context)
    _invalidate_live_uff_before_history_change()
    for handlers in (bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre, bpy.app.handlers.load_pre):
        if _invalidate_live_uff_before_history_change in handlers:
            handlers.remove(_invalidate_live_uff_before_history_change)
    if _queue_auto_live_uff_from_transform in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_queue_auto_live_uff_from_transform)
    if bpy.app.timers.is_registered(_start_auto_live_uff):
        bpy.app.timers.unregister(_start_auto_live_uff)
    STATE.auto_timer_pending = False
    del bpy.types.Scene.blend_mol_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
