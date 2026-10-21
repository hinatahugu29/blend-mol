"""Blender-side smoke test for multi-gesture Live UFF auto-start and restore.

This exercises the real production path end to end for two consecutive
completed (not cancelled) auto Live UFF sessions: a real depsgraph update
triggers ``_queue_auto_live_uff_from_transform``, which defers to a timer
callback (``_start_auto_live_uff``, invoked directly here since a background
script never enters Blender's main loop to fire registered timers), which
starts the modal operator through ``bpy.ops`` exactly as interactive use
would. Each gesture is driven to completion by feeding the real ``modal``
method synthetic timer events and polling the real background worker.

It verifies that each completed gesture overwrites the stored before/after
snapshot with only that gesture's own start and end coordinates (operation
unit granularity), and that ``BLENDMOL_OT_restore_uff_coordinates`` restores
exactly one gesture at a time through the real operator, not an older one.

Real Blender Undo/Redo (Ctrl+Z / Ctrl+Shift+Z) round-tripping through the
window manager's own undo stack cannot be driven from this background
script: ``bpy.ops.ed.undo()`` and ``bpy.ops.ed.redo()`` both fail their
``poll()`` with "context is incorrect" whenever ``bpy.context.area`` is None,
which is always true in ``--background`` mode. That interaction remains a
manual check in the interactive GUI.
"""

import importlib
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

addon = importlib.import_module("blend_mol")
from blend_mol.molfile import read_v2000


def drive_gesture_to_completion(timeout_s: float = 30.0) -> None:
    """Feed synthetic TIMER events until the auto session finishes itself."""
    deadline = time.monotonic() + timeout_s
    timer_event = type("TimerEvent", (), {"type": "TIMER"})()
    while addon._LIVE_UFF_OPERATOR is not None:
        if time.monotonic() > deadline:
            raise AssertionError("Live UFF gesture did not auto-stop in time")
        operator = addon._LIVE_UFF_OPERATOR
        addon.BLENDMOL_OT_toggle_live_uff.modal(operator, addon.bpy.context, timer_event)
        time.sleep(0.005)


def trigger_real_auto_start(atom_object) -> None:
    """Exercise the real depsgraph handler, then fire its deferred timer."""
    assert not addon.bpy.app.timers.is_registered(addon._start_auto_live_uff)
    addon.bpy.context.view_layer.update()  # depsgraph_update_post runs synchronously
    assert addon._AUTO_LIVE_UFF_TIMER_PENDING, "depsgraph handler did not arm the auto-start timer"
    assert addon.bpy.app.timers.is_registered(addon._start_auto_live_uff)
    addon.bpy.app.timers.unregister(addon._start_auto_live_uff)
    addon._start_auto_live_uff()  # simulates Blender's main loop firing the timer
    assert addon._LIVE_UFF_OPERATOR is not None, "auto-start did not invoke the modal operator"


addon.register()
try:
    molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    settings = addon.bpy.context.scene.blend_mol_settings
    root = addon._create_molecule(molecule, settings, "gesture_smoke")
    collection = addon._molecule_collection(root)
    atom_object = next(
        obj for obj in collection.objects
        if obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and obj.get("blend_mol_atom_index") == 0
    )
    for obj in addon.bpy.context.selected_objects:
        obj.select_set(False)
    atom_object.select_set(True)
    addon.bpy.context.view_layer.objects.active = atom_object
    settings.uff_live_auto_start = True
    settings.uff_live_interval = 0.0
    original_position = tuple(molecule.atoms[0].position)

    # --- Gesture 1 -----------------------------------------------------
    atom_object.location.x += 0.15
    trigger_real_auto_start(atom_object)
    drive_gesture_to_completion()
    assert "blend_mol_live_uff_session" in root
    session_1 = json.loads(root["blend_mol_live_uff_session"])
    assert session_1["applied_updates"] >= 1
    before_1 = json.loads(root["blend_mol_uff_before_positions"])
    after_1 = json.loads(root["blend_mol_uff_after_positions"])
    assert math.dist(before_1[0], original_position) < 1e-6, (
        "gesture 1 before-snapshot must match the pre-drag coordinates"
    )
    assert math.dist(after_1[0], original_position) > 1e-3, (
        "gesture 1 after-snapshot must reflect the settled, moved coordinates"
    )
    gesture_1_final = tuple(addon._deserialize_molecule(root["blend_mol_data"]).atoms[0].position)

    # --- Gesture 2 (immediately after, same atom, further move) --------
    live_atom_object = next(
        obj for obj in collection.objects
        if obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and obj.get("blend_mol_atom_index") == 0
    )
    for obj in addon.bpy.context.selected_objects:
        obj.select_set(False)
    live_atom_object.select_set(True)
    addon.bpy.context.view_layer.objects.active = live_atom_object
    live_atom_object.location.y += 0.2
    trigger_real_auto_start(live_atom_object)
    drive_gesture_to_completion()
    session_2 = json.loads(root["blend_mol_live_uff_session"])
    before_2 = json.loads(root["blend_mol_uff_before_positions"])
    after_2 = json.loads(root["blend_mol_uff_after_positions"])

    # Operation-unit granularity: gesture 2's "before" is gesture 1's "after",
    # never gesture 1's own "before" or the original pre-gesture-1 position.
    assert math.dist(before_2[0], gesture_1_final) < 1e-6, (
        "gesture 2 before-snapshot must start exactly where gesture 1 settled"
    )
    assert math.dist(before_2[0], original_position) > 1e-3, (
        "gesture 2 before-snapshot must not fall back to the original coordinates"
    )
    assert math.dist(after_2[0], gesture_1_final) > 1e-3, (
        "gesture 2 after-snapshot must reflect further movement, not be a no-op"
    )

    # --- Restore operator must undo exactly one gesture at a time ------
    def _noop_report(_self, _flags, _message):
        pass

    restored_before = type("RestoreOp", (), {"state": "BEFORE", "report": _noop_report})()
    assert addon.BLENDMOL_OT_restore_uff_coordinates.execute(
        restored_before, addon.bpy.context
    ) == {"FINISHED"}
    restored_molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert math.dist(restored_molecule.atoms[0].position, gesture_1_final) < 1e-6, (
        "restoring BEFORE must land on gesture 2's start, i.e. gesture 1's settled result"
    )

    restored_after = type("RestoreOp", (), {"state": "AFTER", "report": _noop_report})()
    assert addon.BLENDMOL_OT_restore_uff_coordinates.execute(
        restored_after, addon.bpy.context
    ) == {"FINISHED"}
    restored_molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert math.dist(restored_molecule.atoms[0].position, after_2[0]) < 1e-6, (
        "restoring AFTER must return to gesture 2's settled result"
    )

    # --- Gesture 3: real auto-start, cancelled mid-flight via Escape -------
    # Before this session's fix, the real depsgraph handler never armed the
    # auto-start timer at all, so a cancelled *real* auto session could never
    # be exercised; only a manually constructed operator state was tested.
    pre_cancel_molecule = addon._deserialize_molecule(root["blend_mol_data"])
    pre_cancel_position = tuple(pre_cancel_molecule.atoms[0].position)
    cancel_atom_object = next(
        obj for obj in collection.objects
        if obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and obj.get("blend_mol_atom_index") == 0
    )
    for obj in addon.bpy.context.selected_objects:
        obj.select_set(False)
    cancel_atom_object.select_set(True)
    addon.bpy.context.view_layer.objects.active = cancel_atom_object
    cancel_atom_object.location.z += 0.25
    trigger_real_auto_start(cancel_atom_object)
    operator = addon._LIVE_UFF_OPERATOR
    assert operator is not None, "real auto-start must produce a running operator to cancel"
    escape_event = type("EscapeEvent", (), {"type": "ESC", "value": "PRESS"})()
    result = addon.BLENDMOL_OT_toggle_live_uff.modal(operator, addon.bpy.context, escape_event)
    assert result == {"CANCELLED", "PASS_THROUGH"}, result
    assert addon._LIVE_UFF_OPERATOR is None, "cancelling must clear the running operator"
    cancelled_molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert math.dist(cancelled_molecule.atoms[0].position, pre_cancel_position) < 1e-6, (
        "Escape on a real auto-started gesture must restore the pre-drag coordinates"
    )
    assert root["blend_mol_live_uff_phase"] == "cancelled"
    # A cancelled real gesture must not overwrite the previous completed
    # gesture's before/after snapshot used by the restore operator.
    assert json.loads(root["blend_mol_uff_before_positions"]) == before_2
    assert json.loads(root["blend_mol_uff_after_positions"]) == after_2

    # --- Gesture 4: several in-flight position updates within one drag -----
    # A real mouse drag reports many intermediate positions while the modal
    # operator is already running. Only the final position at auto-stop time
    # must be reflected; the depsgraph guard must not spawn extra sessions.
    multi_step_start = tuple(addon._deserialize_molecule(root["blend_mol_data"]).atoms[0].position)
    multi_step_atom_object = next(
        obj for obj in collection.objects
        if obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and obj.get("blend_mol_atom_index") == 0
    )
    for obj in addon.bpy.context.selected_objects:
        obj.select_set(False)
    multi_step_atom_object.select_set(True)
    addon.bpy.context.view_layer.objects.active = multi_step_atom_object
    multi_step_atom_object.location.x += 0.05
    trigger_real_auto_start(multi_step_atom_object)
    operator = addon._LIVE_UFF_OPERATOR
    assert operator is not None
    timer_event = type("TimerEvent", (), {"type": "TIMER"})()
    # One tick submits the first position; nudge further while it is running.
    addon.BLENDMOL_OT_toggle_live_uff.modal(operator, addon.bpy.context, timer_event)
    assert not addon._AUTO_LIVE_UFF_TIMER_PENDING, (
        "the depsgraph guard must not arm a second auto-start while one gesture runs"
    )
    multi_step_atom_object.location.x += 0.05
    addon.bpy.context.view_layer.update()
    assert not addon._AUTO_LIVE_UFF_TIMER_PENDING, (
        "moving the held atom again mid-gesture must not spawn a second session"
    )
    final_multi_step_target = tuple(multi_step_atom_object.matrix_local.translation)
    drive_gesture_to_completion()
    multi_step_final = addon._deserialize_molecule(root["blend_mol_data"]).atoms[0].position
    assert math.dist(multi_step_final, multi_step_start) > 1e-3, (
        "the multi-step gesture must still have moved the held atom overall"
    )
    held_atom_only_moved_by_user = math.dist(
        tuple(final_multi_step_target), multi_step_start
    )
    assert held_atom_only_moved_by_user > 0.05, "sanity: the user-driven displacement must be real"

    # --- Documented automation boundary: real WM undo/redo need a window --
    undo_blocked = False
    try:
        addon.bpy.ops.ed.undo()
    except RuntimeError as exc:
        undo_blocked = "poll" in str(exc)
    assert undo_blocked, (
        "expected bpy.ops.ed.undo() to refuse to run without an interactive "
        "window/area context in background mode; if this now succeeds, real "
        "Undo/Redo round-tripping should be added to this headless suite"
    )

    print(
        "BLEND_MOL_LIVE_UFF_GESTURE_SMOKE_OK",
        "real_depsgraph_to_modal_chain_ok",
        "two_sequential_gestures_ok",
        "per_gesture_snapshot_granularity_ok",
        "restore_before_after_granularity_ok",
        "real_auto_start_escape_cancel_ok",
        "mid_gesture_multi_step_no_duplicate_session_ok",
        "background_undo_redo_poll_boundary_confirmed",
        session_1["applied_updates"],
        session_2["applied_updates"],
    )
finally:
    addon.unregister()
