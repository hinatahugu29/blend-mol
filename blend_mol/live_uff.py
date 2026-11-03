"""Live UFF: the modal operator, automatic start on atom moves, and its shared state."""

from __future__ import annotations

import json
from datetime import datetime
from threading import Event
from time import perf_counter

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty
from bpy.types import Operator

from .display import _update_explicit_display_geometry
from .live_worker import adapt_chunk_iterations, LatestOnlyWorker, should_continue_relaxation
from .ops_edit import _ensure_hydrogens_explicit_for_uff
from .scene_data import (
    _angle_constraints, _apply_serialized_positions, _deserialize_molecule,
    _dihedral_constraints, _distance_constraints, _fixed_atom_indices, _molecule_collection,
    _molecule_revision, _selected_explicit_atom, _serialize_molecule, _serialize_positions,
    _store_live_uff_session, _sync_atom_positions, _topology_signature,
)
from .uff import (
    _gradient_step_metrics, minimize_partial_energy, partial_objective_energy_gradient,
    prepare_uff_context,
)


class LiveUffState:
    """Live UFF state shared by the operator, the handlers and the panels.

    Kept on one object so other modules (and the smoke tests) read and write
    the same values, never a copy made by ``from ... import``.
    """

    def __init__(self):
        # The running BLENDMOL_OT_toggle_live_uff instance, if any.
        self.operator = None
        # True while a deferred automatic start is queued on a timer.
        self.auto_timer_pending = False
        # (root name, atom object name, atom index) for that queued start.
        self.auto_target = None
        # (root name, atom index) -> perf_counter() time of the last automatic
        # start that ended in error; see AUTO_START_ERROR_COOLDOWN_SECONDS.
        self.error_cooldown: dict[tuple[str, int], float] = {}


STATE = LiveUffState()

# How long, after an auto-started Live UFF session ends in error (e.g. the
# collinear-angle error), to hold off auto-restarting it for the same atom.
# Without this, continuing to drag an atom through a degenerate geometry
# re-triggers the depsgraph auto-start handler on every further micro-
# movement, each attempt failing again -- a flood of repeated error reports
# that fights the user's own grab/cancel/confirm instead of just naming the
# problem once.
AUTO_START_ERROR_COOLDOWN_SECONDS = 1.5


def _explicit_atom_has_moved(root, atom_object, atom_index):
    molecule = _deserialize_molecule(root["blend_mol_data"])
    position = atom_object.matrix_local.translation
    return any(abs(position[axis] - molecule.atoms[atom_index].position[axis]) > 1e-7
               for axis in range(3))


@persistent
def _invalidate_live_uff_before_history_change(*_args):
    """Discard pending and in-flight work before Undo, Redo, or file loading."""
    if bpy.app.timers.is_registered(_start_auto_live_uff):
        bpy.app.timers.unregister(_start_auto_live_uff)
    STATE.auto_timer_pending = False
    STATE.auto_target = None
    STATE.error_cooldown.clear()
    if STATE.operator is not None:
        STATE.operator._finish(bpy.context, save_session=False)


def _start_auto_live_uff() -> None:
    """Start Live UFF outside the depsgraph callback after an atom transform."""
    target = STATE.auto_target
    STATE.auto_target = None
    STATE.auto_timer_pending = False
    if STATE.operator is not None:
        return None
    context = bpy.context
    scene = getattr(context, "scene", None)
    if scene is None or not scene.blend_mol_settings.uff_live_auto_start:
        return None
    selected = _selected_explicit_atom(context)
    if selected is None:
        return None
    root, _collection, atom_index = selected
    if target != (root.name, context.active_object.name, atom_index):
        return None
    last_error_time = STATE.error_cooldown.get((root.name, atom_index))
    if last_error_time is not None and perf_counter() - last_error_time < AUTO_START_ERROR_COOLDOWN_SECONDS:
        return None
    try:
        if not _explicit_atom_has_moved(root, context.active_object, atom_index):
            return None
        bpy.ops.blendmol.toggle_live_uff(
            "INVOKE_DEFAULT", start_immediately=True, auto_stop_when_ready=True,
        )
    except (RuntimeError, KeyError, IndexError, TypeError, ValueError, ReferenceError):
        # Blender can legitimately have no suitable window context while loading
        # a file or changing workspaces. The next transform queues a fresh start.
        pass
    return None


@persistent
def _queue_auto_live_uff_from_transform(scene, depsgraph) -> None:
    """Defer automatic Live UFF startup until Blender finishes a transform update."""
    if STATE.operator is not None or STATE.auto_timer_pending:
        return
    settings = getattr(scene, "blend_mol_settings", None)
    if settings is None or not settings.uff_live_auto_start:
        return
    active = bpy.context.view_layer.objects.active if bpy.context.view_layer else None
    if active is None or active.get("blend_mol_kind") != "atom" or active.get("blend_mol_implicit"):
        return
    for update in depsgraph.updates:
        # ``update.id`` is the depsgraph's evaluated copy, which never equals
        # the original data-block even without modifiers; compare ``.original``.
        updated_id = getattr(update.id, "original", update.id)
        if updated_id == active and update.is_updated_transform:
            selected = _selected_explicit_atom(bpy.context)
            if selected is None:
                return
            root, _collection, atom_index = selected
            try:
                if not _explicit_atom_has_moved(root, active, atom_index):
                    return
            except (KeyError, IndexError, TypeError, ValueError, ReferenceError):
                return
            STATE.auto_target = (root.name, active.name, atom_index)
            STATE.auto_timer_pending = True
            bpy.app.timers.register(_start_auto_live_uff, first_interval=0.0)
            return


class BLENDMOL_OT_toggle_live_uff(Operator):
    bl_idname = "blendmol.toggle_live_uff"
    bl_label = "Toggle Live UFF"
    bl_description = "Start or stop repeated partial UFF relaxation while the selected atom is moved"

    _timer = None
    _root_name = ""
    _atom_index = -1
    _last_position = None
    _stop_requested = False
    _worker = None
    _topology_revision = 0
    _topology_at_start = None
    _prepared_uff = None
    _initial_positions = None
    _initial_live_log = None
    _settle_updates = 0
    _applied_results = 0
    _cancel_event = None
    _adaptive_steps = 1
    _optimizer_state = None

    start_immediately: BoolProperty(options={"HIDDEN"}, default=False)
    auto_stop_when_ready: BoolProperty(options={"HIDDEN"}, default=False)

    def _set_live_error(self, root: bpy.types.Object, message: str) -> None:
        """Record a Live UFF failure on root AND surface it to the user.

        Storing the message on root alone (the previous behaviour) left
        Live UFF failing silently: nothing in the Live UFF panel ever read
        that property, and this modal operator's own error paths never
        called self.report, so a failure (e.g. the collinear-angle error)
        just stopped the session with no visible sign anything went wrong.

        WARNING, not ERROR: Blender pops an attention-grabbing "Report:
        Error" box next to the mouse cursor for ERROR-level reports from a
        running modal operator, right on top of an in-progress grab -- it
        has to be dismissed with an extra click before the user's own grab
        can be confirmed or cancelled. WARNING still reaches the status bar
        and stays in the Partial UFF panel without that popup.
        """
        root["blend_mol_live_uff_error"] = message
        self.report({"WARNING"}, message)
        if self._root_name and self._atom_index >= 0:
            STATE.error_cooldown[(self._root_name, self._atom_index)] = perf_counter()

    def _cancel_and_restore(self, context):
        """Invalidate work first, then restore this session's coordinate snapshot."""
        self._finish(context, save_session=False)
        root = bpy.data.objects.get(self._root_name)
        if root is None or self._initial_positions is None:
            return
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
            if (_molecule_revision(root) != self._topology_revision
                    or _topology_signature(molecule) != self._topology_at_start):
                self._set_live_error(root, "Cannot restore Live UFF: molecule structure changed")
                return
            collection = _molecule_collection(root)
            if collection is None:
                return
            restored = _apply_serialized_positions(molecule, self._initial_positions)
            # Save before display changes: transform notifications must see the
            # restored coordinates, rather than queueing a fresh auto session.
            root["blend_mol_data"] = _serialize_molecule(restored)
            if not _update_explicit_display_geometry(
                root, collection, restored, context.scene.blend_mol_settings,
            ):
                self._set_live_error(root, "Cannot restore all Live UFF display objects")
                return
            if self._initial_live_log is None:
                if "blend_mol_uff_live_last" in root:
                    del root["blend_mol_uff_live_last"]
            else:
                root["blend_mol_uff_live_last"] = self._initial_live_log
            root["blend_mol_live_uff_phase"] = "cancelled"
        except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._set_live_error(root, f"Live UFF restore failed: {exc}")

    def _submit(self, molecule, root, settings, phase="calculating"):
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = Event()
        root["blend_mol_live_uff_phase"] = phase
        return self._worker.submit((
            molecule,
            self._adaptive_steps,
            settings.uff_max_step,
            frozenset(_fixed_atom_indices(root) | {self._atom_index}),
            _distance_constraints(root),
            _angle_constraints(root),
            _dihedral_constraints(root),
            self._prepared_uff,
            settings.uff_gradient_tolerance,
            self._cancel_event,
            settings.uff_energy_tolerance,
            settings.uff_optimizer,
            self._optimizer_state,
        ))

    def _finish(self, context, save_session=True):
        if self._stop_requested and self._timer is None and self._worker is None:
            return {"CANCELLED"}
        self._stop_requested = True
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._cancel_event = None
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._prepared_uff = None
        self._optimizer_state = None
        root = bpy.data.objects.get(self._root_name)
        if root is not None:
            root["blend_mol_live_uff_running"] = False
            root["blend_mol_live_uff_phase"] = "stopped"
            if save_session and self._applied_results and self._initial_positions is not None:
                try:
                    _store_live_uff_session(
                        root, self._initial_positions, self._topology_at_start,
                        self._atom_index, self._applied_results,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
        # A queued timer may finish this operator again. Never save a cancelled
        # session (or save a completed snapshot twice) on that second call.
        self._applied_results = 0
        if STATE.operator is self:
            STATE.operator = None
        return {"CANCELLED"}

    def invoke(self, context, event):
        if STATE.operator is not None:
            return STATE.operator._finish(context)
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select one explicit atom to manipulate first")
            return {"CANCELLED"}
        root, collection, atom_index = selected
        self._root_name = root.name
        self._atom_index = atom_index
        self._last_position = (
            None if self.start_immediately
            else tuple(context.active_object.matrix_local.translation)
        )
        self._stop_requested = False
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        molecule = _ensure_hydrogens_explicit_for_uff(
            root, collection, molecule, context.scene.blend_mol_settings,
        )
        self._topology_revision = _molecule_revision(root)
        self._topology_at_start = _topology_signature(molecule)
        self._prepared_uff = prepare_uff_context(molecule)
        self._initial_positions = _serialize_positions(molecule)
        self._initial_live_log = root.get("blend_mol_uff_live_last")
        self._settle_updates = 0
        self._applied_results = 0
        self._cancel_event = None
        self._adaptive_steps = context.scene.blend_mol_settings.uff_live_steps
        self._optimizer_state = None
        if "blend_mol_live_uff_error" in root:
            del root["blend_mol_live_uff_error"]
        self._worker = LatestOnlyWorker(
            lambda job: minimize_partial_energy(
                job[0], job[1], job[2], job[3], job[4], job[5], job[6], job[7], job[8],
                job[9].is_set, job[10], job[11], job[12],
            )
        )
        interval = context.scene.blend_mol_settings.uff_live_interval
        self._timer = context.window_manager.event_timer_add(interval, window=context.window)
        context.window_manager.modal_handler_add(self)
        root["blend_mol_live_uff_running"] = True
        root["blend_mol_live_uff_phase"] = "ready"
        STATE.operator = self
        self.report({"INFO"}, f"Live UFF started with atom {atom_index + 1} held")
        return {"RUNNING_MODAL"}

    def execute(self, context):
        return self.invoke(context, None)

    def modal(self, context, event):
        if self._stop_requested:
            return self._finish(context)
        if event.type == "ESC" or (event.type == "RIGHTMOUSE" and event.value == "PRESS"):
            self._cancel_and_restore(context)
            # Let Blender's active transform also receive its cancel event.
            return {"CANCELLED", "PASS_THROUGH"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._worker is None:
            return self._finish(context)
        root = bpy.data.objects.get(self._root_name)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            return self._finish(context)
        try:
            stored_molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._set_live_error(root, f"Stored molecule changed: {exc}")
            return self._finish(context)
        if (
            _molecule_revision(root) != self._topology_revision
            or _topology_signature(stored_molecule) != self._topology_at_start
        ):
            self._set_live_error(root, "Live UFF stopped because the molecule structure changed")
            return self._finish(context)
        atom_object = next(
            (obj for obj in collection.objects
             if obj.get("blend_mol_kind") == "atom"
             and not obj.get("blend_mol_implicit")
             and obj.get("blend_mol_atom_index") == self._atom_index),
            None,
        )
        if atom_object is None:
            return self._finish(context)
        current_position = tuple(atom_object.matrix_local.translation)
        moved = self._last_position is None or not all(
            abs(current_position[axis] - self._last_position[axis]) < 1e-7 for axis in range(3)
        )
        settings = context.scene.blend_mol_settings
        try:
            if moved:
                self._last_position = current_position
                molecule = _sync_atom_positions(root, collection)
                self._settle_updates = 0
                self._optimizer_state = None
                self._submit(molecule, root, settings)
            completion = self._worker.poll_latest()
            if completion is None:
                return {"PASS_THROUGH"}
            if completion.error is not None:
                raise completion.error
            result = completion.value
            self._optimizer_state = result.optimizer_state
            compute_ms = completion.elapsed_ms
            completed_chunk_steps = self._adaptive_steps
            self._adaptive_steps = adapt_chunk_iterations(
                self._adaptive_steps, compute_ms, settings.uff_live_target_ms,
            )
            if not result.ready:
                self._set_live_error(root, result.messages[0] if result.messages else "UFF step failed")
                return self._finish(context)
            root["blend_mol_data"] = _serialize_molecule(result.molecule)
            self._applied_results += 1
            display_started = perf_counter()
            if not _update_explicit_display_geometry(root, collection, result.molecule, settings):
                self._set_live_error(root, "Live display requires a fully explicit current mesh")
                return self._finish(context)
            display_ms = (perf_counter() - display_started) * 1000.0
            root["blend_mol_uff_live_last"] = json.dumps({
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "initial_energy_kcal_mol": result.initial_energy_kcal_mol,
                "final_energy_kcal_mol": result.final_energy_kcal_mol,
                "steps": result.iterations,
                "held_atom": self._atom_index,
                "elapsed_ms": compute_ms,
                "display_update_ms": display_ms,
                "total_step_ms": compute_ms + display_ms,
                "generation": completion.generation,
                "background": True,
                "settle_update": self._settle_updates,
                "gradient_tolerance": settings.uff_gradient_tolerance,
                "energy_tolerance": settings.uff_energy_tolerance,
                "optimizer": settings.uff_optimizer,
                "requested_steps": completed_chunk_steps,
                "next_adaptive_steps": self._adaptive_steps,
                "compute_budget_ms": settings.uff_live_target_ms,
            }, separators=(",", ":"))
            continue_settling = should_continue_relaxation(
                result.iterations, result.termination,
                self._settle_updates, settings.uff_live_settle_updates,
            )
            if not continue_settling and self._settle_updates < settings.uff_live_settle_updates:
                # A string-based termination reason ("energy tolerance", "line
                # search stalled") can be satisfied well before the gradient
                # itself is anywhere near uff_gradient_tolerance -- confirmed
                # by headless measurement after the 0.1.67 fix. Only the
                # gradient is authoritative, so check it directly before
                # trusting "ready" while settle budget remains.
                residual_fixed = frozenset(_fixed_atom_indices(root) | {self._atom_index})
                _residual_objective, residual_gradient, residual_messages = (
                    partial_objective_energy_gradient(
                        result.molecule, self._prepared_uff, residual_fixed,
                        _distance_constraints(root), _angle_constraints(root),
                        _dihedral_constraints(root), 1e-4,
                    )
                )
                if not residual_messages:
                    residual_max_component, _residual_max_norm = _gradient_step_metrics(
                        residual_gradient, residual_fixed
                    )
                    if residual_max_component >= settings.uff_gradient_tolerance:
                        continue_settling = True
            if continue_settling:
                self._settle_updates += 1
                self._submit(result.molecule, root, settings, "settling")
            else:
                root["blend_mol_live_uff_phase"] = "ready"
                if self.auto_stop_when_ready:
                    return self._finish(context)
        except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._set_live_error(root, str(exc))
            return self._finish(context)
        return {"PASS_THROUGH"}
