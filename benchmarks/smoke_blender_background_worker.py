"""Blender-side smoke test for add-on registration and background UFF work."""

import importlib
import math
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

addon = importlib.import_module("blend_mol")
from blend_mol.live_worker import LatestOnlyWorker, adapt_chunk_iterations
from blend_mol.molfile import read_v2000
from blend_mol.uff import (
    _gradient_step_metrics, _uff_angle_energy, minimize_partial_energy, out_of_plane_energy, partial_energy,
    partial_objective_energy_gradient,
    prepare_uff_context,
)


addon.register()
try:
    native_status = addon.backend_status()
    assert native_status["native_available"], native_status["detail"]
    stopped_operator = type("StoppedLiveUffOperator", (), {})()
    stopped_operator._timer = None
    stopped_operator._worker = None
    stopped_operator._cancel_event = None
    stopped_operator._root_name = ""
    stopped_operator._stop_requested = False
    stopped_operator._prepared_uff = None
    stopped_operator._optimizer_state = None
    stopped_operator._applied_results = 0
    stopped_operator._initial_positions = None
    stopped_operator._finish = lambda context: addon.BLENDMOL_OT_toggle_live_uff._finish(
        stopped_operator, context
    )
    assert stopped_operator._finish(addon.bpy.context) == {"CANCELLED"}
    assert stopped_operator._stop_requested
    late_timer = type("LateTimerEvent", (), {"type": "TIMER"})()
    assert addon.BLENDMOL_OT_toggle_live_uff.modal(
        stopped_operator, addon.bpy.context, late_timer
    ) == {"CANCELLED"}
    molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    root = addon._create_molecule(molecule, addon.bpy.context.scene.blend_mol_settings, "smoke")
    original_revision = addon._molecule_revision(root)
    moved_only = addon.Molecule(
        molecule.name,
        (addon.Atom(molecule.atoms[0].element, (9.0, 0.0, 0.0)),) + molecule.atoms[1:],
        molecule.bonds,
    )
    assert addon._topology_signature(moved_only) == addon._topology_signature(molecule)
    changed = addon.Molecule(
        molecule.name,
        (addon.Atom("N", molecule.atoms[0].position),) + molecule.atoms[1:],
        molecule.bonds,
    )
    assert addon._topology_signature(changed) != addon._topology_signature(molecule)
    addon._store_topology_change(root, changed)
    assert addon._molecule_revision(root) == original_revision + 1
    initial_positions = addon._serialize_positions(changed)
    assert addon._store_live_uff_session(
        root, initial_positions, addon._topology_signature(changed), 0, 3
    )
    session = addon.json.loads(root["blend_mol_live_uff_session"])
    assert session["applied_updates"] == 3
    initial_distance = math.dist(molecule.atoms[0].position, molecule.atoms[10].position)
    target_distance = initial_distance + 0.25
    addon._store_distance_constraints(root, ((0, 10, target_distance, 5000.0),))
    assert addon._distance_constraints(root) == ((0, 10, target_distance, 5000.0),)
    constrained = minimize_partial_energy(
        molecule, 40, 0.03, {0}, addon._distance_constraints(root)
    )
    final_distance = math.dist(
        constrained.molecule.atoms[0].position, constrained.molecule.atoms[10].position
    )
    assert abs(final_distance - target_distance) < abs(initial_distance - target_distance)
    angle_term = (0, 1, 2, math.radians(110.0), 1000.0)
    addon._store_angle_constraints(root, (angle_term,))
    stored_angle = addon._angle_constraints(root)
    assert len(stored_angle) == 1
    assert stored_angle[0][:3] == angle_term[:3]
    assert math.isclose(stored_angle[0][3], angle_term[3])
    angle_constrained = minimize_partial_energy(
        molecule, 10, 0.02, {1}, (), stored_angle
    )
    assert angle_constrained.ready
    dihedral_term = (0, 1, 2, 3, math.radians(10.0), 500.0)
    addon._store_dihedral_constraints(root, (dihedral_term,))
    stored_dihedral = addon._dihedral_constraints(root)
    assert len(stored_dihedral) == 1
    assert stored_dihedral[0][:4] == dihedral_term[:4]
    dihedral_constrained = minimize_partial_energy(
        molecule, 2, 0.02, {1, 2}, (), (), stored_dihedral
    )
    assert dihedral_constrained.ready
    prepared = prepare_uff_context(molecule)
    assert adapt_chunk_iterations(2, 100.0, 50.0, 1.0) == 1
    assert adapt_chunk_iterations(2, 25.0, 50.0, 1.0) == 4
    assert len(prepared.neighbors) == len(molecule.atoms)
    assert partial_energy(molecule, prepared.typing, prepared).energy_kcal_mol == partial_energy(
        molecule, prepared.typing
    ).energy_kcal_mol
    objective_energy, objective_gradient, objective_messages = partial_objective_energy_gradient(
        molecule, prepared, {0}
    )
    assert not objective_messages
    assert math.isclose(objective_energy, partial_energy(molecule).energy_kcal_mol)
    assert objective_gradient[0] == (0.0, 0.0, 0.0)
    assert _gradient_step_metrics([[3.0, 4.0, 0.0], [100.0, 0.0, 0.0]], {1}) == (4.0, 5.0)
    assert out_of_plane_energy(molecule).terms == 54
    assert _uff_angle_energy(100.0, math.radians(120.0), math.radians(120.0), "C_R", 3) > 0.0
    tolerance_result = minimize_partial_energy(
        molecule, 5, 0.02, prepared=prepared, gradient_tolerance=1e20
    )
    assert tolerance_result.termination == "gradient tolerance"
    assert tolerance_result.iterations == 0
    energy_tolerance_result = minimize_partial_energy(
        molecule, 5, 0.02, prepared=prepared, energy_tolerance=1e20
    )
    assert energy_tolerance_result.termination == "energy tolerance"
    assert energy_tolerance_result.iterations == 1
    cancelled_result = minimize_partial_energy(
        molecule, 50, 0.02, prepared=prepared, cancelled=lambda: True
    )
    assert cancelled_result.termination == "cancelled"
    assert cancelled_result.iterations == 0
    fire_first = minimize_partial_energy(
        molecule, 4, 0.02, prepared=prepared, algorithm="FIRE2"
    )
    fire_second = minimize_partial_energy(
        fire_first.molecule, 4, 0.02, prepared=prepared, algorithm="FIRE2",
        optimizer_state=fire_first.optimizer_state,
    )
    assert fire_first.ready and fire_second.ready
    assert fire_second.optimizer_state.initialized
    assert len(fire_second.optimizer_state.velocity) == len(molecule.atoms)
    lbfgs_first = minimize_partial_energy(
        molecule, 4, 0.02, prepared=prepared, algorithm="LBFGS"
    )
    lbfgs_second = minimize_partial_energy(
        lbfgs_first.molecule, 4, 0.02, prepared=prepared, algorithm="LBFGS",
        optimizer_state=lbfgs_first.optimizer_state,
    )
    assert lbfgs_first.ready and lbfgs_second.ready
    assert lbfgs_second.optimizer_state.initialized
    assert len(lbfgs_second.optimizer_state.history_s) <= 7
    worker = LatestOnlyWorker(
        lambda item: minimize_partial_energy(item, 1, 0.02, frozenset(), prepared=prepared)
    )
    completion = None
    for _cycle in range(50):
        generation = worker.submit(molecule)
        deadline = time.monotonic() + 20.0
        completion = None
        while time.monotonic() < deadline and completion is None:
            completion = worker.poll_latest()
            time.sleep(0.005)
        assert completion is not None, "background UFF timed out"
        assert completion.generation == generation
        assert completion.error is None, completion.error
        assert completion.value.ready, completion.value.messages
    worker.stop(wait=True)
    assert not any(thread.name == "BlendMolLiveUFF" for thread in threading.enumerate())
    print(
        "BLEND_MOL_BACKGROUND_SMOKE_OK",
        "50_cycles",
        "revision_ok",
        "distance_constraint_ok",
        "angle_constraint_ok",
        "dihedral_constraint_ok",
        "prepared_context_ok",
        "topology_cache_ok",
        "adaptive_chunk_ok",
        "objective_boundary_ok",
        "per_atom_step_ok",
        "three_oop_terms_ok",
        "planar_angle_repulsion_ok",
        "session_snapshot_ok",
        "tolerance_ok",
        "energy_tolerance_ok",
        "cancellation_ok",
        "stateful_fire2_ok",
        "stateful_lbfgs_ok",
        "late_stop_timer_safe",
        "packaged_rust_dll_loaded",
        len(molecule.atoms),
        completion.elapsed_ms,
        completion.value.final_energy_kcal_mol,
    )
finally:
    addon.unregister()
