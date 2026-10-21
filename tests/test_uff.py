import unittest
import math
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))
from molfile import Atom, Bond, Molecule, read_v2000
from uff import (
    _angle_constraint_energy_gradient, _angle_energy_gradient, _bond_vdw_energy_gradient,
    _dihedral_angle, _dihedral_constraint_energy_gradient, _distance_constraint_energy_gradient,
    _gradient_step_metrics, _wrapped_angle_difference,
    _uff_angle_derivative, _uff_angle_energy,
    angle_bend_energy, bond_stretch_energy, minimize_partial_energy,
    out_of_plane_energy, partial_energy, partial_objective_energy_gradient,
    torsion_energy, type_molecule, van_der_waals_energy, UffOptimizerState,
)
from native_backend import set_backend_preference
import native_backend
import uff

class TestUffTyping(unittest.TestCase):
    def test_lbfgs_state_continues_across_chunks(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        first = minimize_partial_energy(molecule, 4, 0.01, algorithm="LBFGS")
        second = minimize_partial_energy(
            first.molecule, 4, 0.01, algorithm="LBFGS",
            optimizer_state=first.optimizer_state,
        )
        whole = minimize_partial_energy(molecule, 8, 0.01, algorithm="LBFGS")

        self.assertTrue(first.ready and second.ready and whole.ready)
        self.assertEqual(first.optimizer_state.algorithm, "LBFGS")
        self.assertLessEqual(len(first.optimizer_state.history_s), 7)
        self.assertEqual(
            len(first.optimizer_state.history_s), len(first.optimizer_state.history_y)
        )
        for chunked_atom, whole_atom in zip(second.molecule.atoms, whole.molecule.atoms):
            for chunked_value, whole_value in zip(chunked_atom.position, whole_atom.position):
                self.assertAlmostEqual(chunked_value, whole_value, places=9)
        self.assertLess(second.final_energy_kcal_mol, first.initial_energy_kcal_mol)

    def test_lbfgs_fixed_atom_cancellation_and_state_reset(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        incompatible = UffOptimizerState(
            "LBFGS", initialized=True, history_s=((1.0,),), history_y=((1.0,),)
        )
        result = minimize_partial_energy(
            molecule, 5, 0.01, {0}, algorithm="LBFGS",
            optimizer_state=incompatible,
        )
        self.assertTrue(result.ready)
        self.assertEqual(result.molecule.atoms[0].position, molecule.atoms[0].position)
        self.assertTrue(all(len(vector) == len(molecule.atoms) * 3 for vector in result.optimizer_state.history_s))
        cancelled = minimize_partial_energy(
            molecule, 5, 0.01, algorithm="LBFGS", cancelled=lambda: True
        )
        self.assertEqual(cancelled.termination, "cancelled")
        self.assertEqual(cancelled.iterations, 0)

    def test_fire2_state_continues_across_chunks(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        first = minimize_partial_energy(molecule, 5, 0.01, algorithm="FIRE2")
        second = minimize_partial_energy(
            first.molecule, 5, 0.01, algorithm="FIRE2",
            optimizer_state=first.optimizer_state,
        )
        whole = minimize_partial_energy(molecule, 10, 0.01, algorithm="FIRE2")

        self.assertTrue(first.ready and second.ready and whole.ready)
        self.assertTrue(first.optimizer_state.initialized)
        self.assertEqual(first.optimizer_state.algorithm, "FIRE2")
        self.assertEqual(len(first.optimizer_state.velocity), len(molecule.atoms))
        for chunked_atom, whole_atom in zip(second.molecule.atoms, whole.molecule.atoms):
            for chunked_value, whole_value in zip(chunked_atom.position, whole_atom.position):
                self.assertAlmostEqual(chunked_value, whole_value, places=10)
        self.assertLess(second.final_energy_kcal_mol, first.initial_energy_kcal_mol)

    def test_fire2_keeps_fixed_atom_and_velocity_zero(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        result = minimize_partial_energy(
            molecule, 8, 0.01, {0}, algorithm="FIRE2"
        )
        self.assertTrue(result.ready)
        self.assertEqual(result.molecule.atoms[0].position, molecule.atoms[0].position)
        self.assertEqual(result.optimizer_state.velocity[0], (0.0, 0.0, 0.0))

    def test_fire2_resets_incompatible_state_and_validates_algorithm(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        incompatible = UffOptimizerState("FIRE2", ((1.0, 2.0, 3.0),), initialized=True)
        result = minimize_partial_energy(
            molecule, 1, 0.01, algorithm="FIRE2", optimizer_state=incompatible
        )
        self.assertTrue(result.ready)
        self.assertEqual(len(result.optimizer_state.velocity), len(molecule.atoms))
        with self.assertRaisesRegex(ValueError, "optimizer algorithm"):
            minimize_partial_energy(molecule, 1, 0.01, algorithm="unknown")

    def test_fire2_cooperative_cancellation_preserves_state(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        result = minimize_partial_energy(
            molecule, 100, 0.02, algorithm="FIRE2", cancelled=lambda: True
        )
        self.assertTrue(result.ready)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.termination, "cancelled")
        self.assertIsNotNone(result.optimizer_state)

    def test_planar_angle_short_range_repulsion_has_matching_derivative(self):
        ideal = math.radians(120.0)
        expected_kcal = math.exp(-5.0) / 4.184
        self.assertAlmostEqual(_uff_angle_energy(100.0, ideal, ideal, "C_R", 3), expected_kcal)
        step = 1e-7
        finite_difference = (
            _uff_angle_energy(100.0, ideal + step, ideal, "C_R", 3)
            - _uff_angle_energy(100.0, ideal - step, ideal, "C_R", 3)
        ) / (2.0 * step)
        self.assertAlmostEqual(
            _uff_angle_derivative(100.0, ideal, ideal, "C_R", 3),
            finite_difference,
            places=6,
        )

    def test_trigonal_nitrogen_has_three_uff_inversion_terms(self):
        molecule = Molecule(
            "ammonia",
            (
                Atom("N", (0.0, 0.0, 0.25)),
                Atom("H", (1.0, 0.0, 0.0)),
                Atom("H", (-0.5, 0.8660254, 0.0)),
                Atom("H", (-0.5, -0.8660254, 0.0)),
            ),
            (Bond(0, 1, 1), Bond(0, 2, 1), Bond(0, 3, 1)),
        )
        typing = type_molecule(molecule)
        self.assertEqual(typing.types[0], "N_3")
        inversion = out_of_plane_energy(molecule, typing)
        self.assertTrue(inversion.ready)
        self.assertEqual(inversion.terms, 3)
        self.assertGreater(inversion.energy_kcal_mol, 0.0)

        set_backend_preference("python")
        python_result = minimize_partial_energy(molecule, 1, 0.005)
        try:
            set_backend_preference("rust")
            rust_result = minimize_partial_energy(molecule, 1, 0.005)
        finally:
            set_backend_preference("auto")
        self.assertTrue(python_result.ready and rust_result.ready)
        self.assertAlmostEqual(
            python_result.final_energy_kcal_mol, rust_result.final_energy_kcal_mol, places=7
        )

    def test_gradient_metrics_ignore_fixed_atoms_and_atom_steps_are_capped(self):
        self.assertEqual(_gradient_step_metrics([[3.0, 4.0, 0.0], [100.0, 0.0, 0.0]], {1}), (4.0, 5.0))
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        max_step = 0.01
        result = minimize_partial_energy(molecule, 1, max_step)
        self.assertTrue(result.ready)
        displacements = [
            math.dist(before.position, after.position)
            for before, after in zip(molecule.atoms, result.molecule.atoms)
        ]
        self.assertLessEqual(max(displacements), max_step + 1e-10)
        self.assertGreater(max(displacements), 0.0)

    def test_minimizer_cooperative_cancellation(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        result = minimize_partial_energy(
            molecule, 100, 0.02, cancelled=lambda: True
        )
        self.assertTrue(result.ready)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.termination, "cancelled")
        self.assertEqual(
            tuple(atom.position for atom in result.molecule.atoms),
            tuple(atom.position for atom in molecule.atoms),
        )

    def test_configurable_gradient_tolerance(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        result = minimize_partial_energy(molecule, 20, 0.02, gradient_tolerance=1e20)
        self.assertTrue(result.ready)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.termination, "gradient tolerance")
        with self.assertRaisesRegex(ValueError, "gradient tolerance"):
            minimize_partial_energy(molecule, 1, 0.02, gradient_tolerance=0.0)

    def test_configurable_energy_tolerance(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        result = minimize_partial_energy(molecule, 20, 0.02, energy_tolerance=1e20)
        self.assertTrue(result.ready)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.termination, "energy tolerance")
        with self.assertRaisesRegex(ValueError, "energy tolerance"):
            minimize_partial_energy(molecule, 1, 0.02, energy_tolerance=0.0)

    def test_prepared_context_reuses_typing_and_rejects_other_topology(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        original = uff.type_molecule
        with mock.patch.object(uff, "type_molecule", wraps=original) as typing:
            prepared = uff.prepare_uff_context(molecule)
            result = minimize_partial_energy(
                molecule, 3, 0.01, prepared=prepared
            )
        self.assertTrue(result.ready)
        self.assertEqual(typing.call_count, 1)

        changed = Molecule(
            molecule.name,
            (Atom("N", molecule.atoms[0].position),) + molecule.atoms[1:],
            molecule.bonds,
        )
        with self.assertRaisesRegex(ValueError, "different molecular topology"):
            minimize_partial_energy(changed, 1, 0.01, prepared=prepared)

    def test_prepared_topology_tables_match_uncached_evaluation(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        prepared = uff.prepare_uff_context(molecule)
        self.assertEqual(len(prepared.neighbors), len(molecule.atoms))
        self.assertTrue(prepared.vdw_pairs)

        atoms = list(molecule.atoms)
        moved = atoms[0]
        atoms[0] = Atom(
            moved.element,
            (moved.position[0] + 0.03, moved.position[1] - 0.02, moved.position[2] + 0.01),
            moved.formal_charge,
            moved.chirality,
        )
        displaced = Molecule(molecule.name, tuple(atoms), molecule.bonds)
        uncached = partial_energy(displaced, prepared.typing)
        cached = partial_energy(displaced, prepared.typing, prepared)
        self.assertTrue(uncached.ready and cached.ready)
        self.assertAlmostEqual(uncached.energy_kcal_mol, cached.energy_kcal_mol, places=10)

        for evaluator, table in (
            (_angle_energy_gradient, prepared.neighbors),
            (_bond_vdw_energy_gradient, prepared.vdw_pairs),
        ):
            uncached_energy, uncached_gradient, uncached_messages = evaluator(
                displaced, prepared.typing
            )
            cached_energy, cached_gradient, cached_messages = evaluator(
                displaced, prepared.typing, table
            )
            self.assertEqual(uncached_messages, cached_messages)
            self.assertAlmostEqual(uncached_energy, cached_energy, places=10)
            for uncached_vector, cached_vector in zip(uncached_gradient, cached_gradient):
                for uncached_component, cached_component in zip(uncached_vector, cached_vector):
                    self.assertAlmostEqual(uncached_component, cached_component, places=10)

    def test_objective_boundary_matches_energy_and_projects_fixed_atoms(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        prepared = uff.prepare_uff_context(molecule)
        constraint = ((0, 1, 1.7, 25.0),)
        constraint_energy, _constraint_gradient = _distance_constraint_energy_gradient(
            molecule, constraint
        )
        energy, gradient, messages = partial_objective_energy_gradient(
            molecule, prepared, {0}, constraint
        )
        self.assertFalse(messages)
        self.assertAlmostEqual(
            energy,
            partial_energy(molecule, prepared.typing, prepared).energy_kcal_mol
            + constraint_energy,
            places=8,
        )
        self.assertEqual(gradient[0], (0.0, 0.0, 0.0))
        self.assertTrue(any(abs(component) > 1e-8 for vector in gradient[1:] for component in vector))
        with self.assertRaisesRegex(ValueError, "fixed atom"):
            partial_objective_energy_gradient(molecule, prepared, {len(molecule.atoms)})

    def test_dihedral_constraint_wrap_gradient_and_minimization(self):
        self.assertAlmostEqual(
            math.degrees(_wrapped_angle_difference(math.radians(179), math.radians(-179))),
            -2.0,
        )
        molecule = Molecule(
            "dihedral",
            (Atom("H", (0.0, 1.0, 0.0)), Atom("H", (0.0, 0.0, 0.0)),
             Atom("H", (1.0, 0.0, 0.0)), Atom("H", (1.0, 1.0, 1.0))),
            (),
        )
        observed = _dihedral_angle(*(atom.position for atom in molecule.atoms))
        constraint = ((0, 1, 2, 3, observed + 0.2, 100.0),)
        energy, gradient = _dihedral_constraint_energy_gradient(molecule, constraint)
        step = 1e-6
        atoms = list(molecule.atoms)
        atoms[3] = Atom("H", (1.0, 1.0, 1.0 + step))
        higher, _ = _dihedral_constraint_energy_gradient(
            Molecule("dihedral", tuple(atoms), ()), constraint
        )
        atoms[3] = Atom("H", (1.0, 1.0, 1.0 - step))
        lower, _ = _dihedral_constraint_energy_gradient(
            Molecule("dihedral", tuple(atoms), ()), constraint
        )
        self.assertGreater(energy, 0.0)
        self.assertAlmostEqual(gradient[3][2], (higher - lower) / (2.0 * step), places=4)

        triphenylene = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        path = (0, 1, 2, 3)
        initial = _dihedral_angle(*(triphenylene.atoms[index].position for index in path))
        target = _wrapped_angle_difference(initial + math.radians(12.0), 0.0)
        result = minimize_partial_energy(
            triphenylene, 100, 0.02, {1, 2}, (), (),
            (path + (target, 5000.0),),
        )
        final = _dihedral_angle(*(result.molecule.atoms[index].position for index in path))
        self.assertTrue(result.ready)
        self.assertLess(
            abs(_wrapped_angle_difference(final, target)),
            abs(_wrapped_angle_difference(initial, target)),
        )

    def test_angle_constraint_gradient_and_minimization(self):
        molecule = Molecule(
            "angle",
            (Atom("H", (1.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.0)),
             Atom("H", (0.0, 1.0, 0.0))),
            (),
        )
        constraint = ((0, 1, 2, math.radians(60.0), 100.0),)
        energy, gradient = _angle_constraint_energy_gradient(molecule, constraint)
        step = 1e-6
        atoms = list(molecule.atoms)
        atoms[0] = Atom("H", (1.0, step, 0.0))
        higher, _ = _angle_constraint_energy_gradient(
            Molecule("angle", tuple(atoms), ()), constraint
        )
        atoms[0] = Atom("H", (1.0, -step, 0.0))
        lower, _ = _angle_constraint_energy_gradient(
            Molecule("angle", tuple(atoms), ()), constraint
        )
        self.assertGreater(energy, 0.0)
        self.assertAlmostEqual(gradient[0][1], (higher - lower) / (2.0 * step), places=5)

        triphenylene = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        first, center, second = 0, 1, 2
        initial_vectors = (
            tuple(triphenylene.atoms[first].position[i] - triphenylene.atoms[center].position[i] for i in range(3)),
            tuple(triphenylene.atoms[second].position[i] - triphenylene.atoms[center].position[i] for i in range(3)),
        )
        initial_angle = math.acos(sum(a * b for a, b in zip(*initial_vectors)) / (
            math.dist((0.0, 0.0, 0.0), initial_vectors[0])
            * math.dist((0.0, 0.0, 0.0), initial_vectors[1])
        ))
        target = initial_angle - math.radians(10.0)
        result = minimize_partial_energy(
            triphenylene, 80, 0.03, {center}, (),
            ((first, center, second, target, 5000.0),),
        )
        final_vectors = (
            tuple(result.molecule.atoms[first].position[i] - result.molecule.atoms[center].position[i] for i in range(3)),
            tuple(result.molecule.atoms[second].position[i] - result.molecule.atoms[center].position[i] for i in range(3)),
        )
        final_angle = math.acos(sum(a * b for a, b in zip(*final_vectors)) / (
            math.dist((0.0, 0.0, 0.0), final_vectors[0])
            * math.dist((0.0, 0.0, 0.0), final_vectors[1])
        ))
        self.assertTrue(result.ready)
        self.assertLess(abs(final_angle - target), abs(initial_angle - target))

    def test_distance_constraint_gradient_and_minimization(self):
        pair = Molecule(
            "pair", (Atom("H", (0.0, 0.0, 0.0)), Atom("H", (2.0, 0.0, 0.0))), ()
        )
        energy, gradient = _distance_constraint_energy_gradient(
            pair, ((0, 1, 1.5, 100.0),)
        )
        self.assertAlmostEqual(energy, 12.5)
        self.assertAlmostEqual(gradient[0][0], -50.0)
        self.assertAlmostEqual(gradient[1][0], 50.0)

        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        first, second = 0, 10
        initial_distance = math.dist(
            molecule.atoms[first].position, molecule.atoms[second].position
        )
        target = initial_distance + 0.4
        result = minimize_partial_energy(
            molecule, 80, 0.03, {first}, ((first, second, target, 5000.0),)
        )
        final_distance = math.dist(
            result.molecule.atoms[first].position, result.molecule.atoms[second].position
        )
        self.assertTrue(result.ready)
        self.assertEqual(result.molecule.atoms[first].position, molecule.atoms[first].position)
        self.assertLess(abs(final_distance - target), abs(initial_distance - target))

    def test_triphenylene_energy_matches_openbabel_uff_reference(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        conversion = 4.184
        # Open Babel 3 / Avogadro 2.0.0 reference values, in kJ/mol.
        reference = {
            "bond": 25.362,
            "angle": 20.977,
            "torsion": 0.435,
            "vdw": 237.076,
            "total": 283.85130,
        }
        observed = {
            "bond": bond_stretch_energy(molecule).energy_kcal_mol * conversion,
            "angle": angle_bend_energy(molecule).energy_kcal_mol * conversion,
            "torsion": torsion_energy(molecule).energy_kcal_mol * conversion,
            "vdw": van_der_waals_energy(molecule).energy_kcal_mol * conversion,
            "total": partial_energy(molecule).energy_kcal_mol * conversion,
        }
        tolerances = {"bond": 0.05, "angle": 1.0, "torsion": 0.2, "vdw": 0.5, "total": 1.5}
        for component, expected in reference.items():
            self.assertAlmostEqual(
                observed[component], expected, delta=tolerances[component], msg=component
            )

    def test_auto_falls_back_and_forced_rust_reports_missing_dll(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        saved_library = native_backend._LIBRARY
        saved_error = native_backend._LOAD_ERROR
        try:
            native_backend._LIBRARY = None
            native_backend._LOAD_ERROR = None
            set_backend_preference("auto")
            with mock.patch.object(native_backend.ctypes, "CDLL", side_effect=OSError("missing test DLL")):
                energy, _gradient, messages = _bond_vdw_energy_gradient(molecule)
            self.assertEqual(messages, ())
            self.assertIsInstance(energy, float)
            self.assertEqual(native_backend.backend_status()["active"], "python")
            set_backend_preference("rust")
            with self.assertRaisesRegex(RuntimeError, "requested but unavailable"):
                _bond_vdw_energy_gradient(molecule)
        finally:
            native_backend._LIBRARY = saved_library
            native_backend._LOAD_ERROR = saved_error
            set_backend_preference("auto")

    def test_rust_and_python_vdw_gradients_match(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        set_backend_preference("python")
        python_energy, python_gradient, python_messages = _bond_vdw_energy_gradient(molecule)
        set_backend_preference("rust")
        try:
            rust_energy, rust_gradient, rust_messages = _bond_vdw_energy_gradient(molecule)
        finally:
            set_backend_preference("auto")

        self.assertEqual(python_messages, ())
        self.assertEqual(rust_messages, ())
        self.assertAlmostEqual(python_energy, rust_energy, places=9)
        for python_atom, rust_atom in zip(python_gradient, rust_gradient):
            for python_value, rust_value in zip(python_atom, rust_atom):
                self.assertAlmostEqual(python_value, rust_value, places=8)

    def test_rust_and_python_one_step_minimization_match(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        set_backend_preference("python")
        python_result = minimize_partial_energy(molecule, 1, 0.02)
        set_backend_preference("rust")
        try:
            rust_result = minimize_partial_energy(molecule, 1, 0.02)
        finally:
            set_backend_preference("auto")

        self.assertTrue(python_result.ready)
        self.assertTrue(rust_result.ready)
        self.assertEqual(python_result.iterations, rust_result.iterations)
        self.assertAlmostEqual(
            python_result.final_energy_kcal_mol,
            rust_result.final_energy_kcal_mol,
            places=7,
        )
        for python_atom, rust_atom in zip(
            python_result.molecule.atoms, rust_result.molecule.atoms
        ):
            for python_value, rust_value in zip(python_atom.position, rust_atom.position):
                self.assertAlmostEqual(python_value, rust_value, places=8)

    def test_analytic_angle_gradient_matches_finite_difference(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        energy, gradient, messages = _angle_energy_gradient(molecule)
        step = 1e-5
        atoms = list(molecule.atoms)
        position = list(atoms[0].position)
        position[0] += step
        atoms[0] = Atom(atoms[0].element, tuple(position), atoms[0].formal_charge, atoms[0].chirality)
        higher = angle_bend_energy(Molecule(molecule.name, tuple(atoms), molecule.bonds))
        position[0] -= 2.0 * step
        atoms[0] = Atom(atoms[0].element, tuple(position), atoms[0].formal_charge, atoms[0].chirality)
        lower = angle_bend_energy(Molecule(molecule.name, tuple(atoms), molecule.bonds))
        finite_difference = (higher.energy_kcal_mol - lower.energy_kcal_mol) / (2.0 * step)

        self.assertEqual(messages, ())
        self.assertAlmostEqual(energy, angle_bend_energy(molecule).energy_kcal_mol, places=8)
        self.assertAlmostEqual(
            gradient[0][0], finite_difference,
            delta=max(1e-3, abs(finite_difference) * 1e-4),
        )

    def test_analytic_bond_and_vdw_gradient_matches_finite_difference(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
        energy, gradient, messages = _bond_vdw_energy_gradient(molecule)
        expected_energy = (
            bond_stretch_energy(molecule).energy_kcal_mol
            + van_der_waals_energy(molecule).energy_kcal_mol
        )
        step = 1e-5
        atoms = list(molecule.atoms)
        position = list(atoms[0].position)
        position[0] += step
        atoms[0] = Atom(atoms[0].element, tuple(position), atoms[0].formal_charge, atoms[0].chirality)
        higher, _gradient, higher_messages = _bond_vdw_energy_gradient(
            Molecule(molecule.name, tuple(atoms), molecule.bonds)
        )
        position[0] -= 2.0 * step
        atoms[0] = Atom(atoms[0].element, tuple(position), atoms[0].formal_charge, atoms[0].chirality)
        lower, _gradient, lower_messages = _bond_vdw_energy_gradient(
            Molecule(molecule.name, tuple(atoms), molecule.bonds)
        )
        finite_difference = (higher - lower) / (2.0 * step)

        self.assertEqual(messages, ())
        self.assertEqual(higher_messages, ())
        self.assertEqual(lower_messages, ())
        self.assertAlmostEqual(energy, expected_energy, places=8)
        self.assertAlmostEqual(
            gradient[0][0], finite_difference,
            delta=max(1e-3, abs(finite_difference) * 1e-4),
        )

    def test_types_saturated_neutral_phosphorus_and_sulfur(self):
        for element, hydrogen_count, expected_type in (("P", 3, "P_3+3"), ("S", 2, "S_3+2")):
            atoms = (Atom(element, (0.0, 0.0, 0.0)),) + tuple(
                Atom("H", (1.4, float(index), 0.3 * index)) for index in range(hydrogen_count)
            )
            molecule = Molecule(
                f"{element} hydride", atoms, tuple(Bond(0, index + 1, 1) for index in range(hydrogen_count))
            )
            typing = type_molecule(molecule)
            energy = bond_stretch_energy(molecule)

            self.assertTrue(typing.ready)
            self.assertEqual(typing.types[0], expected_type)
            self.assertTrue(energy.ready)
            self.assertTrue(math.isfinite(energy.energy_kcal_mol))

    def test_types_and_evaluates_explicit_monovalent_halogens(self):
        for element, expected_type in (("Cl", "Cl"), ("Br", "Br"), ("I", "I_")):
            molecule = Molecule(
                f"CH3{element}",
                (
                    Atom("C", (0.0, 0.0, 0.0)),
                    Atom("H", (1.09, 0.0, 0.0)),
                    Atom("H", (-0.36, 1.03, 0.0)),
                    Atom("H", (-0.36, -0.51, 0.89)),
                    Atom(element, (-1.2, -0.51, -0.89)),
                ),
                (Bond(0, 1, 1), Bond(0, 2, 1), Bond(0, 3, 1), Bond(0, 4, 1)),
            )

            typing = type_molecule(molecule)
            energy = bond_stretch_energy(molecule)

            self.assertTrue(typing.ready)
            self.assertEqual(typing.types[-1], expected_type)
            self.assertTrue(energy.ready)
            self.assertEqual(energy.terms, 4)
            self.assertTrue(math.isfinite(energy.energy_kcal_mol))

    def test_cumulated_double_bonds_type_as_sp_not_sp2(self):
        # A plain "highest bond order" rule would type CO2's carbon sp2
        # (max order 2), which linearizes to a 120 degree bend instead of
        # the correct 180 degree linear sp geometry. UFF/Open Babel reach
        # sp here because the carbon carries two separate double bonds.
        carbon_dioxide = Molecule(
            "CO2",
            (
                Atom("C", (0.0, 0.0, 0.0)),
                Atom("O", (1.3, 0.2, 0.0)),
                Atom("O", (-1.2, -0.3, 0.1)),
            ),
            (Bond(0, 1, 2), Bond(0, 2, 2)),
        )
        typing = type_molecule(carbon_dioxide)
        self.assertTrue(typing.ready)
        self.assertEqual(typing.types[0], "C_1")

        result = minimize_partial_energy(carbon_dioxide, 60, 0.05, frozenset())
        self.assertTrue(result.ready, result.messages)
        final = result.molecule
        center, outer_a, outer_b = (atom.position for atom in final.atoms)
        first = tuple(a - c for a, c in zip(outer_a, center))
        second = tuple(b - c for b, c in zip(outer_b, center))
        dot = sum(a * b for a, b in zip(first, second))
        length_a = math.sqrt(sum(a * a for a in first))
        length_b = math.sqrt(sum(b * b for b in second))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / (length_a * length_b)))))
        self.assertGreater(angle, 175.0)

        # An allene-style cumulated C=C=C centre must type the same way.
        allene = Molecule(
            "allene",
            (
                Atom("C", (0.0, 0.0, 0.0)),
                Atom("C", (1.3, 0.0, 0.0)),
                Atom("C", (-1.3, 0.0, 0.0)),
            ),
            (Bond(0, 1, 2), Bond(0, 2, 2)),
        )
        self.assertEqual(type_molecule(allene).types[0], "C_1")

        # A single, non-cumulated double bond must still type sp2 (e.g. the
        # carbonyl carbon in formaldehyde), so this fix must not regress the
        # common case by over-triggering on any order-2 bond.
        formaldehyde = Molecule(
            "CH2O",
            (
                Atom("C", (0.0, 0.0, 0.0)),
                Atom("O", (1.2, 0.0, 0.0)),
                Atom("H", (-0.5, 0.9, 0.0)),
                Atom("H", (-0.5, -0.9, 0.0)),
            ),
            (Bond(0, 1, 2), Bond(0, 2, 1), Bond(0, 3, 1)),
        )
        self.assertEqual(type_molecule(formaldehyde).types[0], "C_R")

    def test_linear_sp_atom_with_further_substituents_minimizes(self):
        # Any linear (sp, coordination "1") center with a substituent beyond
        # its own two bonded atoms exposes a removable 1/sin(theta)
        # singularity in the analytic angle gradient at exactly its own 180
        # degree equilibrium, and a mathematically undefined torsion/dihedral
        # through its own axis. Before this fix both aborted minimization
        # with a "collinear" error for any such molecule, including a plain
        # nitrile like acetonitrile, not just the cumulated-bond systems
        # above (acetonitrile's sp carbon comes from an ordinary triple
        # bond, already part of the supported organic subset).
        acetonitrile = Molecule(
            "acetonitrile",
            (
                Atom("C", (0.0, 0.0, 0.0)),
                Atom("N", (1.16, 0.0, 0.0)),
                Atom("C", (-1.47, 0.0, 0.0)),
                Atom("H", (-1.84, 1.02, 0.0)),
                Atom("H", (-1.84, -0.51, 0.88)),
                Atom("H", (-1.84, -0.51, -0.88)),
            ),
            (Bond(0, 1, 3), Bond(0, 2, 1), Bond(2, 3, 1), Bond(2, 4, 1), Bond(2, 5, 1)),
        )
        self.assertEqual(type_molecule(acetonitrile).types[0], "C_1")
        result = minimize_partial_energy(acetonitrile, 80, 0.05, frozenset())
        self.assertTrue(result.ready, result.messages)

        # The gradient must agree with a finite-difference check even when
        # the starting geometry sits exactly at the singular 180 degree
        # point, not just somewhere nearby.
        analytic_energy, analytic_gradient, messages = _angle_energy_gradient(acetonitrile)
        self.assertEqual(messages, ())
        epsilon = 1e-5
        for atom_index in range(len(acetonitrile.atoms)):
            for axis in range(3):
                atoms = list(acetonitrile.atoms)
                position = list(atoms[atom_index].position)
                position[axis] += epsilon
                atoms[atom_index] = Atom(
                    atoms[atom_index].element, tuple(position), atoms[atom_index].formal_charge,
                    atoms[atom_index].chirality,
                )
                perturbed = Molecule(acetonitrile.name, tuple(atoms), acetonitrile.bonds)
                plus_energy = angle_bend_energy(perturbed).energy_kcal_mol
                position[axis] -= 2 * epsilon
                atoms[atom_index] = Atom(
                    atoms[atom_index].element, tuple(position), atoms[atom_index].formal_charge,
                    atoms[atom_index].chirality,
                )
                perturbed = Molecule(acetonitrile.name, tuple(atoms), acetonitrile.bonds)
                minus_energy = angle_bend_energy(perturbed).energy_kcal_mol
                finite_difference = (plus_energy - minus_energy) / (2 * epsilon)
                self.assertAlmostEqual(
                    analytic_gradient[atom_index][axis], finite_difference, places=2,
                )

    def test_types_triphenylene_without_external_dependencies(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")

        report = type_molecule(molecule)

        self.assertTrue(report.ready)
        self.assertEqual(len(report.types), 30)
        self.assertEqual(report.types.count("H_"), 12)
        self.assertTrue(all(atom_type in {"H_", "C_R"} for atom_type in report.types))
        energy = bond_stretch_energy(molecule)
        self.assertTrue(energy.ready)
        self.assertEqual(energy.terms, 33)
        self.assertGreaterEqual(energy.energy_kcal_mol, 0.0)
        angles = angle_bend_energy(molecule)
        self.assertTrue(angles.ready)
        self.assertEqual(angles.terms, 54)
        self.assertGreaterEqual(angles.energy_kcal_mol, 0.0)
        torsions = torsion_energy(molecule)
        self.assertTrue(torsions.ready)
        self.assertGreater(torsions.terms, 0)
        self.assertGreaterEqual(torsions.energy_kcal_mol, 0.0)
        vdw = van_der_waals_energy(molecule)
        self.assertTrue(vdw.ready)
        self.assertGreater(vdw.terms, 0)
        self.assertTrue(math.isfinite(vdw.energy_kcal_mol))
        inversion = out_of_plane_energy(molecule)
        self.assertTrue(inversion.ready)
        self.assertEqual(inversion.terms, 54)
        self.assertGreaterEqual(inversion.energy_kcal_mol, 0.0)
        total = partial_energy(molecule)
        self.assertTrue(total.ready)
        self.assertTrue(math.isfinite(total.energy_kcal_mol))
        minimized = minimize_partial_energy(molecule, 2, 0.01, {0})
        self.assertTrue(minimized.ready)
        self.assertLessEqual(minimized.final_energy_kcal_mol, minimized.initial_energy_kcal_mol)
        self.assertEqual(minimized.molecule.atoms[0].position, molecule.atoms[0].position)
        self.assertIn(minimized.termination, {"gradient tolerance", "line search stalled", "iteration limit"})

    def test_requires_explicit_hydrogens_for_bond_stretch_energy(self):
        molecule = read_v2000(ROOT / "217-59-4(トリフェニレン).mol")

        energy = bond_stretch_energy(molecule)

        self.assertFalse(energy.ready)
        self.assertIsNone(energy.energy_kcal_mol)
        self.assertIn("implicit hydrogen", " ".join(energy.messages))
        angles = angle_bend_energy(molecule)
        self.assertFalse(angles.ready)
        self.assertIsNone(angles.energy_kcal_mol)
        torsions = torsion_energy(molecule)
        self.assertFalse(torsions.ready)
        self.assertIsNone(torsions.energy_kcal_mol)
        vdw = van_der_waals_energy(molecule)
        self.assertFalse(vdw.ready)
        self.assertIsNone(vdw.energy_kcal_mol)
        inversion = out_of_plane_energy(molecule)
        self.assertFalse(inversion.ready)
        self.assertIsNone(inversion.energy_kcal_mol)
        total = partial_energy(molecule)
        self.assertFalse(total.ready)
        self.assertIsNone(total.energy_kcal_mol)

    def test_reports_an_element_outside_the_first_supported_subset(self):
        molecule = read_v2000(ROOT / "131159-39-2(フラーレン).mol")

        report = type_molecule(molecule)

        self.assertTrue(report.ready)


if __name__ == "__main__":
    unittest.main()
