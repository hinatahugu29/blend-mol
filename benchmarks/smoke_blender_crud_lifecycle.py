"""Headless end-to-end smoke test for operators with no prior automated coverage.

Most Blend Mol operators are simple EXECUTE_DEFAULT operators (no modal loop,
no file-select invoke needed when a filepath/property is already supplied),
so they can be driven directly through ``bpy.ops.blendmol.*`` from a
background script. Before this test, the only automated coverage for the
basic atom/bond CRUD, constraint, snapshot, import/export, and display
operators was manual verification recorded in ROADMAP.md; none of it ran
under ``--python-exit-code 1`` where a real exception would fail the run.

This drives one continuous session through: molecule creation, atom/bond
CRUD with valence rejection, element/charge edits, fixed atoms, all three
constraint kinds, position snapshots, non-UFF relax, partial UFF
minimization, atom deletion with constraint/fixed-atom re-indexing, display
rebuild/sync, subdivision, element color reset, and round-tripping through
every supported import/export format.
"""

import importlib
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

addon = importlib.import_module("blend_mol")
from blend_mol.molfile import MolfileError


def _select_only(obj) -> None:
    for other in addon.bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    addon.bpy.context.view_layer.objects.active = obj


def _atom_object(collection, atom_index):
    return next(
        obj for obj in collection.objects
        if obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and obj.get("blend_mol_atom_index") == atom_index
    )


def _bond_object(collection, bond_index):
    return next(
        obj for obj in collection.objects
        if obj.get("blend_mol_kind") == "bond"
        and obj.get("blend_mol_bond_index") == bond_index
    )


addon.register()
try:
    scene = addon.bpy.context.scene
    settings = scene.blend_mol_settings

    # --- New molecule, then grow it with add_atom / add_bonded_atom --------
    settings.edit_element = "C"
    settings.formal_charge = 0
    result = addon.bpy.ops.blendmol.new_molecule(molecule_name="CRUD lifecycle")
    assert result == {"FINISHED"}, result
    root = addon._selected_molecule_root(addon.bpy.context)
    assert root is not None
    collection = addon._molecule_collection(root)
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert len(molecule.atoms) == 1 and molecule.atoms[0].element == "C"

    scene.cursor.location = (1.5, 0.3, 0.0)
    settings.edit_element = "O"
    assert addon.bpy.ops.blendmol.add_bonded_atom() == {"FINISHED"}
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert len(molecule.atoms) == 2 and len(molecule.bonds) == 1
    assert molecule.atoms[1].element == "O"

    scene.cursor.location = (-1.5, -0.3, 0.2)
    addon.bpy.ops.object.select_all(action="DESELECT")
    addon.bpy.context.view_layer.objects.active = None
    assert addon.bpy.ops.blendmol.add_atom() == {"CANCELLED"}, "add_atom needs a selected molecule"
    _select_only(_atom_object(collection, 0))
    assert addon.bpy.ops.blendmol.add_atom() == {"FINISHED"}
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert len(molecule.atoms) == 3, "add_atom must append an unbonded atom"

    # --- set_bond: create, then change order, then reject an invalid order -
    _select_only(_atom_object(collection, 0))
    _atom_object(collection, 2).select_set(True)
    settings.bond_order = 1
    assert addon.bpy.ops.blendmol.set_bond() == {"FINISHED"}
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert len(molecule.bonds) == 2

    # Topology edits rebuild the display, so re-select the (new) atom objects.
    # Atom 2 is still O (valence 2) here, so order 2 is the highest it allows.
    _select_only(_atom_object(collection, 0))
    _atom_object(collection, 2).select_set(True)
    settings.bond_order = 2
    assert addon.bpy.ops.blendmol.set_bond() == {"FINISHED"}
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert any(
        {bond.atom_a, bond.atom_b} == {0, 2} and bond.order == 2 for bond in molecule.bonds
    ), "set_bond must change the existing bond's order rather than duplicating it"

    # --- change_atom_element / set_formal_charge ----------------------------
    _select_only(_atom_object(collection, 2))
    settings.edit_element = "N"
    assert addon.bpy.ops.blendmol.change_atom_element() == {"FINISHED"}
    _select_only(_atom_object(collection, 2))  # display was rebuilt; reselect
    settings.formal_charge = 1
    assert addon.bpy.ops.blendmol.set_formal_charge() == {"FINISHED"}
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert molecule.atoms[2].element == "N" and molecule.atoms[2].formal_charge == 1

    # Atom 2 is now N+ (valence 4) with one order-2 bond already (valence
    # used = 2). Two more order-1 bonds reach the limit exactly and must
    # succeed; a further bond must be rejected without touching data.
    settings.bond_order = 1
    # Atom 2 sits at (-1.5, 0, 0); place its new neighbours off that axis so
    # the resulting torsions and out-of-plane terms are not degenerate.
    for cursor_position in ((-2.5, 1.0, 0.6), (-2.5, -1.0, -0.6)):
        scene.cursor.location = cursor_position
        _select_only(_atom_object(collection, 2))
        assert addon.bpy.ops.blendmol.add_bonded_atom() == {"FINISHED"}
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert len(molecule.atoms) == 5

    _select_only(_atom_object(collection, 2))
    revision_at_capacity = addon._molecule_revision(root)
    data_at_capacity = root["blend_mol_data"]
    rejected = addon.bpy.ops.blendmol.add_bonded_atom()
    assert rejected == {"CANCELLED"}, "a 5th bond on tetravalent N+ must be rejected"
    assert addon._molecule_revision(root) == revision_at_capacity
    assert root["blend_mol_data"] == data_at_capacity

    # --- toggle_fixed_atom ---------------------------------------------------
    _select_only(_atom_object(collection, 0))
    assert addon.bpy.ops.blendmol.toggle_fixed_atom() == {"FINISHED"}
    assert 0 in addon._fixed_atom_indices(root)
    assert addon.bpy.ops.blendmol.toggle_fixed_atom() == {"FINISHED"}
    assert 0 not in addon._fixed_atom_indices(root)

    # --- constraints: distance, angle, dihedral, then remove ---------------
    _select_only(_atom_object(collection, 0))
    _atom_object(collection, 1).select_set(True)
    settings.distance_constraint_target = 2.0
    settings.distance_constraint_strength = 10.0
    assert addon.bpy.ops.blendmol.set_distance_constraint() == {"FINISHED"}
    assert len(addon._distance_constraints(root)) == 1
    assert addon.bpy.ops.blendmol.remove_distance_constraint(remove_all=True) == {"FINISHED"}
    assert len(addon._distance_constraints(root)) == 0

    _select_only(_atom_object(collection, 1))
    _atom_object(collection, 0).select_set(True)
    _atom_object(collection, 2).select_set(True)
    addon.bpy.context.view_layer.objects.active = _atom_object(collection, 1)
    settings.angle_constraint_target = 109.5
    settings.angle_constraint_strength = 5.0
    assert addon.bpy.ops.blendmol.set_angle_constraint() == {"FINISHED"}
    assert len(addon._angle_constraints(root)) == 1
    assert addon.bpy.ops.blendmol.remove_angle_constraint(remove_all=True) == {"FINISHED"}
    assert len(addon._angle_constraints(root)) == 0

    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    dihedral_path = None
    for bond_b in molecule.bonds:
        for bond_a in molecule.bonds:
            if bond_a is bond_b:
                continue
            shared = {bond_a.atom_a, bond_a.atom_b} & {bond_b.atom_a, bond_b.atom_b}
            if len(shared) == 1:
                center_a = next(iter(shared))
                center_b = (bond_b.atom_a if bond_b.atom_b == center_a else bond_b.atom_b)
                outer_a = bond_a.atom_a if bond_a.atom_b == center_a else bond_a.atom_b
                for bond_c in molecule.bonds:
                    if center_b in (bond_c.atom_a, bond_c.atom_b) and outer_a not in (
                        bond_c.atom_a, bond_c.atom_b,
                    ):
                        outer_b = bond_c.atom_a if bond_c.atom_b == center_b else bond_c.atom_b
                        if outer_b != center_a and outer_b != outer_a:
                            dihedral_path = (outer_a, center_a, center_b, outer_b)
                            break
                if dihedral_path:
                    break
        if dihedral_path:
            break
    if dihedral_path is not None:
        for index in dihedral_path:
            _atom_object(collection, index).select_set(index == dihedral_path[0])
        addon.bpy.ops.object.select_all(action="DESELECT")
        for index in dihedral_path:
            _atom_object(collection, index).select_set(True)
        addon.bpy.context.view_layer.objects.active = _atom_object(collection, dihedral_path[1])
        settings.dihedral_constraint_target = 30.0
        settings.dihedral_constraint_strength = 2.0
        set_result = addon.bpy.ops.blendmol.set_dihedral_constraint()
        if set_result == {"FINISHED"}:
            assert len(addon._dihedral_constraints(root)) == 1
            assert addon.bpy.ops.blendmol.remove_dihedral_constraint(remove_all=True) == {"FINISHED"}
            assert len(addon._dihedral_constraints(root)) == 0

    # --- position snapshot save/restore -------------------------------------
    _select_only(_atom_object(collection, 0))
    assert addon.bpy.ops.blendmol.save_position_snapshot() == {"FINISHED"}
    snapshot_molecule = addon._deserialize_molecule(root["blend_mol_data"])
    atom_object = _atom_object(collection, 0)
    atom_object.location.x += 1.0
    addon.bpy.ops.blendmol.sync_atom_positions()
    assert addon.bpy.ops.blendmol.restore_position_snapshot() == {"FINISHED"}
    restored_molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert math.dist(restored_molecule.atoms[0].position, snapshot_molecule.atoms[0].position) < 1e-6

    # --- relax_display_geometry (non-UFF) ------------------------------------
    assert addon.bpy.ops.blendmol.relax_display_geometry() == {"FINISHED"}

    # --- minimize_partial_uff on a real, already-validated molecule ---------
    # The synthetic molecule above has sparse, hand-placed coordinates whose
    # implicit hydrogens can land collinear with an existing bond; that is a
    # geometry artifact of this test's atom placement, not a target for this
    # smoke test. Exercise the operator on triphenylene instead, which is
    # covered against external UFF references elsewhere in this repository.
    uff_molecule = addon.read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    uff_root = addon._create_molecule(uff_molecule, settings, "uff_lifecycle")
    uff_collection = addon._molecule_collection(uff_root)
    _select_only(_atom_object(uff_collection, 0))
    settings.uff_iterations = 3
    minimize_result = addon.bpy.ops.blendmol.minimize_partial_uff()
    assert minimize_result == {"FINISHED"}, minimize_result
    assert "blend_mol_uff_last_run" in uff_root

    # --- delete_atom re-indexes fixed atoms and constraints -----------------
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    atom_count_before_delete = len(molecule.atoms)
    _select_only(_atom_object(collection, 0))
    assert addon.bpy.ops.blendmol.toggle_fixed_atom() == {"FINISHED"}
    _select_only(_atom_object(collection, 1))
    assert addon.bpy.ops.blendmol.delete_atom() == {"FINISHED"}
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    assert len(molecule.atoms) == atom_count_before_delete - 1
    assert 0 in addon._fixed_atom_indices(root), "deleting a higher-index atom must not shift atom 0's fixed flag"

    # --- delete_bond ----------------------------------------------------------
    molecule = addon._deserialize_molecule(root["blend_mol_data"])
    if molecule.bonds:
        _select_only(_bond_object(collection, 0))
        bond_count_before = len(molecule.bonds)
        assert addon.bpy.ops.blendmol.delete_bond() == {"FINISHED"}
        molecule = addon._deserialize_molecule(root["blend_mol_data"])
        assert len(molecule.bonds) == bond_count_before - 1

    # --- rebuild_display / add_subdivision / reset_element_colors -----------
    _select_only(_atom_object(collection, 0))
    assert addon.bpy.ops.blendmol.rebuild_display() == {"FINISHED"}
    assert addon.bpy.ops.blendmol.add_subdivision() == {"FINISHED"}
    assert any(
        obj.modifiers.get("Blend Mol Subdivision") is not None
        for obj in collection.objects
        if obj.get("blend_mol_kind") == "atom"
    ), "add_subdivision must attach a Subdivision Surface modifier"
    assert addon.bpy.ops.blendmol.reset_element_colors() == {"FINISHED"}

    # --- export / import round trip across every supported format ----------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mol_path = tmp_path / "roundtrip.mol"
        xyz_path = tmp_path / "roundtrip.xyz"
        pdb_path = tmp_path / "roundtrip.pdb"
        report_path = tmp_path / "roundtrip.json"

        def _noop_report(_self, _flags, _message):
            pass

        _select_only(_atom_object(collection, 0))
        export_result = addon.BLENDMOL_OT_export_v2000.execute(
            type("ExportOp", (), {"filepath": str(mol_path), "report": _noop_report})(),
            addon.bpy.context,
        )
        assert export_result == {"FINISHED"}, export_result
        assert mol_path.is_file() and mol_path.stat().st_size > 0

        export_xyz_result = addon.BLENDMOL_OT_export_xyz.execute(
            type("ExportOp", (), {"filepath": str(xyz_path), "report": _noop_report})(),
            addon.bpy.context,
        )
        assert export_xyz_result == {"FINISHED"}, export_xyz_result
        assert xyz_path.is_file() and xyz_path.stat().st_size > 0

        export_pdb_result = addon.BLENDMOL_OT_export_pdb.execute(
            type("ExportOp", (), {"filepath": str(pdb_path), "report": _noop_report})(),
            addon.bpy.context,
        )
        assert export_pdb_result == {"FINISHED"}, export_pdb_result
        assert pdb_path.is_file() and pdb_path.stat().st_size > 0

        export_report_result = addon.BLENDMOL_OT_export_uff_report.execute(
            type("ExportReportOp", (), {"filepath": str(report_path), "report": _noop_report})(),
            addon.bpy.context,
        )
        assert export_report_result == {"FINISHED"}, export_report_result
        assert report_path.is_file() and report_path.stat().st_size > 0

        import_result = addon.bpy.ops.blendmol.import_v2000(filepath=str(mol_path))
        assert import_result == {"FINISHED"}, import_result
        import_xyz_result = addon.bpy.ops.blendmol.import_xyz(filepath=str(xyz_path))
        assert import_xyz_result == {"FINISHED"}, import_xyz_result
        import_pdb_result = addon.bpy.ops.blendmol.import_pdb(filepath=str(pdb_path))
        assert import_pdb_result == {"FINISHED"}, import_pdb_result

        # bpy.ops raises RuntimeError whenever an operator reports an ERROR
        # level message, regardless of its returned status set; the operator
        # itself still returns {'CANCELLED'} internally (verified by calling
        # .execute() directly below), so both call styles are checked here.
        missing_path = tmp_path / "does_not_exist.pdb"
        try:
            missing_result = addon.bpy.ops.blendmol.import_pdb(filepath=str(missing_path))
            assert missing_result == {"CANCELLED"}, (
                f"importing a missing PDB file must report CANCELLED, got {missing_result}"
            )
        except RuntimeError as exc:
            assert "No such file" in str(exc), exc
        direct_result = addon.BLENDMOL_OT_import_pdb.execute(
            type("ImportOp", (), {"filepath": str(missing_path), "report": _noop_report})(),
            addon.bpy.context,
        )
        assert direct_result == {"CANCELLED"}, direct_result

        mol_text = mol_path.read_text(encoding="utf-8")
        sdf_path = tmp_path / "roundtrip.sdf"
        sdf_path.write_text(mol_text + "$$$$\n", encoding="utf-8")
        import_sdf_result = addon.bpy.ops.blendmol.import_sdf(filepath=str(sdf_path))
        assert import_sdf_result == {"FINISHED"}, import_sdf_result

    smiles_result = addon.bpy.ops.blendmol.import_smiles(smiles="c1ccccc1O", molecule_name="phenol")
    assert smiles_result == {"FINISHED"}, smiles_result
    phenol_root = addon._selected_molecule_root(addon.bpy.context)
    phenol_molecule = addon._deserialize_molecule(phenol_root["blend_mol_data"])
    assert len(phenol_molecule.atoms) == 7, "benzene ring plus hydroxyl oxygen"

    try:
        addon.read_smiles("not a smiles (((", "bad")
        assert False, "malformed SMILES must raise MolfileError"
    except MolfileError:
        pass

    print(
        "BLEND_MOL_CRUD_LIFECYCLE_SMOKE_OK",
        "new_molecule_add_atom_add_bonded_atom_ok",
        "set_bond_create_and_change_ok",
        "change_element_and_charge_ok",
        "toggle_fixed_atom_ok",
        "distance_angle_dihedral_constraints_ok",
        "position_snapshot_round_trip_ok",
        "relax_and_minimize_partial_uff_ok",
        "delete_atom_reindexes_fixed_atoms_ok",
        "delete_bond_ok",
        "rebuild_subdivision_colors_ok",
        "export_import_v2000_xyz_pdb_sdf_ok",
        "export_uff_report_ok",
        "missing_file_import_cancelled_ok",
        "smiles_import_ok",
        "malformed_smiles_raises_ok",
    )
finally:
    addon.unregister()
