"""Blender regression: editing after UFF must not silently overbond atoms."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import blend_mol as addon
import bpy


def select_atoms(root, indices):
    bpy.ops.object.select_all(action="DESELECT")
    collection = addon._molecule_collection(root)
    objects = [obj for obj in collection.objects
               if obj.get("blend_mol_kind") == "atom"
               and not obj.get("blend_mol_implicit")
               and obj.get("blend_mol_atom_index") in indices]
    assert len(objects) == len(indices)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def snapshot(root):
    return (
        root["blend_mol_data"], addon._molecule_revision(root),
        {obj.name: obj.as_pointer() for obj in addon._molecule_collection(root).objects},
        root.get("blend_mol_uff_last_run"),
    )


addon.register()
try:
    settings = bpy.context.scene.blend_mol_settings
    water = addon.Molecule("Water regression", (addon.Atom("O", (0, 0, 0)),), ())
    root = addon._create_molecule(water, settings, "regression")
    select_atoms(root, {0})
    assert bpy.ops.blendmol.make_hydrogens_explicit() == {"FINISHED"}
    assert bpy.ops.blendmol.minimize_partial_uff() == {"FINISHED"}
    select_atoms(root, {0})
    settings.edit_element = "C"
    bpy.context.scene.cursor.location = (2, 0, 0)
    before = snapshot(root)
    for order in (1, 2, 3):
        settings.bond_order = order
        assert bpy.ops.blendmol.add_bonded_atom() == {"CANCELLED"}
        assert snapshot(root) == before, "Rejected edit mutated molecule or display"
    # Adding an unbonded atom is valid; connecting it to saturated O is not.
    assert bpy.ops.blendmol.add_atom() == {"FINISHED"}
    select_atoms(root, {0, 3})
    settings.bond_order = 1
    before = snapshot(root)
    assert bpy.ops.blendmol.set_bond() == {"CANCELLED"}
    assert snapshot(root) == before
    # Increasing an O-H bond order must also check the hydrogen endpoint.
    select_atoms(root, {0, 1})
    settings.bond_order = 2
    assert bpy.ops.blendmol.set_bond() == {"CANCELLED"}
    assert snapshot(root) == before
    # Valid double-bond additions are still supported.
    carbon = addon.Molecule("Carbon regression", (addon.Atom("C", (0, 0, 0)),), ())
    other = addon._create_molecule(carbon, settings, "regression")
    select_atoms(other, {0})
    settings.edit_element = "O"
    assert bpy.ops.blendmol.add_bonded_atom() == {"FINISHED"}
    current = addon._deserialize_molecule(other["blend_mol_data"])
    assert len(current.atoms) == 2 and current.bonds[0].order == 2
    # A new hydrogen cannot receive a double bond, even if the parent can.
    select_atoms(other, {0})
    settings.edit_element = "H"
    before = snapshot(other)
    assert bpy.ops.blendmol.add_bonded_atom() == {"CANCELLED"}
    assert snapshot(other) == before
    # Existing overbonded imports can be repaired incrementally.
    atoms = (addon.Atom("O", (0, 0, 0)), addon.Atom("C", (2, 0, 0)),
             addon.Atom("C", (-2, 0, 0)))
    invalid = addon.Molecule("Repair", atoms, (addon.Bond(0, 1, 2), addon.Bond(0, 2, 2)))
    repair = addon._create_molecule(invalid, settings, "regression")
    select_atoms(repair, {0, 1})
    settings.bond_order = 1
    assert bpy.ops.blendmol.set_bond() == {"FINISHED"}
    repaired = addon._deserialize_molecule(repair["blend_mol_data"])
    assert sum(b.order for b in repaired.bonds) == 3
    # Charged nitrogen keeps the existing four-bond valence allowance.
    ammonium = addon.Molecule("Ammonium", (addon.Atom("N", (0, 0, 0), 1),) + tuple(
        addon.Atom("H", position) for position in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ), tuple(addon.Bond(0, index, 1) for index in range(1, 4)))
    charged = addon._create_molecule(ammonium, settings, "regression")
    select_atoms(charged, {0})
    settings.edit_element = "H"
    settings.formal_charge = 0
    assert bpy.ops.blendmol.add_bonded_atom() == {"FINISHED"}
    assert not addon._molecule_validation_messages(addon._deserialize_molecule(charged["blend_mol_data"]))
    print("BLEND_MOL_BOND_VALENCE_OK post_uff atomic_rejection valid_addition repair charged_nitrogen")
finally:
    addon.unregister()
