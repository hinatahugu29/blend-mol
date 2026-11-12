"""Atom and bond editing operators."""

from __future__ import annotations

import json
from dataclasses import replace

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from .display import (
    _clear_molecule_display, _create_molecule, _implicit_hydrogens, _render_molecule,
)
from .molfile import Atom, Bond, Molecule
from .scene_data import (
    _activate_explicit_atom, _angle_constraints, _bond_edit_valence_error,
    _deserialize_molecule, _dihedral_constraints, _distance_constraints, _fixed_atom_indices,
    _molecule_collection, _selected_explicit_atom, _selected_explicit_atoms,
    _selected_explicit_atoms_any, _selected_explicit_bond, _selected_molecule_root,
    _store_angle_constraints, _store_dihedral_constraints, _store_distance_constraints,
    _store_fixed_atom_indices, _store_topology_change, _sync_atom_positions,
)


class BLENDMOL_OT_new_molecule(Operator):
    bl_idname = "blendmol.new_molecule"
    bl_label = "New Blend Mol Molecule"
    bl_description = "Create a molecule with its first atom at the 3D cursor"
    bl_options = {"REGISTER", "UNDO"}

    molecule_name: StringProperty(name="Molecule Name", default="Untitled Molecule", maxlen=128)

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        layout.prop(self, "molecule_name")
        layout.prop(settings, "edit_element", text="First Element")
        layout.prop(settings, "formal_charge", text="Formal Charge")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        name = self.molecule_name.strip()
        if not name:
            self.report({"WARNING"}, "Enter a molecule name")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        molecule = Molecule(
            name,
            (Atom(settings.edit_element, tuple(context.scene.cursor.location), settings.formal_charge),),
            (),
        )
        root = _create_molecule(molecule, settings, "Created in Blend Mol")
        _activate_explicit_atom(context, root, 0)
        self.report({"INFO"}, f"Created {name} with one {settings.edit_element} atom")
        return {"FINISHED"}


class BLENDMOL_OT_sync_atom_positions(Operator):
    bl_idname = "blendmol.sync_atom_positions"
    bl_label = "Sync Edited Atom Positions"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        collection = _molecule_collection(root)
        if collection is None or "blend_mol_data" not in root:
            self.report({"ERROR"}, "This molecule does not contain editable Blend Mol data")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot synchronize atom positions: {exc}")
            return {"CANCELLED"}
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, molecule, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Synchronized {len(molecule.atoms)} atom positions")
        return {"FINISHED"}


class BLENDMOL_OT_add_atom(Operator):
    bl_idname = "blendmol.add_atom"
    bl_label = "Add Atom at 3D Cursor"
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
        local_position = root.matrix_world.inverted() @ context.scene.cursor.location
        element = context.scene.blend_mol_settings.edit_element
        updated = Molecule(
            molecule.name,
            molecule.atoms + (Atom(element, tuple(local_position), context.scene.blend_mol_settings.formal_charge),),
            molecule.bonds,
        )
        _store_topology_change(root, updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        _activate_explicit_atom(context, root, len(updated.atoms) - 1)
        self.report({"INFO"}, f"Added {element} at the 3D cursor")
        return {"FINISHED"}


class BLENDMOL_OT_add_bonded_atom(Operator):
    bl_idname = "blendmol.add_bonded_atom"
    bl_label = "Add Bonded Atom at 3D Cursor"
    bl_description = (
        "Add an atom at the 3D cursor and bond it to the selected atom; "
        "if that atom is already saturated, replaces as many of its "
        "explicit hydrogens as the bond order needs instead of failing"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select one non-implicit Blend Mol atom first")
            return {"CANCELLED"}
        root, collection, atom_index = selected
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        local_position = root.matrix_world.inverted() @ context.scene.cursor.location

        def _try_add(base_molecule: Molecule, center_index: int):
            new_index = len(base_molecule.atoms)
            candidate = Molecule(
                base_molecule.name,
                base_molecule.atoms
                + (Atom(settings.edit_element, tuple(local_position), settings.formal_charge),),
                base_molecule.bonds + (Bond(center_index, new_index, settings.bond_order),),
            )
            return candidate, _bond_edit_valence_error(base_molecule, candidate, (center_index, new_index))

        updated, error = _try_add(molecule, atom_index)
        pending_constraints = None
        replaced_hydrogens = 0
        if error:
            # No room at the selected atom's current valence. If it carries
            # explicit hydrogens, replace as many of them as the requested
            # bond order needs instead of making the user delete them by
            # hand first -- the same outcome as selecting each hydrogen and
            # using "Change Selected Atom Element", just reachable directly
            # from the saturated heavy atom without hunting for which
            # neighbours are the spare hydrogens. A double/triple bond
            # addition may need to free up more than one hydrogen's worth
            # of valence.
            working_molecule, working_index = molecule, atom_index
            fixed_atoms = _fixed_atom_indices(root)
            distances, angles, dihedrals = (
                _distance_constraints(root), _angle_constraints(root), _dihedral_constraints(root),
            )
            while error:
                hydrogen_index = next(
                    (
                        other for bond in working_molecule.bonds
                        for other in (
                            (bond.atom_b,) if bond.atom_a == working_index
                            else (bond.atom_a,) if bond.atom_b == working_index else ()
                        )
                        if working_molecule.atoms[other].element == "H"
                    ),
                    None,
                )
                if hydrogen_index is None:
                    break
                working_molecule, fixed_atoms, distances, angles, dihedrals = _remove_atom_reindexed_from(
                    working_molecule, hydrogen_index, fixed_atoms, distances, angles, dihedrals,
                )
                working_index -= working_index > hydrogen_index
                replaced_hydrogens += 1
                updated, error = _try_add(working_molecule, working_index)
            if not error:
                atom_index = working_index
                pending_constraints = (fixed_atoms, distances, angles, dihedrals)
            else:
                replaced_hydrogens = 0
        if error:
            self.report({"WARNING"}, error)
            return {"CANCELLED"}
        new_atom_index = len(updated.atoms) - 1
        _store_topology_change(root, updated)
        if pending_constraints is not None:
            fixed_atoms, distances, angles, dihedrals = pending_constraints
            _store_fixed_atom_indices(root, fixed_atoms)
            _store_distance_constraints(root, distances)
            _store_angle_constraints(root, angles)
            _store_dihedral_constraints(root, dihedrals)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, settings)
        _activate_explicit_atom(context, root, new_atom_index)
        if replaced_hydrogens:
            noun = "hydrogen" if replaced_hydrogens == 1 else "hydrogens"
            self.report(
                {"INFO"},
                f"Replaced {replaced_hydrogens} explicit {noun} on atom {atom_index + 1} with "
                f"{settings.edit_element} and bond {atom_index + 1}-{new_atom_index + 1}",
            )
        else:
            self.report(
                {"INFO"},
                f"Added {settings.edit_element} and bond {atom_index + 1}-{new_atom_index + 1}",
            )
        return {"FINISHED"}


def _with_hydrogens_explicit(molecule: Molecule):
    """Return (updated_molecule, added_count); pure, writes nothing to root."""
    hydrogens = _implicit_hydrogens(molecule)
    if not hydrogens:
        return molecule, 0
    atoms = list(molecule.atoms)
    bonds = list(molecule.bonds)
    for parent_index, position in hydrogens:
        hydrogen_index = len(atoms)
        atoms.append(Atom("H", tuple(position)))
        bonds.append(Bond(parent_index, hydrogen_index, 1))
    return Molecule(molecule.name, tuple(atoms), tuple(bonds)), len(hydrogens)


def _ensure_hydrogens_explicit_for_uff(root: bpy.types.Object, collection: bpy.types.Collection,
                                        molecule: Molecule, settings) -> Molecule:
    """If molecule has implicit hydrogens, make them explicit and commit that
    change to root before a UFF run, since UFF only ever sees explicit atoms.
    Returns molecule unchanged (and touches nothing) when there is nothing to do.
    """
    updated, count = _with_hydrogens_explicit(molecule)
    if count == 0:
        return molecule
    _store_topology_change(root, updated)
    _clear_molecule_display(root, collection)
    _render_molecule(root, collection, updated, settings)
    return updated


class BLENDMOL_OT_make_hydrogens_explicit(Operator):
    bl_idname = "blendmol.make_hydrogens_explicit"
    bl_label = "Make Hydrogens Explicit"
    bl_description = "Prepare automatic Live UFF by converting implicit hydrogens into editable atoms and bonds"
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
        updated, count = _with_hydrogens_explicit(molecule)
        if count == 0:
            self.report({"INFO"}, "This molecule has no implicit hydrogens")
            return {"CANCELLED"}
        selected = _selected_explicit_atom(context)
        selected_index = selected[2] if selected and selected[0] == root else None
        _store_topology_change(root, updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        if selected_index is not None:
            _activate_explicit_atom(context, root, selected_index)
        else:
            bpy.ops.object.select_all(action="DESELECT")
            root.select_set(True)
            context.view_layer.objects.active = root
        self.report({"INFO"}, f"Made {count} hydrogen(s) explicit")
        return {"FINISHED"}


class BLENDMOL_OT_change_atom_element(Operator):
    bl_idname = "blendmol.change_atom_element"
    bl_label = "Change Selected Atom Element"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select a non-implicit Blend Mol atom first")
            return {"CANCELLED"}
        root, collection, atom_index = selected
        molecule = _deserialize_molecule(root["blend_mol_data"])
        element = context.scene.blend_mol_settings.edit_element
        atoms = list(molecule.atoms)
        # A mass number belongs to the old element, so the isotope label is dropped.
        atoms[atom_index] = Atom(
            element, atoms[atom_index].position, atoms[atom_index].formal_charge, atoms[atom_index].chirality
        )
        updated = Molecule(molecule.name, tuple(atoms), molecule.bonds)
        _store_topology_change(root, updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Changed atom {atom_index + 1} to {element}")
        return {"FINISHED"}


class BLENDMOL_OT_set_formal_charge(Operator):
    bl_idname = "blendmol.set_formal_charge"
    bl_label = "Set Selected Atom Charge"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select a non-implicit Blend Mol atom first")
            return {"CANCELLED"}
        root, collection, atom_index = selected
        molecule = _deserialize_molecule(root["blend_mol_data"])
        charge = context.scene.blend_mol_settings.formal_charge
        atoms = list(molecule.atoms)
        atom = atoms[atom_index]
        atoms[atom_index] = replace(atom, formal_charge=charge)
        updated = Molecule(molecule.name, tuple(atoms), molecule.bonds)
        _store_topology_change(root, updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Set atom {atom_index + 1} charge to {charge:+d}")
        return {"FINISHED"}


def _remove_atom_reindexed_from(molecule: Molecule, atom_index: int, fixed_atoms, distances, angles, dihedrals):
    """Compute the molecule and fixed-atom/constraint data after deleting one atom.

    Pure: takes the current fixed-atom/constraint data explicitly (rather than
    reading root) so a caller can chain several removals, feeding each result
    into the next call, without touching root until it decides to commit.
    """
    atoms = tuple(atom for index, atom in enumerate(molecule.atoms) if index != atom_index)
    bonds = []
    for bond in molecule.bonds:
        if atom_index in {bond.atom_a, bond.atom_b}:
            continue
        bonds.append(Bond(
            bond.atom_a - (bond.atom_a > atom_index),
            bond.atom_b - (bond.atom_b > atom_index),
            bond.order,
            bond.stereo,
        ))
    updated = Molecule(molecule.name, atoms, tuple(bonds))
    remapped_fixed_atoms = {
        index - (index > atom_index)
        for index in fixed_atoms
        if index != atom_index
    }
    remapped_distances = tuple(
        (first - (first > atom_index), second - (second > atom_index), target, strength)
        for first, second, target, strength in distances
        if atom_index not in {first, second}
    )
    remapped_angles = tuple(
        (
            first - (first > atom_index), center - (center > atom_index),
            second - (second > atom_index), target, strength,
        )
        for first, center, second, target, strength in angles
        if atom_index not in {first, center, second}
    )
    remapped_dihedrals = tuple(
        tuple(index - (index > atom_index) for index in (first, center_a, center_b, fourth))
        + (target, strength)
        for first, center_a, center_b, fourth, target, strength in dihedrals
        if atom_index not in (first, center_a, center_b, fourth)
    )
    return updated, remapped_fixed_atoms, remapped_distances, remapped_angles, remapped_dihedrals


def _remove_atom_reindexed(root: bpy.types.Object, molecule: Molecule, atom_index: int):
    """_remove_atom_reindexed_from, reading the current constraint data from root."""
    return _remove_atom_reindexed_from(
        molecule, atom_index,
        _fixed_atom_indices(root), _distance_constraints(root),
        _angle_constraints(root), _dihedral_constraints(root),
    )


class BLENDMOL_OT_delete_atom(Operator):
    bl_idname = "blendmol.delete_atom"
    bl_label = "Delete Selected Atom"
    bl_description = "Delete the selected non-implicit atom(s) and their connected bonds"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atoms_any(context)
        if selected is None:
            self.report({"WARNING"}, "Select one or more non-implicit Blend Mol atoms first")
            return {"CANCELLED"}
        root, collection, atom_indices = selected
        molecule = _deserialize_molecule(root["blend_mol_data"])
        fixed_atoms = _fixed_atom_indices(root)
        distances, angles, dihedrals = (
            _distance_constraints(root), _angle_constraints(root), _dihedral_constraints(root),
        )
        # Remove highest index first: that never shifts the still-pending
        # lower indices, so no index bookkeeping is needed between removals.
        for atom_index in reversed(atom_indices):
            molecule, fixed_atoms, distances, angles, dihedrals = _remove_atom_reindexed_from(
                molecule, atom_index, fixed_atoms, distances, angles, dihedrals,
            )
        _store_topology_change(root, molecule)
        _store_fixed_atom_indices(root, fixed_atoms)
        _store_distance_constraints(root, distances)
        _store_angle_constraints(root, angles)
        _store_dihedral_constraints(root, dihedrals)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, molecule, context.scene.blend_mol_settings)
        if len(atom_indices) == 1:
            self.report({"INFO"}, f"Deleted atom {atom_indices[0] + 1} and its connected bonds")
        else:
            self.report({"INFO"}, f"Deleted {len(atom_indices)} atoms and their connected bonds")
        return {"FINISHED"}


class BLENDMOL_OT_set_bond(Operator):
    bl_idname = "blendmol.set_bond"
    bl_label = "Create or Change Selected-Atom Bond"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atoms(context)
        if selected is None:
            self.report({"WARNING"}, "Select exactly two non-implicit atoms in one molecule")
            return {"CANCELLED"}
        root, collection, (atom_a, atom_b) = selected
        molecule = _deserialize_molecule(root["blend_mol_data"])
        order = context.scene.blend_mol_settings.bond_order
        bonds = list(molecule.bonds)
        for index, bond in enumerate(bonds):
            if {bond.atom_a, bond.atom_b} == {atom_a, atom_b}:
                bonds[index] = Bond(
                    bond.atom_a, bond.atom_b, order, bond.stereo if order == 1 else None
                )
                action = "Changed"
                break
        else:
            bonds.append(Bond(atom_a, atom_b, order))
            action = "Created"
        updated = Molecule(molecule.name, molecule.atoms, tuple(bonds))
        error = _bond_edit_valence_error(molecule, updated, (atom_a, atom_b))
        if error:
            self.report({"WARNING"}, error)
            return {"CANCELLED"}
        _store_topology_change(root, updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"{action} bond {atom_a + 1}-{atom_b + 1} with order {order}")
        return {"FINISHED"}


class BLENDMOL_OT_delete_bond(Operator):
    bl_idname = "blendmol.delete_bond"
    bl_label = "Delete Selected Bond"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_bond(context)
        if selected is None:
            self.report({"WARNING"}, "Select a non-implicit Blend Mol bond first")
            return {"CANCELLED"}
        root, collection, bond_index = selected
        molecule = _deserialize_molecule(root["blend_mol_data"])
        if not 0 <= bond_index < len(molecule.bonds):
            self.report({"ERROR"}, "Selected bond is not part of the stored molecule data")
            return {"CANCELLED"}
        bonds = tuple(bond for index, bond in enumerate(molecule.bonds) if index != bond_index)
        updated = Molecule(molecule.name, molecule.atoms, bonds)
        _store_topology_change(root, updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Deleted bond {bond_index + 1}")
        return {"FINISHED"}
