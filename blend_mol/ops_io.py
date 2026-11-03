"""Import and export operators."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from .display import _create_molecule
from .molfile import (
    MolfileError, read_pdb, read_sdf, read_smiles, read_v2000, read_xyz, write_pdb, write_sdf,
    write_v2000, write_xyz,
)
from .scene_data import (
    _angle_constraints, _dihedral_constraints, _distance_constraints, _fixed_atom_indices,
    _molecule_collection, _selected_molecule_root, _serialize_molecule, _sync_atom_positions,
)
from .uff import (
    angle_bend_energy, bond_stretch_energy, out_of_plane_energy, partial_energy, torsion_energy,
    type_molecule, van_der_waals_energy,
)


class BLENDMOL_OT_import_v2000(Operator):
    bl_idname = "blendmol.import_v2000"
    bl_label = "Import MOL (V2000)"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mol", options={"HIDDEN"})

    def execute(self, context):
        try:
            molecule = read_v2000(self.filepath)
        except (OSError, MolfileError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _create_molecule(molecule, context.scene.blend_mol_settings, self.filepath)
        self.report({"INFO"}, f"Imported {len(molecule.atoms)} atoms and {len(molecule.bonds)} bonds")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_import_xyz(Operator):
    bl_idname = "blendmol.import_xyz"
    bl_label = "Import XYZ"
    bl_description = "Import XYZ atom coordinates without inferring bonds or implicit hydrogens"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xyz", options={"HIDDEN"})

    def execute(self, context):
        try:
            molecule = read_xyz(self.filepath)
        except (OSError, MolfileError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _create_molecule(
            molecule, context.scene.blend_mol_settings, self.filepath, auto_implicit_hydrogens=False,
        )
        self.report({"INFO"}, f"Imported {len(molecule.atoms)} XYZ atom coordinates; add bonds manually")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_import_pdb(Operator):
    bl_idname = "blendmol.import_pdb"
    bl_label = "Import PDB"
    bl_description = "Import PDB atom coordinates and any CONECT single bonds"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.pdb", options={"HIDDEN"})

    def execute(self, context):
        try:
            molecule = read_pdb(self.filepath)
        except (OSError, MolfileError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _create_molecule(
            molecule, context.scene.blend_mol_settings, self.filepath, auto_implicit_hydrogens=False,
        )
        self.report({"INFO"}, f"Imported {len(molecule.atoms)} PDB atoms and {len(molecule.bonds)} CONECT bonds")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_import_sdf(Operator):
    bl_idname = "blendmol.import_sdf"
    bl_label = "Import SDF (first record)"
    bl_description = "Import the first V2000 structure record in an SD file"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.sdf", options={"HIDDEN"})

    def execute(self, context):
        try:
            molecule = read_sdf(self.filepath)
        except (OSError, MolfileError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _create_molecule(molecule, context.scene.blend_mol_settings, self.filepath)
        self.report({"INFO"}, f"Imported first SDF record: {len(molecule.atoms)} atoms and {len(molecule.bonds)} bonds")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_import_smiles(Operator):
    bl_idname = "blendmol.import_smiles"
    bl_label = "Create from SMILES"
    bl_description = "Create a molecule from basic organic SMILES; it receives a 2D starting layout"
    bl_options = {"REGISTER", "UNDO"}

    smiles: StringProperty(name="SMILES", default="CC")
    molecule_name: StringProperty(name="Molecule Name", default="SMILES molecule")

    def execute(self, context):
        try:
            molecule = read_smiles(self.smiles, self.molecule_name.strip() or "SMILES molecule")
        except MolfileError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _create_molecule(molecule, context.scene.blend_mol_settings, f"SMILES: {self.smiles}")
        self.report({"INFO"}, f"Created {len(molecule.atoms)} atoms and {len(molecule.bonds)} bonds from SMILES")
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=360)


class BLENDMOL_OT_export_v2000(Operator):
    bl_idname = "blendmol.export_v2000"
    bl_label = "Export Selected MOL (V2000)"
    bl_description = "Export the selected Blend Mol molecule as an MDL V2000 MOL file"
    bl_options = {"REGISTER"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mol", options={"HIDDEN"})

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
            write_v2000(molecule, self.filepath)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot export MOL: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(molecule)
        self.report({"INFO"}, f"Exported {len(molecule.atoms)} atoms and {len(molecule.bonds)} bonds")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_export_xyz(Operator):
    bl_idname = "blendmol.export_xyz"
    bl_label = "Export Selected XYZ"
    bl_description = "Export selected Blend Mol atom elements and coordinates as an XYZ file"
    bl_options = {"REGISTER"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xyz", options={"HIDDEN"})

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
            write_xyz(molecule, self.filepath)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot export XYZ: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(molecule)
        self.report({"INFO"}, f"Exported {len(molecule.atoms)} atom coordinates")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_export_pdb(Operator):
    bl_idname = "blendmol.export_pdb"
    bl_label = "Export Selected PDB"
    bl_description = (
        "Export selected Blend Mol atom coordinates and connectivity as a PDB file; "
        "bond order and formal charges are not preserved, matching PDB Import"
    )
    bl_options = {"REGISTER"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.pdb", options={"HIDDEN"})

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
            write_pdb(molecule, self.filepath)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot export PDB: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(molecule)
        self.report({"INFO"}, f"Exported {len(molecule.atoms)} atoms and {len(molecule.bonds)} CONECT bonds")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_export_sdf(Operator):
    bl_idname = "blendmol.export_sdf"
    bl_label = "Export Selected SDF"
    bl_description = "Export the selected molecule as a single-record SD file (V2000 block + $$$$)"
    bl_options = {"REGISTER"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.sdf", options={"HIDDEN"})

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
            write_sdf(molecule, self.filepath)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot export SDF: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(molecule)
        self.report({"INFO"}, f"Exported {len(molecule.atoms)} atoms and {len(molecule.bonds)} bonds")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_export_uff_report(Operator):
    bl_idname = "blendmol.export_uff_report"
    bl_label = "Export Partial UFF Report"
    bl_description = "Export UFF typing, implemented energy terms, and the last minimization log as JSON"
    bl_options = {"REGISTER"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
            typing = type_molecule(molecule)
            reports = {
                "bond_stretch": bond_stretch_energy(molecule),
                "angle_bend": angle_bend_energy(molecule),
                "torsion": torsion_energy(molecule),
                "out_of_plane": out_of_plane_energy(molecule),
                "van_der_waals": van_der_waals_energy(molecule),
                "partial_total": partial_energy(molecule),
            }
            last_run = json.loads(root["blend_mol_uff_last_run"]) if "blend_mol_uff_last_run" in root else None
            coordinate_snapshots = None
            if "blend_mol_uff_before_positions" in root and "blend_mol_uff_after_positions" in root:
                coordinate_snapshots = {
                    "before": json.loads(root["blend_mol_uff_before_positions"]),
                    "after": json.loads(root["blend_mol_uff_after_positions"]),
                }
            benchmark = (
                json.loads(root["blend_mol_uff_benchmark"])
                if "blend_mol_uff_benchmark" in root else None
            )
            payload = {
                "format": "Blend Mol partial UFF report",
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "molecule": json.loads(_serialize_molecule(molecule)),
                "atom_types": list(typing.types),
                "typing_messages": list(typing.messages),
                "energy_terms": {
                    name: {
                        "energy_kcal_mol": report.energy_kcal_mol,
                        "terms": getattr(report, "terms", None),
                        "messages": list(report.messages),
                    }
                    for name, report in reports.items()
                },
                "fixed_atom_indices": sorted(_fixed_atom_indices(root)),
                "distance_constraints": [
                    {"atom_a": first, "atom_b": second, "target_angstrom": target,
                     "strength_kcal_mol_angstrom2": strength}
                    for first, second, target, strength in _distance_constraints(root)
                ],
                "angle_constraints": [
                    {"outer_a": first, "center": center, "outer_b": second,
                     "target_degrees": math.degrees(target),
                     "strength_kcal_mol_radian2": strength}
                    for first, center, second, target, strength in _angle_constraints(root)
                ],
                "dihedral_constraints": [
                    {"atom_a": first, "atom_b": center_a, "atom_c": center_b,
                     "atom_d": fourth, "target_degrees": math.degrees(target),
                     "strength_kcal_mol_radian2": strength}
                    for first, center_a, center_b, fourth, target, strength
                    in _dihedral_constraints(root)
                ],
                "last_minimization": last_run,
                "last_minimization_coordinates": coordinate_snapshots,
                "performance_benchmark": benchmark,
            }
            Path(self.filepath).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot export UFF report: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(molecule)
        self.report({"INFO"}, "Exported partial UFF report")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}
