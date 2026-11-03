"""Import and export operators."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import Operator

from .display import _create_molecule
from .molfile import (
    MolfileError, perceive_bonds, read_mol, read_pdb, read_sdf_records, read_smiles, read_xyz,
    SdfRecord, write_mol, write_pdb, write_sdf_records, write_xyz,
)
from .scene_data import (
    _angle_constraints, _dihedral_constraints, _distance_constraints, _fixed_atom_indices,
    _molecule_collection, _selected_molecule_root, _selected_molecule_roots, _serialize_molecule,
    _sync_atom_positions,
)

MOLFILE_VERSION_ITEMS = (
    ("AUTO", "Auto", "V2000, or V3000 when the molecule has more than 999 atoms or bonds"),
    ("V2000", "V2000", "Classic fixed-column format (up to 999 atoms and 999 bonds)"),
    ("V3000", "V3000", "Extended format with no atom or bond count limit"),
)
# Gap left between molecules when SDF records are arranged in a row.
_SDF_ARRANGE_GAP = 3.0


def _sdf_data(root) -> tuple[tuple[str, str], ...]:
    """SD data items stored on a molecule root by SDF Import, if any."""
    try:
        return tuple((str(name), str(value)) for name, value in json.loads(root["blend_mol_sdf_data"]))
    except (KeyError, TypeError, ValueError):
        return ()
from .uff import (
    angle_bend_energy, bond_stretch_energy, out_of_plane_energy, partial_energy, torsion_energy,
    type_molecule, van_der_waals_energy,
)


class BLENDMOL_OT_import_v2000(Operator):
    bl_idname = "blendmol.import_v2000"
    bl_label = "Import MOL"
    bl_description = (
        "Import an MDL MOL file (V2000 or V3000); aromatic bonds are converted to single/double"
    )
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mol", options={"HIDDEN"})

    def execute(self, context):
        try:
            molecule = read_mol(self.filepath)
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
    bl_description = (
        "Import XYZ atom coordinates, optionally adding single bonds from interatomic distances; "
        "no implicit hydrogens are added"
    )
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xyz", options={"HIDDEN"})
    perceive_bonds: BoolProperty(
        name="Perceive Bonds",
        description="Add single bonds between atoms closer than their covalent radii plus 0.45 Å",
        default=True,
    )

    def execute(self, context):
        try:
            molecule = read_xyz(self.filepath)
        except (OSError, MolfileError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if self.perceive_bonds:
            molecule = perceive_bonds(molecule)
        _create_molecule(
            molecule, context.scene.blend_mol_settings, self.filepath, auto_implicit_hydrogens=False,
        )
        if self.perceive_bonds:
            self.report(
                {"INFO"},
                f"Imported {len(molecule.atoms)} XYZ atoms with {len(molecule.bonds)} perceived single bonds",
            )
        else:
            self.report({"INFO"}, f"Imported {len(molecule.atoms)} XYZ atom coordinates; add bonds manually")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_import_pdb(Operator):
    bl_idname = "blendmol.import_pdb"
    bl_label = "Import PDB"
    bl_description = (
        "Import PDB atom coordinates and CONECT single bonds, or bonds from distances when the "
        "file has no CONECT records"
    )
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.pdb", options={"HIDDEN"})
    perceive_bonds: BoolProperty(
        name="Perceive Bonds Without CONECT",
        description="When the file has no CONECT records, add single bonds from interatomic distances",
        default=True,
    )

    def execute(self, context):
        try:
            molecule = read_pdb(self.filepath, perceive_if_no_conect=self.perceive_bonds)
        except (OSError, MolfileError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _create_molecule(
            molecule, context.scene.blend_mol_settings, self.filepath, auto_implicit_hydrogens=False,
        )
        self.report({"INFO"}, f"Imported {len(molecule.atoms)} PDB atoms and {len(molecule.bonds)} bonds")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class BLENDMOL_OT_import_sdf(Operator):
    bl_idname = "blendmol.import_sdf"
    bl_label = "Import SDF"
    bl_description = (
        "Import structure records (V2000 or V3000) from an SD file, one molecule per record, "
        "keeping each record's data items"
    )
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.sdf;*.sd", options={"HIDDEN"})
    max_records: IntProperty(
        name="Max Records", description="Read at most this many records from the start of the file",
        default=100, min=1, max=100000,
    )
    skip_invalid: BoolProperty(
        name="Skip Invalid Records",
        description="Skip records that cannot be read instead of cancelling the import",
        default=True,
    )
    arrange: BoolProperty(
        name="Arrange in a Row",
        description="Offset each molecule along X so records do not overlap; atom coordinates are unchanged",
        default=True,
    )

    def execute(self, context):
        try:
            result = read_sdf_records(self.filepath, self.max_records, self.skip_invalid)
        except (OSError, MolfileError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not result.records:
            number, reason = result.skipped[0]
            self.report({"ERROR"}, f"No readable SDF records (record {number}: {reason})")
            return {"CANCELLED"}
        cursor = None
        for record in result.records:
            root = _create_molecule(record.molecule, context.scene.blend_mol_settings, self.filepath)
            if record.data:
                root["blend_mol_sdf_data"] = json.dumps([list(item) for item in record.data])
            if not self.arrange:
                continue
            # Only the root moves; stored atom coordinates stay as in the file.
            xs = [atom.position[0] for atom in record.molecule.atoms] or [0.0]
            if cursor is not None:
                root.location.x = cursor - min(xs)
            cursor = root.location.x + max(xs) + _SDF_ARRANGE_GAP
        message = f"Imported {len(result.records)} SDF record(s)"
        if result.skipped:
            message += f"; skipped {len(result.skipped)} unreadable (first: record {result.skipped[0][0]})"
        if result.remaining:
            message += f"; {result.remaining} more not read (Max Records)"
        self.report({"WARNING"} if result.skipped else {"INFO"}, message)
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
    bl_label = "Export Selected MOL"
    bl_description = "Export the selected Blend Mol molecule as an MDL MOL file (V2000 or V3000)"
    bl_options = {"REGISTER"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mol", options={"HIDDEN"})
    version: EnumProperty(name="Format", items=MOLFILE_VERSION_ITEMS, default="AUTO")

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
            written = write_mol(molecule, self.filepath, self.version)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot export MOL: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(molecule)
        self.report({"INFO"}, f"Exported {len(molecule.atoms)} atoms and {len(molecule.bonds)} bonds as {written}")
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
    bl_description = (
        "Export the selected molecules as an SD file, one record each, with any data items "
        "that SDF Import stored on them"
    )
    bl_options = {"REGISTER"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.sdf;*.sd", options={"HIDDEN"})
    version: EnumProperty(name="Format", items=MOLFILE_VERSION_ITEMS, default="AUTO")
    all_selected: BoolProperty(
        name="All Selected Molecules",
        description="Write every molecule with a selected atom, bond or root; otherwise only the active one",
        default=True,
    )

    def execute(self, context):
        if self.all_selected:
            roots = _selected_molecule_roots(context)
        else:
            active = _selected_molecule_root(context)
            roots = [active] if active is not None else []
        roots = [root for root in roots if _molecule_collection(root) is not None]
        if not roots:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecules = [_sync_atom_positions(root, _molecule_collection(root)) for root in roots]
            write_sdf_records(
                [SdfRecord(molecule, _sdf_data(root)) for root, molecule in zip(roots, molecules)],
                self.filepath, self.version,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot export SDF: {exc}")
            return {"CANCELLED"}
        for root, molecule in zip(roots, molecules):
            root["blend_mol_data"] = _serialize_molecule(molecule)
        atoms = sum(len(molecule.atoms) for molecule in molecules)
        self.report({"INFO"}, f"Exported {len(molecules)} SDF record(s), {atoms} atoms")
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
