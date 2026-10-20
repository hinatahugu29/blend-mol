"""Blend Mol: an initial V2000 MOL importer and molecular display tool."""

from __future__ import annotations

import json
import math
import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    IntProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Matrix, Vector

from .molfile import Atom, Bond, Molecule, MolfileError, read_v2000
from .uff import angle_bend_energy, bond_stretch_energy, type_molecule


ELEMENT_COLORS = {
    "H": (0.92, 0.92, 0.92, 1.0),
    "C": (0.18, 0.18, 0.18, 1.0),
    "N": (0.12, 0.26, 0.80, 1.0),
    "O": (0.82, 0.08, 0.08, 1.0),
    "P": (1.00, 0.50, 0.00, 1.0),
    "S": (0.92, 0.78, 0.05, 1.0),
    "F": (0.15, 0.75, 0.20, 1.0),
    "Cl": (0.10, 0.60, 0.15, 1.0),
    "Br": (0.45, 0.12, 0.05, 1.0),
    "I": (0.36, 0.08, 0.62, 1.0),
}
ELEMENT_RADII = {"H": 0.22, "C": 0.34, "N": 0.33, "O": 0.31, "P": 0.38, "S": 0.38}
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07, "S": 1.05,
                   "F": 0.57, "Cl": 1.02, "Br": 1.20, "I": 1.39}
TYPICAL_VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
ELEMENT_ITEMS = tuple((element, element, "") for element in ELEMENT_COLORS)


def _update_element_color(item, _context) -> None:
    """Update every displayed atom of an element as soon as its colour changes."""
    if not item.element:
        return
    for obj in bpy.data.objects:
        if obj.get("blend_mol_kind") == "atom" and obj.get("blend_mol_element") == item.element:
            obj.color = item.color


class BlendMolElementColor(PropertyGroup):
    element: StringProperty()
    color: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.65, 0.25, 0.65, 1.0),
        update=_update_element_color,
    )


class BlendMolSettings(PropertyGroup):
    atom_scale: FloatProperty(name="Atom Scale", default=1.0, min=0.1, max=4.0)
    bond_radius: FloatProperty(name="Bond Radius", default=0.09, min=0.01, max=0.5)
    atom_segments: IntProperty(name="Atom Segments", default=24, min=8, max=64)
    atom_rings: IntProperty(name="Atom Rings", default=12, min=4, max=32)
    bond_sides: IntProperty(name="Bond Sides", default=16, min=6, max=64)
    bond_order: IntProperty(name="Bond Order", default=1, min=1, max=3)
    formal_charge: IntProperty(name="Formal Charge", default=0, min=-3, max=3)
    relax_iterations: IntProperty(name="Relax Iterations", default=60, min=1, max=500)
    subdivision_levels: IntProperty(name="Subdivision Levels", default=1, min=1, max=3)
    roughness: FloatProperty(name="Roughness", default=0.34, min=0.0, max=1.0)
    metallic: FloatProperty(name="Metallic", default=0.0, min=0.0, max=1.0)
    show_implicit_hydrogens: BoolProperty(name="Show Implicit Hydrogens", default=True)
    edit_element: EnumProperty(name="Element", items=ELEMENT_ITEMS, default="C")
    element_colors: CollectionProperty(type=BlendMolElementColor)


def _palette_item(settings: BlendMolSettings, element: str) -> BlendMolElementColor:
    for item in settings.element_colors:
        if item.element == element:
            return item
    item = settings.element_colors.add()
    item.element = element
    item.color = ELEMENT_COLORS.get(element, (0.65, 0.25, 0.65, 1.0))
    return item


def _element_color(settings: BlendMolSettings, element: str) -> tuple[float, float, float, float]:
    return tuple(_palette_item(settings, element).color)


def _target_valence(atom: Atom) -> int | None:
    """Common valence rule used for implicit hydrogens and basic validation."""
    if atom.element == "N" and atom.formal_charge > 0:
        return 4
    if atom.element == "O" and atom.formal_charge < 0:
        return 1
    return TYPICAL_VALENCE.get(atom.element)


def _managed_atom_elements() -> dict[str, int]:
    elements: dict[str, int] = {}
    for obj in bpy.data.objects:
        if obj.get("blend_mol_kind") != "atom":
            continue
        element = obj.get("blend_mol_element")
        if element:
            elements[element] = elements.get(element, 0) + 1
    return elements


def _material(name: str, roughness: float, metallic: float) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    object_info = nodes.new("ShaderNodeObjectInfo")
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    links.new(object_info.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _link_object(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    collection.objects.link(obj)


def _smooth_mesh(obj: bpy.types.Object) -> None:
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def _sphere(
    name: str, location: Vector, radius: float, color, material, collection, element: str,
    segments: int, rings: int, atom_index: int | None = None, implicit: bool = False,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=radius, location=location)
    obj = bpy.context.object
    for linked in list(obj.users_collection):
        linked.objects.unlink(obj)
    _link_object(obj, collection)
    obj.name = name
    obj.color = color
    obj["blend_mol_kind"] = "atom"
    obj["blend_mol_element"] = element
    obj["blend_mol_implicit"] = implicit
    if atom_index is not None:
        obj["blend_mol_atom_index"] = atom_index
    _smooth_mesh(obj)
    obj.data.materials.append(material)
    return obj


def _cylinder(
    name: str, start: Vector, end: Vector, radius: float, color, material, collection, sides: int,
    bond_index: int | None = None, bond_order: int | None = None, implicit: bool = False,
) -> bpy.types.Object:
    direction = end - start
    bpy.ops.mesh.primitive_cylinder_add(vertices=sides, radius=radius, depth=direction.length, location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction)
    for linked in list(obj.users_collection):
        linked.objects.unlink(obj)
    _link_object(obj, collection)
    obj.name = name
    obj.color = color
    obj["blend_mol_kind"] = "bond"
    obj["blend_mol_implicit"] = implicit
    if bond_index is not None:
        obj["blend_mol_bond_index"] = bond_index
        obj["blend_mol_bond_order"] = bond_order
    _smooth_mesh(obj)
    obj.data.materials.append(material)
    return obj


def _bond_offsets(start: Vector, end: Vector, order: int, radius: float) -> tuple[Vector, ...]:
    if order == 1:
        return (Vector(),)
    direction = (end - start).normalized()
    perpendicular = direction.cross(Vector((0.0, 0.0, 1.0)))
    if perpendicular.length < 0.001:
        perpendicular = direction.cross(Vector((0.0, 1.0, 0.0)))
    perpendicular.normalize()
    distance = radius * 1.8
    if order == 2:
        return (-perpendicular * distance, perpendicular * distance)
    return (-perpendicular * distance * 2, Vector(), perpendicular * distance * 2)


def _implicit_hydrogens(molecule: Molecule) -> list[tuple[int, Vector]]:
    bond_order_sum = [0] * len(molecule.atoms)
    neighbors: list[list[int]] = [[] for _ in molecule.atoms]
    for bond in molecule.bonds:
        bond_order_sum[bond.atom_a] += bond.order
        bond_order_sum[bond.atom_b] += bond.order
        neighbors[bond.atom_a].append(bond.atom_b)
        neighbors[bond.atom_b].append(bond.atom_a)

    hydrogens = []
    for atom_index, atom in enumerate(molecule.atoms):
        count = max(0, (_target_valence(atom) or 0) - bond_order_sum[atom_index])
        if not count:
            continue
        origin = Vector(atom.position)
        direction = Vector()
        for neighbor_index in neighbors[atom_index]:
            direction += (Vector(molecule.atoms[neighbor_index].position) - origin).normalized()
        direction = -direction
        if direction.length < 0.001:
            direction = Vector((1.0, 0.0, 0.0))
        direction.normalize()
        for count_index in range(count):
            angle = (2 * math.pi * count_index / count) if count > 1 else 0.0
            rotated = Matrix.Rotation(angle, 4, "Z") @ direction
            hydrogens.append((atom_index, origin + rotated * 1.09))
    return hydrogens


def _serialize_molecule(molecule: Molecule) -> str:
    return json.dumps({
        "name": molecule.name,
        "atoms": [
            {"element": atom.element, "position": atom.position, "formal_charge": atom.formal_charge}
            for atom in molecule.atoms
        ],
        "bonds": [{"atom_a": bond.atom_a, "atom_b": bond.atom_b, "order": bond.order} for bond in molecule.bonds],
    }, separators=(",", ":"))


def _deserialize_molecule(payload: str) -> Molecule:
    data = json.loads(payload)
    return Molecule(
        name=data["name"],
        atoms=tuple(
            Atom(atom["element"], tuple(atom["position"]), atom.get("formal_charge", 0))
            for atom in data["atoms"]
        ),
        bonds=tuple(Bond(bond["atom_a"], bond["atom_b"], bond["order"]) for bond in data["bonds"]),
    )


def _molecule_collection(root: bpy.types.Object) -> bpy.types.Collection | None:
    collection_name = root.get("blend_mol_collection")
    return bpy.data.collections.get(collection_name) if collection_name else None


def _selected_molecule_root(context) -> bpy.types.Object | None:
    obj = context.active_object
    while obj:
        if obj.get("blend_mol_managed"):
            return obj
        obj = obj.parent
    return None


def _selected_explicit_atom(context) -> tuple[bpy.types.Object, bpy.types.Collection, int] | None:
    obj = context.active_object
    if obj is None or obj.get("blend_mol_kind") != "atom" or obj.get("blend_mol_implicit"):
        return None
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    atom_index = obj.get("blend_mol_atom_index")
    if root is None or collection is None or atom_index is None:
        return None
    return root, collection, atom_index


def _selected_explicit_atoms(context) -> tuple[bpy.types.Object, bpy.types.Collection, tuple[int, int]] | None:
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    if root is None or collection is None:
        return None
    indices = []
    for obj in context.selected_objects:
        if obj.parent != root or obj.get("blend_mol_kind") != "atom" or obj.get("blend_mol_implicit"):
            continue
        atom_index = obj.get("blend_mol_atom_index")
        if atom_index is not None:
            indices.append(atom_index)
    indices = tuple(sorted(set(indices)))
    if len(indices) != 2:
        return None
    return root, collection, indices


def _selected_explicit_bond(context) -> tuple[bpy.types.Object, bpy.types.Collection, int] | None:
    obj = context.active_object
    if obj is None or obj.get("blend_mol_kind") != "bond" or obj.get("blend_mol_implicit"):
        return None
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    bond_index = obj.get("blend_mol_bond_index")
    if root is None or collection is None or bond_index is None:
        return None
    return root, collection, bond_index


def _clear_molecule_display(root: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for obj in list(collection.objects):
        if obj != root:
            bpy.data.objects.remove(obj, do_unlink=True)


def _sync_atom_positions(root: bpy.types.Object, collection: bpy.types.Collection) -> Molecule:
    """Copy user-edited displayed atom positions back into the stored molecule graph."""
    molecule = _deserialize_molecule(root["blend_mol_data"])
    positions: dict[int, tuple[float, float, float]] = {}
    for obj in collection.objects:
        if obj.get("blend_mol_kind") != "atom" or obj.get("blend_mol_implicit"):
            continue
        atom_index = obj.get("blend_mol_atom_index")
        if atom_index is None:
            continue
        positions[atom_index] = tuple(obj.matrix_local.translation)
    expected = set(range(len(molecule.atoms)))
    if set(positions) != expected:
        raise ValueError("Displayed atoms no longer match the stored molecule data")
    updated_atoms = tuple(
        Atom(atom.element, positions[index], atom.formal_charge)
        for index, atom in enumerate(molecule.atoms)
    )
    updated = Molecule(molecule.name, updated_atoms, molecule.bonds)
    root["blend_mol_data"] = _serialize_molecule(updated)
    return updated


def _molecule_validation_messages(molecule: Molecule) -> list[str]:
    bond_order_sum = [0] * len(molecule.atoms)
    messages = []
    for bond in molecule.bonds:
        if bond.atom_a == bond.atom_b:
            messages.append(f"Bond connects atom {bond.atom_a + 1} to itself")
            continue
        if not 0 <= bond.atom_a < len(molecule.atoms) or not 0 <= bond.atom_b < len(molecule.atoms):
            messages.append("Bond references an atom that does not exist")
            continue
        bond_order_sum[bond.atom_a] += bond.order
        bond_order_sum[bond.atom_b] += bond.order
    for index, atom in enumerate(molecule.atoms):
        valence = _target_valence(atom)
        if valence is None:
            messages.append(f"Atom {index + 1} ({atom.element}) has no valence rule")
        elif bond_order_sum[index] > valence:
            messages.append(
                f"Atom {index + 1} ({atom.element}) has bond order {bond_order_sum[index]} > {valence}"
            )
    return messages


def _covalent_radius(element: str) -> float:
    return COVALENT_RADII.get(element, 0.8)


def _target_bond_length(atom_a: Atom, atom_b: Atom, order: int) -> float:
    """Return a display-oriented bond length; this is not a UFF parameterization."""
    return (_covalent_radius(atom_a.element) + _covalent_radius(atom_b.element)) * (
        1.0 - 0.12 * (order - 1)
    )


def _relax_display_geometry(molecule: Molecule, iterations: int, fixed_atoms: set[int]) -> Molecule:
    """Lightweight bond-length and overlap relaxation for editing, not force-field optimization."""
    positions = [Vector(atom.position) for atom in molecule.atoms]
    bonded_pairs = {tuple(sorted((bond.atom_a, bond.atom_b))) for bond in molecule.bonds}
    strength = 0.30
    for _ in range(iterations):
        for bond in molecule.bonds:
            start, end = positions[bond.atom_a], positions[bond.atom_b]
            delta = end - start
            distance = delta.length
            direction = delta.normalized() if distance > 1e-8 else Vector((1.0, 0.0, 0.0))
            target = _target_bond_length(molecule.atoms[bond.atom_a], molecule.atoms[bond.atom_b], bond.order)
            correction = direction * ((distance - target) * strength)
            movable_a = bond.atom_a not in fixed_atoms
            movable_b = bond.atom_b not in fixed_atoms
            if movable_a and movable_b:
                positions[bond.atom_a] += correction * 0.5
                positions[bond.atom_b] -= correction * 0.5
            elif movable_a:
                positions[bond.atom_a] += correction
            elif movable_b:
                positions[bond.atom_b] -= correction
        for atom_a in range(len(positions)):
            for atom_b in range(atom_a + 1, len(positions)):
                if (atom_a, atom_b) in bonded_pairs:
                    continue
                delta = positions[atom_b] - positions[atom_a]
                distance = delta.length
                minimum = 0.65 * (
                    _covalent_radius(molecule.atoms[atom_a].element)
                    + _covalent_radius(molecule.atoms[atom_b].element)
                )
                if distance >= minimum:
                    continue
                direction = delta.normalized() if distance > 1e-8 else Vector((1.0, 0.0, 0.0))
                correction = direction * ((minimum - distance) * strength)
                movable_a = atom_a not in fixed_atoms
                movable_b = atom_b not in fixed_atoms
                if movable_a and movable_b:
                    positions[atom_a] -= correction * 0.5
                    positions[atom_b] += correction * 0.5
                elif movable_a:
                    positions[atom_a] -= correction
                elif movable_b:
                    positions[atom_b] += correction
    atoms = tuple(
        Atom(atom.element, tuple(positions[index]), atom.formal_charge)
        for index, atom in enumerate(molecule.atoms)
    )
    return Molecule(molecule.name, atoms, molecule.bonds)


def _store_position_snapshot(root: bpy.types.Object, molecule: Molecule) -> None:
    root["blend_mol_position_snapshot"] = json.dumps(
        [atom.position for atom in molecule.atoms], separators=(",", ":")
    )


def _fixed_atom_indices(root: bpy.types.Object) -> set[int]:
    try:
        return set(json.loads(root.get("blend_mol_fixed_atoms", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def _store_fixed_atom_indices(root: bpy.types.Object, indices: set[int]) -> None:
    root["blend_mol_fixed_atoms"] = json.dumps(sorted(indices))


def _restore_position_snapshot(root: bpy.types.Object, molecule: Molecule) -> Molecule:
    positions = json.loads(root["blend_mol_position_snapshot"])
    if len(positions) != len(molecule.atoms):
        raise ValueError("The snapshot belongs to a different atom count")
    atoms = tuple(
        Atom(atom.element, tuple(positions[index]), atom.formal_charge)
        for index, atom in enumerate(molecule.atoms)
    )
    return Molecule(molecule.name, atoms, molecule.bonds)


def _render_molecule(
    root: bpy.types.Object, collection: bpy.types.Collection, molecule: Molecule, settings: BlendMolSettings,
) -> None:
    root["blend_mol_atom_count"] = len(molecule.atoms)
    root["blend_mol_bond_count"] = len(molecule.bonds)
    atom_material = _material("Blend Mol Atom", settings.roughness, settings.metallic)
    bond_material = _material("Blend Mol Bond", settings.roughness, settings.metallic)

    for index, atom in enumerate(molecule.atoms):
        color = _element_color(settings, atom.element)
        _sphere(
            f"{atom.element}_{index + 1:03d}", Vector(atom.position),
            ELEMENT_RADII.get(atom.element, 0.34) * settings.atom_scale,
            color, atom_material, collection, atom.element, settings.atom_segments, settings.atom_rings,
            atom_index=index,
        )

    for index, bond in enumerate(molecule.bonds):
        start, end = Vector(molecule.atoms[bond.atom_a].position), Vector(molecule.atoms[bond.atom_b].position)
        for segment_index, offset in enumerate(_bond_offsets(start, end, bond.order, settings.bond_radius)):
            _cylinder(f"Bond_{index + 1:03d}_{segment_index + 1}", start + offset, end + offset,
                      settings.bond_radius, (0.30, 0.30, 0.30, 1.0), bond_material, collection,
                      settings.bond_sides, bond_index=index, bond_order=bond.order)

    if settings.show_implicit_hydrogens:
        for index, (parent, position) in enumerate(_implicit_hydrogens(molecule)):
            _cylinder(f"H_Bond_{index + 1:03d}", Vector(molecule.atoms[parent].position), position,
                      settings.bond_radius * 0.72, (0.55, 0.55, 0.55, 1.0), bond_material, collection,
                      settings.bond_sides, implicit=True)
            _sphere(f"H_{index + 1:03d}", position, ELEMENT_RADII["H"] * settings.atom_scale,
                    _element_color(settings, "H"), atom_material, collection, "H",
                    settings.atom_segments, settings.atom_rings, implicit=True)
    for obj in collection.objects:
        if obj != root:
            obj.parent = root


def _create_molecule(molecule: Molecule, settings: BlendMolSettings, source_path: str) -> bpy.types.Object:
    collection = bpy.data.collections.new(f"{molecule.name} atoms and bonds")
    bpy.context.collection.children.link(collection)
    root = bpy.data.objects.new(molecule.name, None)
    _link_object(root, collection)
    root["blend_mol_source"] = source_path
    root["blend_mol_managed"] = True
    root["blend_mol_data_version"] = 1
    root["blend_mol_data"] = _serialize_molecule(molecule)
    root["blend_mol_collection"] = collection.name
    _render_molecule(root, collection, molecule, settings)
    return root


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


class BLENDMOL_OT_rebuild_display(Operator):
    bl_idname = "blendmol.rebuild_display"
    bl_label = "Rebuild Selected Molecule"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        collection = _molecule_collection(root)
        if collection is None or "blend_mol_data" not in root:
            self.report({"ERROR"}, "This molecule does not contain rebuildable Blend Mol data")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, molecule, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Rebuilt {molecule.name}")
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
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Added {element} at the 3D cursor")
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
        atoms[atom_index] = Atom(element, atoms[atom_index].position, atoms[atom_index].formal_charge)
        updated = Molecule(molecule.name, tuple(atoms), molecule.bonds)
        root["blend_mol_data"] = _serialize_molecule(updated)
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
        atoms[atom_index] = Atom(atom.element, atom.position, charge)
        updated = Molecule(molecule.name, tuple(atoms), molecule.bonds)
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Set atom {atom_index + 1} charge to {charge:+d}")
        return {"FINISHED"}


class BLENDMOL_OT_delete_atom(Operator):
    bl_idname = "blendmol.delete_atom"
    bl_label = "Delete Selected Atom"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select a non-implicit Blend Mol atom first")
            return {"CANCELLED"}
        root, collection, atom_index = selected
        molecule = _deserialize_molecule(root["blend_mol_data"])
        atoms = tuple(atom for index, atom in enumerate(molecule.atoms) if index != atom_index)
        bonds = []
        for bond in molecule.bonds:
            if atom_index in {bond.atom_a, bond.atom_b}:
                continue
            bonds.append(Bond(
                bond.atom_a - (bond.atom_a > atom_index),
                bond.atom_b - (bond.atom_b > atom_index),
                bond.order,
            ))
        updated = Molecule(molecule.name, atoms, tuple(bonds))
        root["blend_mol_data"] = _serialize_molecule(updated)
        fixed_atoms = {
            index - (index > atom_index)
            for index in _fixed_atom_indices(root)
            if index != atom_index
        }
        _store_fixed_atom_indices(root, fixed_atoms)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Deleted atom {atom_index + 1} and its connected bonds")
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
                bonds[index] = Bond(bond.atom_a, bond.atom_b, order)
                action = "Changed"
                break
        else:
            bonds.append(Bond(atom_a, atom_b, order))
            action = "Created"
        updated = Molecule(molecule.name, molecule.atoms, tuple(bonds))
        root["blend_mol_data"] = _serialize_molecule(updated)
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
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Deleted bond {bond_index + 1}")
        return {"FINISHED"}


class BLENDMOL_OT_relax_display_geometry(Operator):
    bl_idname = "blendmol.relax_display_geometry"
    bl_label = "Relax Display Geometry"
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
        _store_position_snapshot(root, molecule)
        updated = _relax_display_geometry(
            molecule, context.scene.blend_mol_settings.relax_iterations, _fixed_atom_indices(root)
        )
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, "Relaxed display geometry (not UFF)")
        return {"FINISHED"}


class BLENDMOL_OT_toggle_fixed_atom(Operator):
    bl_idname = "blendmol.toggle_fixed_atom"
    bl_label = "Toggle Fixed Selected Atom"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select a non-implicit Blend Mol atom first")
            return {"CANCELLED"}
        root, _collection, atom_index = selected
        fixed_atoms = _fixed_atom_indices(root)
        if atom_index in fixed_atoms:
            fixed_atoms.remove(atom_index)
            state = "unfixed"
        else:
            fixed_atoms.add(atom_index)
            state = "fixed"
        _store_fixed_atom_indices(root, fixed_atoms)
        self.report({"INFO"}, f"Atom {atom_index + 1} is now {state}")
        return {"FINISHED"}


class BLENDMOL_OT_save_position_snapshot(Operator):
    bl_idname = "blendmol.save_position_snapshot"
    bl_label = "Save Position Snapshot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        _store_position_snapshot(root, molecule)
        self.report({"INFO"}, f"Saved positions for {len(molecule.atoms)} atoms")
        return {"FINISHED"}


class BLENDMOL_OT_restore_position_snapshot(Operator):
    bl_idname = "blendmol.restore_position_snapshot"
    bl_label = "Restore Position Snapshot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None or "blend_mol_position_snapshot" not in root:
            self.report({"WARNING"}, "Select a molecule with a saved position snapshot first")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
            updated = _restore_position_snapshot(root, molecule)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot restore position snapshot: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        self.report({"INFO"}, f"Restored positions for {len(updated.atoms)} atoms")
        return {"FINISHED"}


class BLENDMOL_OT_reset_element_colors(Operator):
    bl_idname = "blendmol.reset_element_colors"
    bl_label = "Reset to CPK Colors"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.blend_mol_settings
        elements = _managed_atom_elements()
        for element in elements:
            _palette_item(settings, element).color = ELEMENT_COLORS.get(
                element, (0.65, 0.25, 0.65, 1.0)
            )
        self.report({"INFO"}, f"Reset colors for {sum(elements.values())} atoms")
        return {"FINISHED"}


class BLENDMOL_OT_add_subdivision(Operator):
    bl_idname = "blendmol.add_subdivision"
    bl_label = "Add Subdivision Surface"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        levels = context.scene.blend_mol_settings.subdivision_levels
        updated = 0
        for obj in bpy.data.objects:
            if obj.get("blend_mol_kind") not in {"atom", "bond"}:
                continue
            modifier = obj.modifiers.get("Blend Mol Subdivision")
            if modifier is None:
                modifier = obj.modifiers.new("Blend Mol Subdivision", "SUBSURF")
            modifier.subdivision_type = "CATMULL_CLARK"
            modifier.levels = levels
            modifier.render_levels = levels
            updated += 1
        self.report({"INFO"}, f"Added subdivision level {levels} to {updated} objects")
        return {"FINISHED"}


class BLENDMOL_PT_panel(Panel):
    bl_label = "Blend Mol"
    bl_idname = "BLENDMOL_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        layout.operator(BLENDMOL_OT_import_v2000.bl_idname, icon="IMPORT")
        layout.operator(BLENDMOL_OT_rebuild_display.bl_idname, icon="FILE_REFRESH")
        root = _selected_molecule_root(context)
        if root:
            edit_box = layout.box()
            edit_box.label(text=f"Editing: {root.name}")
            edit_box.label(text="Move atom objects, then synchronize")
            edit_box.operator(BLENDMOL_OT_sync_atom_positions.bl_idname, icon="CON_TRANSFORM")
            edit_box.separator()
            edit_box.prop(settings, "edit_element")
            edit_box.prop(settings, "formal_charge")
            edit_box.operator(BLENDMOL_OT_add_atom.bl_idname, icon="ADD")
            selected_atom = _selected_explicit_atom(context)
            if selected_atom:
                fixed_atoms = _fixed_atom_indices(root)
                state = "fixed" if selected_atom[2] in fixed_atoms else "movable"
                molecule = _deserialize_molecule(root["blend_mol_data"])
                charge = molecule.atoms[selected_atom[2]].formal_charge
                edit_box.label(text=f"Selected atom: {selected_atom[2] + 1} ({state}, {charge:+d})")
                edit_box.operator(BLENDMOL_OT_change_atom_element.bl_idname, icon="MODIFIER")
                edit_box.operator(BLENDMOL_OT_set_formal_charge.bl_idname, icon="ADD")
                edit_box.operator(BLENDMOL_OT_toggle_fixed_atom.bl_idname, icon="PINNED")
                edit_box.operator(BLENDMOL_OT_delete_atom.bl_idname, icon="TRASH")
            edit_box.separator()
            edit_box.label(text="Select two atoms to create or change a bond")
            edit_box.prop(settings, "bond_order")
            edit_box.operator(BLENDMOL_OT_set_bond.bl_idname, icon="CON_TRACKTO")
            selected_bond = _selected_explicit_bond(context)
            if selected_bond:
                edit_box.label(text=f"Selected bond: {selected_bond[2] + 1}")
                edit_box.operator(BLENDMOL_OT_delete_bond.bl_idname, icon="TRASH")
            try:
                molecule = _deserialize_molecule(root["blend_mol_data"])
                validation_messages = _molecule_validation_messages(molecule)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                validation_messages = ["Stored molecule data cannot be validated"]
            validation_box = edit_box.box()
            if validation_messages:
                validation_box.alert = True
                validation_box.label(text="Valence warnings", icon="ERROR")
                for message in validation_messages[:4]:
                    validation_box.label(text=message)
                if len(validation_messages) > 4:
                    validation_box.label(text=f"… and {len(validation_messages) - 4} more")
                validation_box.alert = False
            else:
                validation_box.label(text="Valence: no over-valence warnings", icon="CHECKMARK")
            try:
                uff_report = type_molecule(molecule)
            except (IndexError, TypeError, ValueError):
                uff_report = None
            uff_box = edit_box.box()
            uff_box.label(text="UFF preparation (typing only)")
            if uff_report is None:
                uff_box.alert = True
                uff_box.label(text="Cannot inspect stored molecule", icon="ERROR")
                uff_box.alert = False
            elif uff_report.ready:
                counts: dict[str, int] = {}
                for atom_type in uff_report.types:
                    counts[atom_type] = counts.get(atom_type, 0) + 1
                summary = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
                uff_box.label(text="Atom typing: ready", icon="CHECKMARK")
                uff_box.label(text=summary)
                stretch = bond_stretch_energy(molecule)
                if stretch.ready:
                    uff_box.label(
                        text=f"Bond stretch: {stretch.energy_kcal_mol:.3f} kcal/mol ({stretch.terms} terms)"
                    )
                else:
                    uff_box.label(text="Bond stretch: needs explicit hydrogens", icon="INFO")
                angle = angle_bend_energy(molecule)
                if angle.ready:
                    uff_box.label(
                        text=f"Angle bend: {angle.energy_kcal_mol:.3f} kcal/mol ({angle.terms} terms)"
                    )
                else:
                    uff_box.label(text="Angle bend: needs explicit hydrogens", icon="INFO")
                uff_box.label(text="No torsion/non-bonded terms or minimisation yet")
            else:
                uff_box.alert = True
                uff_box.label(text="Not ready for UFF", icon="ERROR")
                for message in uff_report.messages[:3]:
                    uff_box.label(text=message)
                if len(uff_report.messages) > 3:
                    uff_box.label(text=f"… and {len(uff_report.messages) - 3} more")
                uff_box.alert = False
            relax_box = edit_box.box()
            relax_box.label(text="Simple display relaxation (not UFF)")
            relax_box.prop(settings, "relax_iterations")
            relax_box.operator(BLENDMOL_OT_relax_display_geometry.bl_idname, icon="MOD_SMOOTH")
            relax_box.operator(BLENDMOL_OT_save_position_snapshot.bl_idname, icon="BOOKMARKS")
            if "blend_mol_position_snapshot" in root:
                relax_box.operator(BLENDMOL_OT_restore_position_snapshot.bl_idname, icon="LOOP_BACK")
        layout.separator()
        elements = _managed_atom_elements()
        if elements:
            element_box = layout.box()
            element_box.label(text="Elements in Blend Mol")
            for element, count in sorted(elements.items()):
                item = _palette_item(settings, element)
                element_box.prop(item, "color", text=f"{element} ({count})")
            element_box.operator(BLENDMOL_OT_reset_element_colors.bl_idname, icon="LOOP_BACK")
            layout.separator()
        layout.label(text="Display defaults for next import")
        layout.prop(settings, "atom_scale")
        layout.prop(settings, "bond_radius")
        layout.prop(settings, "roughness")
        layout.prop(settings, "metallic")
        layout.prop(settings, "show_implicit_hydrogens")
        layout.separator()
        mesh_box = layout.box()
        mesh_box.label(text="Mesh detail")
        mesh_box.label(text="Defaults for next import")
        mesh_box.prop(settings, "atom_segments")
        mesh_box.prop(settings, "atom_rings")
        mesh_box.prop(settings, "bond_sides")
        mesh_box.separator()
        mesh_box.prop(settings, "subdivision_levels")
        mesh_box.operator(BLENDMOL_OT_add_subdivision.bl_idname, icon="MOD_SUBSURF")


CLASSES = (
    BlendMolElementColor,
    BlendMolSettings,
    BLENDMOL_OT_import_v2000,
    BLENDMOL_OT_rebuild_display,
    BLENDMOL_OT_sync_atom_positions,
    BLENDMOL_OT_add_atom,
    BLENDMOL_OT_change_atom_element,
    BLENDMOL_OT_set_formal_charge,
    BLENDMOL_OT_delete_atom,
    BLENDMOL_OT_set_bond,
    BLENDMOL_OT_delete_bond,
    BLENDMOL_OT_relax_display_geometry,
    BLENDMOL_OT_toggle_fixed_atom,
    BLENDMOL_OT_save_position_snapshot,
    BLENDMOL_OT_restore_position_snapshot,
    BLENDMOL_OT_reset_element_colors,
    BLENDMOL_OT_add_subdivision,
    BLENDMOL_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.blend_mol_settings = PointerProperty(type=BlendMolSettings)


def unregister():
    del bpy.types.Scene.blend_mol_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
