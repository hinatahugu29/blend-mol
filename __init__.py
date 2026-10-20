"""Blend Mol: an initial V2000 MOL importer and molecular display tool."""

from __future__ import annotations

import json
import math
import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    IntProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Matrix, Vector

from .molfile import Atom, Bond, Molecule, MolfileError, read_v2000


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
TYPICAL_VALENCE = {"C": 4, "N": 3, "O": 2, "P": 3, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}


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
    subdivision_levels: IntProperty(name="Subdivision Levels", default=1, min=1, max=3)
    roughness: FloatProperty(name="Roughness", default=0.34, min=0.0, max=1.0)
    metallic: FloatProperty(name="Metallic", default=0.0, min=0.0, max=1.0)
    show_implicit_hydrogens: BoolProperty(name="Show Implicit Hydrogens", default=True)
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
        count = max(0, TYPICAL_VALENCE.get(atom.element, 0) - bond_order_sum[atom_index])
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
        "atoms": [{"element": atom.element, "position": atom.position} for atom in molecule.atoms],
        "bonds": [{"atom_a": bond.atom_a, "atom_b": bond.atom_b, "order": bond.order} for bond in molecule.bonds],
    }, separators=(",", ":"))


def _deserialize_molecule(payload: str) -> Molecule:
    data = json.loads(payload)
    return Molecule(
        name=data["name"],
        atoms=tuple(Atom(atom["element"], tuple(atom["position"])) for atom in data["atoms"]),
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


def _clear_molecule_display(root: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for obj in list(collection.objects):
        if obj != root:
            bpy.data.objects.remove(obj, do_unlink=True)


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
