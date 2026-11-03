"""Mesh creation and display updates for Blend Mol molecules."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import bpy
from mathutils import Matrix, Vector

from .elements import ELEMENT_COLORS, ELEMENT_RADII, UNKNOWN_COLOR, UNKNOWN_DISPLAY_RADIUS
from .molfile import _target_valence
from .scene_data import _serialize_molecule

if TYPE_CHECKING:
    # Annotation-only; properties.py imports this module, so a runtime import
    # would be circular.
    from .molfile import Molecule
    from .properties import BlendMolElementColor, BlendMolSettings


def _palette_item(settings: BlendMolSettings, element: str) -> BlendMolElementColor:
    for item in settings.element_colors:
        if item.element == element:
            return item
    item = settings.element_colors.add()
    item.element = element
    item.color = ELEMENT_COLORS.get(element, UNKNOWN_COLOR)
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
    chirality: str | None = None,
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
    if chirality is not None:
        obj["blend_mol_chirality"] = chirality
    _smooth_mesh(obj)
    obj.data.materials.append(material)
    return obj


def _cylinder(
    name: str, start: Vector, end: Vector, radius: float, color, material, collection, sides: int,
    bond_index: int | None = None, bond_order: int | None = None, implicit: bool = False,
    length_segments: int = 1, bond_stereo: str | None = None,
) -> bpy.types.Object:
    direction = end - start
    rings = max(1, length_segments)
    vertices = []
    for ring in range(rings + 1):
        z = direction.length * (ring / rings - 0.5)
        for side in range(sides):
            angle = 2 * math.pi * side / sides
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    faces = []
    for ring in range(rings):
        first = ring * sides
        second = (ring + 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            faces.append((first + side, first + next_side, second + next_side, second + side))
    faces.append(tuple(reversed(range(sides))))
    final_ring = rings * sides
    faces.append(tuple(final_ring + side for side in range(sides)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    _link_object(obj, collection)
    obj.location = (start + end) / 2
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction)
    obj.color = color
    obj["blend_mol_kind"] = "bond"
    obj["blend_mol_implicit"] = implicit
    obj["blend_mol_base_length"] = direction.length
    if bond_index is not None:
        obj["blend_mol_bond_index"] = bond_index
        obj["blend_mol_bond_order"] = bond_order
        if bond_stereo is not None:
            obj["blend_mol_bond_stereo"] = bond_stereo
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
        if count == 4 and not neighbors[atom_index]:
            directions = (
                Vector((1.0, 1.0, 1.0)).normalized(),
                Vector((1.0, -1.0, -1.0)).normalized(),
                Vector((-1.0, 1.0, -1.0)).normalized(),
                Vector((-1.0, -1.0, 1.0)).normalized(),
            )
        elif count in (1, 2, 3) and len(neighbors[atom_index]) == 1:
            # Arrange the missing hydrogen(s) tetrahedrally around the bond,
            # not along `direction` itself (exactly opposite the one existing
            # bond). For count == 3 the previous planar fan could produce
            # collinear H-C-C-H torsions; for count == 1 or 2 (e.g. any -OH,
            # -SH, or secondary-amine -NH- missing its last hydrogen)
            # `direction` unmodified placed the hydrogen exactly 180 degrees
            # from the parent bond. That is a genuine (non-removable)
            # collinear singularity for any non-linear parent type -- O_3's
            # own ~104.5 degree bend is nowhere near 180 -- and UFF's angle
            # gradient correctly refuses to evaluate it, reported as "UFF:
            # collinear atoms have no stable angle gradient". Reproduced on
            # every -OH in a SMILES-imported sugar ring.
            reference = Vector((0.0, 0.0, 1.0))
            if abs(direction.dot(reference)) > 0.9:
                reference = Vector((0.0, 1.0, 0.0))
            first_axis = direction.cross(reference).normalized()
            second_axis = direction.cross(first_axis).normalized()
            axial = 1.0 / 3.0
            radial = math.sqrt(1.0 - axial ** 2)
            directions = tuple(
                (direction * axial + first_axis * (radial * math.cos(2 * math.pi * index / count))
                 + second_axis * (radial * math.sin(2 * math.pi * index / count))).normalized()
                for index in range(count)
            )
        else:
            directions = tuple(
                Matrix.Rotation(2 * math.pi * index / count, 4, "Z") @ direction
                for index in range(count)
            )
        for hydrogen_direction in directions:
            hydrogens.append((atom_index, origin + hydrogen_direction * 1.09))
    return hydrogens


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
            ELEMENT_RADII.get(atom.element, UNKNOWN_DISPLAY_RADIUS) * settings.atom_scale,
            color, atom_material, collection, atom.element, settings.atom_segments, settings.atom_rings,
            atom_index=index, chirality=atom.chirality,
        )

    for index, bond in enumerate(molecule.bonds):
        start, end = Vector(molecule.atoms[bond.atom_a].position), Vector(molecule.atoms[bond.atom_b].position)
        for segment_index, offset in enumerate(_bond_offsets(start, end, bond.order, settings.bond_radius)):
            _cylinder(f"Bond_{index + 1:03d}_{segment_index + 1}", start + offset, end + offset,
                      settings.bond_radius, (0.30, 0.30, 0.30, 1.0), bond_material, collection,
                      settings.bond_sides, bond_index=index, bond_order=bond.order,
                      length_segments=settings.bond_length_segments, bond_stereo=bond.stereo)

    if settings.show_implicit_hydrogens and root.get("blend_mol_auto_implicit_hydrogens", True):
        for index, (parent, position) in enumerate(_implicit_hydrogens(molecule)):
            _cylinder(f"H_Bond_{index + 1:03d}", Vector(molecule.atoms[parent].position), position,
                      settings.bond_radius * 0.72, (0.55, 0.55, 0.55, 1.0), bond_material, collection,
                      settings.bond_sides, implicit=True, length_segments=1)
            _sphere(f"H_{index + 1:03d}", position, ELEMENT_RADII["H"] * settings.atom_scale,
                    _element_color(settings, "H"), atom_material, collection, "H",
                    settings.atom_segments, settings.atom_rings, implicit=True)
    for obj in collection.objects:
        if obj != root:
            obj.parent = root


def _update_explicit_display_geometry(
    root: bpy.types.Object, collection: bpy.types.Collection, molecule: Molecule,
    settings: BlendMolSettings,
) -> bool:
    """Update an explicit molecule in place; return False when a rebuild is required."""

    if any(obj.get("blend_mol_implicit") for obj in collection.objects):
        return False
    atom_objects = {
        obj.get("blend_mol_atom_index"): obj
        for obj in collection.objects
        if obj.get("blend_mol_kind") == "atom" and not obj.get("blend_mol_implicit")
    }
    if set(atom_objects) != set(range(len(molecule.atoms))):
        return False
    bond_objects: dict[int, list[bpy.types.Object]] = {}
    for obj in collection.objects:
        if obj.get("blend_mol_kind") != "bond" or obj.get("blend_mol_implicit"):
            continue
        bond_index = obj.get("blend_mol_bond_index")
        if bond_index is None or "blend_mol_base_length" not in obj:
            return False
        bond_objects.setdefault(bond_index, []).append(obj)
    if set(bond_objects) != set(range(len(molecule.bonds))):
        return False
    for index, atom in enumerate(molecule.atoms):
        atom_objects[index].location = atom.position
    for bond_index, bond in enumerate(molecule.bonds):
        objects = sorted(bond_objects[bond_index], key=lambda obj: obj.name)
        if len(objects) != bond.order:
            return False
        start = Vector(molecule.atoms[bond.atom_a].position)
        end = Vector(molecule.atoms[bond.atom_b].position)
        direction = end - start
        if direction.length < 1e-10:
            return False
        offsets = _bond_offsets(start, end, bond.order, settings.bond_radius)
        for obj, offset in zip(objects, offsets):
            obj.location = (start + end) / 2 + offset
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction)
            base_length = float(obj["blend_mol_base_length"])
            if base_length < 1e-10:
                return False
            obj.scale.z = direction.length / base_length
    root["blend_mol_atom_count"] = len(molecule.atoms)
    root["blend_mol_bond_count"] = len(molecule.bonds)
    return True


def _create_molecule(
    molecule: Molecule, settings: BlendMolSettings, source_path: str, auto_implicit_hydrogens: bool = True,
) -> bpy.types.Object:
    collection = bpy.data.collections.new(f"{molecule.name} atoms and bonds")
    bpy.context.collection.children.link(collection)
    root = bpy.data.objects.new(molecule.name, None)
    _link_object(root, collection)
    root["blend_mol_source"] = source_path
    root["blend_mol_managed"] = True
    root["blend_mol_data_version"] = 1
    root["blend_mol_revision"] = 1
    root["blend_mol_data"] = _serialize_molecule(molecule)
    root["blend_mol_collection"] = collection.name
    root["blend_mol_auto_implicit_hydrogens"] = auto_implicit_hydrogens
    _render_molecule(root, collection, molecule, settings)
    return root
