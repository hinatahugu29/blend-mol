"""Blend Mol: an initial V2000 MOL importer and molecular display tool."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from itertools import permutations
from pathlib import Path
from threading import Event
from time import perf_counter
import bpy
from bpy.app.handlers import persistent
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

from .molfile import (
    Atom, Bond, Molecule, MolfileError, TYPICAL_VALENCE, _max_legitimate_valence, _target_valence,
    read_pdb, read_sdf, read_smiles, read_v2000, read_xyz,
    write_pdb, write_sdf, write_v2000, write_xyz,
)
from .uff import (
    _dihedral_angle, _gradient_step_metrics, angle_bend_energy, bond_stretch_energy,
    minimize_partial_energy, numeric_gradient_atom_indices, out_of_plane_energy, partial_energy,
    partial_objective_energy_gradient, prepare_uff_context, torsion_energy, type_molecule,
    van_der_waals_energy,
)
from .native_backend import backend_status, set_backend_preference
from .live_worker import LatestOnlyWorker, adapt_chunk_iterations, should_continue_relaxation


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
ELEMENT_ITEMS = tuple((element, element, "") for element in ELEMENT_COLORS)


def _update_element_color(item, _context) -> None:
    """Update every displayed atom of an element as soon as its colour changes."""
    if not item.element:
        return
    for obj in bpy.data.objects:
        if obj.get("blend_mol_kind") == "atom" and obj.get("blend_mol_element") == item.element:
            obj.color = item.color


def _update_uff_backend(settings, _context) -> None:
    set_backend_preference(settings.uff_backend)


def _update_live_material(settings, _context) -> None:
    """Re-apply roughness/metallic to the shared atom/bond materials immediately.

    Unlike atom_scale/bond_radius/show_implicit_hydrogens, these are plain
    material inputs rather than mesh geometry, so refreshing them on every
    slider step (via _material, defined further below but already in module
    globals by the time this runs) is cheap and needs no display rebuild.
    """
    _material("Blend Mol Atom", settings.roughness, settings.metallic)
    _material("Blend Mol Bond", settings.roughness, settings.metallic)


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
    bond_length_segments: IntProperty(name="Bond Length Segments", default=6, min=1, max=32)
    bond_order: IntProperty(name="Bond Order", default=1, min=1, max=3)
    formal_charge: IntProperty(name="Formal Charge", default=0, min=-3, max=3)
    relax_iterations: IntProperty(name="Relax Iterations", default=60, min=1, max=500)
    uff_iterations: IntProperty(name="UFF Iterations", default=30, min=1, max=300)
    uff_max_step: FloatProperty(
        name="UFF Max Atom Step", default=0.02, min=0.001, max=0.10,
        description="Maximum displacement of any atom in one accepted optimization step",
    )
    uff_gradient_tolerance: FloatProperty(
        name="Gradient Tolerance", default=1e-5, min=1e-8, max=1.0,
        precision=6, description="Stop when the largest movable Cartesian gradient component is below this value",
    )
    uff_energy_tolerance: FloatProperty(
        name="Energy Tolerance", default=1e-6, min=1e-10, max=1.0,
        precision=8, description="Stop when accepted energy improvement falls below this value",
    )
    uff_optimizer: EnumProperty(
        name="UFF Optimizer",
        items=(
            ("STEEPEST", "Steepest Descent", "Stable reference optimizer with backtracking"),
            ("FIRE2", "FIRE2", "Stateful inertial optimizer suited to interactive relaxation"),
            ("LBFGS", "L-BFGS", "Limited-memory quasi-Newton optimizer for final convergence"),
        ),
        default="LBFGS",
    )
    uff_live_steps: IntProperty(name="Interactive UFF Steps", default=1, min=1, max=10)
    uff_live_target_ms: FloatProperty(
        name="Live Compute Budget", default=40.0, min=5.0, max=500.0,
        description="Target milliseconds per background UFF update; step count adapts automatically",
    )
    uff_live_interval: FloatProperty(name="Live Update Interval", default=0.08, min=0.03, max=1.0)
    uff_live_auto_start: BoolProperty(
        name="Start Live UFF When Moving an Atom",
        default=False,
        description="Automatically relax surrounding atoms after moving one explicit atom; stops after settling",
    )
    uff_live_settle_updates: IntProperty(
        name="Maximum Settle Updates", default=75, min=1, max=500,
        description="Maximum background updates after each atom movement",
    )
    jitter_magnitude: FloatProperty(
        name="Jitter Magnitude (Å)", default=0.05, min=0.001, max=0.5, precision=3,
        description=(
            "Maximum random displacement per axis for Jitter All Atom Positions; "
            "an emergency escape from an exactly-degenerate starting geometry"
        ),
    )
    distance_constraint_target: FloatProperty(
        name="Target Distance (Å)", default=1.50, min=0.05, max=100.0, precision=3
    )
    distance_constraint_strength: FloatProperty(
        name="Constraint Strength", default=500.0, min=1.0, max=10000.0,
        description="Harmonic restraint strength in kcal/mol/Å²",
    )
    angle_constraint_target: FloatProperty(
        name="Target Angle (°)", default=109.5, min=1.0, max=179.0, precision=2
    )
    angle_constraint_strength: FloatProperty(
        name="Angle Strength", default=200.0, min=1.0, max=10000.0,
        description="Harmonic restraint strength in kcal/mol/radian²",
    )
    dihedral_constraint_target: FloatProperty(
        name="Target Dihedral (°)", default=180.0, min=-180.0, max=180.0, precision=2
    )
    dihedral_constraint_strength: FloatProperty(
        name="Dihedral Strength", default=100.0, min=1.0, max=10000.0,
        description="Periodic harmonic restraint strength in kcal/mol/radian²",
    )
    uff_backend: EnumProperty(
        name="UFF Compute Backend",
        items=(
            ("AUTO", "Auto", "Use Rust when available, otherwise Python"),
            ("PYTHON", "Python", "Always use the reference Python implementation"),
            ("RUST", "Rust", "Require the native Rust DLL and report an error if unavailable"),
        ),
        default="AUTO",
        update=_update_uff_backend,
    )
    subdivision_levels: IntProperty(name="Subdivision Levels", default=1, min=1, max=3)
    roughness: FloatProperty(
        name="Roughness", default=0.34, min=0.0, max=1.0, update=_update_live_material,
    )
    metallic: FloatProperty(
        name="Metallic", default=0.0, min=0.0, max=1.0, update=_update_live_material,
    )
    show_implicit_hydrogens: BoolProperty(name="Show Implicit Hydrogens", default=True)
    show_uff_energy_breakdown: BoolProperty(
        name="Show Energy Breakdown", default=False,
        description="Show the per-term UFF energy breakdown below the total",
    )
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


def _serialize_molecule(molecule: Molecule) -> str:
    return json.dumps({
        "name": molecule.name,
        "atoms": [
            {"element": atom.element, "position": atom.position, "formal_charge": atom.formal_charge,
             "chirality": atom.chirality}
            for atom in molecule.atoms
        ],
        "bonds": [
            {"atom_a": bond.atom_a, "atom_b": bond.atom_b, "order": bond.order, "stereo": bond.stereo}
            for bond in molecule.bonds
        ],
    }, separators=(",", ":"))


def _molecule_revision(root: bpy.types.Object) -> int:
    return int(root.get("blend_mol_revision", 1))


def _store_topology_change(root: bpy.types.Object, molecule: Molecule) -> None:
    """Store a structural edit and invalidate calculations using older topology."""
    root["blend_mol_data"] = _serialize_molecule(molecule)
    root["blend_mol_revision"] = _molecule_revision(root) + 1


def _topology_signature(molecule: Molecule) -> tuple:
    """Undo-safe identity for all data that changes UFF terms or parameters."""
    return (
        tuple((atom.element, atom.formal_charge, atom.chirality) for atom in molecule.atoms),
        tuple((bond.atom_a, bond.atom_b, bond.order, bond.stereo) for bond in molecule.bonds),
    )


def _store_live_uff_session(
    root: bpy.types.Object, initial_positions: str, topology: tuple,
    held_atom: int, applied_updates: int,
) -> bool:
    """Save a completed interactive session when its topology is still valid."""
    final_molecule = _deserialize_molecule(root["blend_mol_data"])
    if _topology_signature(final_molecule) != topology:
        return False
    root["blend_mol_uff_before_positions"] = initial_positions
    root["blend_mol_uff_after_positions"] = _serialize_positions(final_molecule)
    root["blend_mol_live_uff_session"] = json.dumps({
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "held_atom": held_atom,
        "applied_updates": applied_updates,
        "final_phase": "stopped",
    }, separators=(",", ":"))
    return True


def _deserialize_molecule(payload: str) -> Molecule:
    data = json.loads(payload)
    return Molecule(
        name=data["name"],
        atoms=tuple(
            Atom(atom["element"], tuple(atom["position"]), atom.get("formal_charge", 0), atom.get("chirality"))
            for atom in data["atoms"]
        ),
        bonds=tuple(
            Bond(bond["atom_a"], bond["atom_b"], bond["order"], bond.get("stereo"))
            for bond in data["bonds"]
        ),
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


def _activate_explicit_atom(context, root: bpy.types.Object, atom_index: int) -> None:
    """Make an atom the active selection after creating or rebuilding a molecule."""
    collection = _molecule_collection(root)
    if collection is None:
        return
    atom = next(
        (obj for obj in collection.objects if obj.get("blend_mol_atom_index") == atom_index),
        None,
    )
    bpy.ops.object.select_all(action="DESELECT")
    if atom is not None:
        atom.select_set(True)
        context.view_layer.objects.active = atom
    else:
        root.select_set(True)
        context.view_layer.objects.active = root


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


def _selected_explicit_atoms_any(context) -> tuple[bpy.types.Object, bpy.types.Collection, tuple[int, ...]] | None:
    """Like _selected_explicit_atoms, but any non-zero count of atoms, for
    operations (e.g. delete) that apply the same way to one atom or many.
    """
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
    if not indices:
        return None
    return root, collection, indices


def _selected_angle_atoms(
    context,
) -> tuple[bpy.types.Object, bpy.types.Collection, tuple[int, int, int]] | None:
    """Return outer-center-outer indices, using the active atom as center."""
    active = _selected_explicit_atom(context)
    if active is None:
        return None
    root, collection, center = active
    indices = sorted({
        int(obj["blend_mol_atom_index"])
        for obj in context.selected_objects
        if obj.parent == root
        and obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and "blend_mol_atom_index" in obj
    })
    if len(indices) != 3 or center not in indices:
        return None
    outer = [index for index in indices if index != center]
    return root, collection, (outer[0], center, outer[1])


def _selected_dihedral_atoms(
    context,
) -> tuple[bpy.types.Object, bpy.types.Collection, tuple[int, int, int, int]] | None:
    """Infer one unambiguous bonded four-atom path from the selection."""
    root = _selected_molecule_root(context)
    collection = _molecule_collection(root) if root else None
    if root is None or collection is None:
        return None
    indices = sorted({
        int(obj["blend_mol_atom_index"])
        for obj in context.selected_objects
        if obj.parent == root
        and obj.get("blend_mol_kind") == "atom"
        and not obj.get("blend_mol_implicit")
        and "blend_mol_atom_index" in obj
    })
    if len(indices) != 4:
        return None
    molecule = _deserialize_molecule(root["blend_mol_data"])
    bonds = {frozenset((bond.atom_a, bond.atom_b)) for bond in molecule.bonds}
    paths = set()
    for candidate in permutations(indices):
        if all(frozenset(candidate[offset:offset + 2]) in bonds for offset in range(3)):
            reverse = tuple(reversed(candidate))
            paths.add(min(candidate, reverse))
    if len(paths) != 1:
        return None
    return root, collection, next(iter(paths))


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
        Atom(atom.element, positions[index], atom.formal_charge, atom.chirality)
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
        valence = _max_legitimate_valence(atom)
        if valence is None:
            messages.append(f"Atom {index + 1} ({atom.element}) has no valence rule")
        elif bond_order_sum[index] > valence:
            messages.append(
                f"Atom {index + 1} ({atom.element}) has bond order {bond_order_sum[index]} > {valence}"
            )
    return messages


def _bond_edit_valence_error(original: Molecule, updated: Molecule, indices) -> str | None:
    """Reject edits that introduce or worsen a known valence excess.

    Allow an existing invalid structure to be repaired one bond order at a time.
    These are the same conservative, charge-aware rules used by the panel.
    """
    def bond_order_sum(molecule, index):
        return sum(bond.order for bond in molecule.bonds
                   if index in (bond.atom_a, bond.atom_b))

    for index in indices:
        atom = updated.atoms[index]
        target = _max_legitimate_valence(atom)
        total = bond_order_sum(updated, index)
        previous = bond_order_sum(original, index) if index < len(original.atoms) else 0
        if target is not None and total > target and total > previous:
            return (
                f"Cannot add this bond: atom {index + 1} ({atom.element}) would have "
                f"bond order {total}, above its valence {target}. "
                "Remove an existing bond or explicit hydrogen first. "
                "To replace a hydrogen, select it and use Change Selected Atom Element."
            )
    return None


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
        Atom(atom.element, tuple(positions[index]), atom.formal_charge, atom.chirality)
        for index, atom in enumerate(molecule.atoms)
    )
    return Molecule(molecule.name, atoms, molecule.bonds)


def _store_position_snapshot(root: bpy.types.Object, molecule: Molecule) -> None:
    root["blend_mol_position_snapshot"] = _serialize_positions(molecule)


def _serialize_positions(molecule: Molecule) -> str:
    return json.dumps([atom.position for atom in molecule.atoms], separators=(",", ":"))


def _apply_serialized_positions(molecule: Molecule, serialized: str) -> Molecule:
    positions = json.loads(serialized)
    if len(positions) != len(molecule.atoms):
        raise ValueError("The saved coordinates belong to a different atom count")
    atoms = tuple(
        Atom(atom.element, tuple(positions[index]), atom.formal_charge, atom.chirality)
        for index, atom in enumerate(molecule.atoms)
    )
    return Molecule(molecule.name, atoms, molecule.bonds)


def _fixed_atom_indices(root: bpy.types.Object) -> set[int]:
    try:
        return set(json.loads(root.get("blend_mol_fixed_atoms", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def _store_fixed_atom_indices(root: bpy.types.Object, indices: set[int]) -> None:
    root["blend_mol_fixed_atoms"] = json.dumps(sorted(indices))


def _distance_constraints(root: bpy.types.Object) -> tuple[tuple[int, int, float, float], ...]:
    data = json.loads(root.get("blend_mol_distance_constraints", "[]"))
    constraints = []
    for item in data:
        first, second = sorted((int(item["atom_a"]), int(item["atom_b"])))
        constraints.append((first, second, float(item["target"]), float(item["strength"])))
    return tuple(constraints)


def _store_distance_constraints(
    root: bpy.types.Object, constraints: tuple[tuple[int, int, float, float], ...]
) -> None:
    root["blend_mol_distance_constraints"] = json.dumps([
        {"atom_a": first, "atom_b": second, "target": target, "strength": strength}
        for first, second, target, strength in constraints
    ], separators=(",", ":"))
    root["blend_mol_revision"] = _molecule_revision(root) + 1


def _angle_constraints(root: bpy.types.Object) -> tuple[tuple[int, int, int, float, float], ...]:
    data = json.loads(root.get("blend_mol_angle_constraints", "[]"))
    constraints = []
    for item in data:
        first, second = sorted((int(item["outer_a"]), int(item["outer_b"])))
        constraints.append((
            first, int(item["center"]), second,
            math.radians(float(item["target_degrees"])), float(item["strength"]),
        ))
    return tuple(constraints)


def _store_angle_constraints(
    root: bpy.types.Object, constraints: tuple[tuple[int, int, int, float, float], ...]
) -> None:
    root["blend_mol_angle_constraints"] = json.dumps([
        {"outer_a": first, "center": center, "outer_b": second,
         "target_degrees": math.degrees(target), "strength": strength}
        for first, center, second, target, strength in constraints
    ], separators=(",", ":"))
    root["blend_mol_revision"] = _molecule_revision(root) + 1


def _dihedral_constraints(
    root: bpy.types.Object,
) -> tuple[tuple[int, int, int, int, float, float], ...]:
    data = json.loads(root.get("blend_mol_dihedral_constraints", "[]"))
    return tuple((
        int(item["atom_a"]), int(item["atom_b"]), int(item["atom_c"]), int(item["atom_d"]),
        math.radians(float(item["target_degrees"])), float(item["strength"]),
    ) for item in data)


def _store_dihedral_constraints(
    root: bpy.types.Object,
    constraints: tuple[tuple[int, int, int, int, float, float], ...],
) -> None:
    root["blend_mol_dihedral_constraints"] = json.dumps([
        {"atom_a": first, "atom_b": center_a, "atom_c": center_b, "atom_d": fourth,
         "target_degrees": math.degrees(target), "strength": strength}
        for first, center_a, center_b, fourth, target, strength in constraints
    ], separators=(",", ":"))
    root["blend_mol_revision"] = _molecule_revision(root) + 1


def _restore_position_snapshot(root: bpy.types.Object, molecule: Molecule) -> Molecule:
    return _apply_serialized_positions(molecule, root["blend_mol_position_snapshot"])


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
    bl_label = "Make Implicit Hydrogens Explicit"
    bl_description = "Convert displayed implicit hydrogens into editable atoms and bonds"
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
        atoms[atom_index] = Atom(atom.element, atom.position, charge, atom.chirality)
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


class BLENDMOL_OT_jitter_atom_positions(Operator):
    bl_idname = "blendmol.jitter_atom_positions"
    bl_label = "Jitter All Atom Positions"
    bl_description = (
        "Nudge every non-fixed atom in the selected molecule by a small random "
        "amount; an emergency escape from an exactly-degenerate starting geometry "
        "(e.g. a collinear-angle UFF error) that keeps happening even after moving "
        "an atom by hand"
    )
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
        settings = context.scene.blend_mol_settings
        magnitude = settings.jitter_magnitude
        fixed_atoms = _fixed_atom_indices(root)
        _store_position_snapshot(root, molecule)
        jittered_count = 0
        atoms = []
        for index, atom in enumerate(molecule.atoms):
            if index in fixed_atoms:
                atoms.append(atom)
                continue
            jittered_count += 1
            atoms.append(Atom(
                atom.element,
                tuple(
                    coordinate + random.uniform(-magnitude, magnitude)
                    for coordinate in atom.position
                ),
                atom.formal_charge,
                atom.chirality,
            ))
        updated = Molecule(molecule.name, tuple(atoms), molecule.bonds)
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, settings)
        self.report(
            {"INFO"},
            f"Jittered {jittered_count} atom(s) by up to ±{magnitude:.3f} Å "
            f"(Restore Position Snapshot to undo)",
        )
        return {"FINISHED"}


class BLENDMOL_OT_minimize_partial_uff(Operator):
    bl_idname = "blendmol.minimize_partial_uff"
    bl_label = "Minimize Partial UFF"
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
        settings = context.scene.blend_mol_settings
        molecule = _ensure_hydrogens_explicit_for_uff(root, collection, molecule, settings)
        try:
            result = minimize_partial_energy(
                molecule, settings.uff_iterations, settings.uff_max_step,
                _fixed_atom_indices(root), _distance_constraints(root), _angle_constraints(root),
                _dihedral_constraints(root),
                gradient_tolerance=settings.uff_gradient_tolerance,
                energy_tolerance=settings.uff_energy_tolerance,
                algorithm=settings.uff_optimizer,
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, f"UFF minimization stopped; original coordinates kept: {exc}")
            return {"CANCELLED"}
        if not result.ready:
            message = result.messages[0] if result.messages else "UFF minimization failed"
            self.report({"ERROR"}, f"{message}; original coordinates kept")
            return {"CANCELLED"}
        _store_position_snapshot(root, molecule)
        root["blend_mol_uff_before_positions"] = _serialize_positions(molecule)
        root["blend_mol_uff_after_positions"] = _serialize_positions(result.molecule)
        root["blend_mol_data"] = _serialize_molecule(result.molecule)
        root["blend_mol_uff_last_run"] = json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "initial_energy_kcal_mol": result.initial_energy_kcal_mol,
            "final_energy_kcal_mol": result.final_energy_kcal_mol,
            "iterations": result.iterations,
            "termination": result.termination,
            "fixed_atoms": len(_fixed_atom_indices(root)),
            "distance_constraints": len(_distance_constraints(root)),
            "angle_constraints": len(_angle_constraints(root)),
            "dihedral_constraints": len(_dihedral_constraints(root)),
            "gradient_tolerance": settings.uff_gradient_tolerance,
            "energy_tolerance": settings.uff_energy_tolerance,
            "optimizer": settings.uff_optimizer,
        }, separators=(",", ":"))
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, result.molecule, settings)
        self.report(
            {"INFO"},
            f"Partial UFF: {result.initial_energy_kcal_mol:.3f} → {result.final_energy_kcal_mol:.3f} kcal/mol"
        )
        return {"FINISHED"}


class BLENDMOL_OT_relax_around_selected_atom(Operator):
    bl_idname = "blendmol.relax_around_selected_atom"
    bl_label = "Relax Around Selected Atom"
    bl_description = "Keep the selected moved atom fixed and run a few partial UFF steps on its surroundings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select one moved, non-implicit atom first")
            return {"CANCELLED"}
        root, collection, atom_index = selected
        try:
            molecule = _sync_atom_positions(root, collection)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot synchronize moved atom: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        temporary_fixed = _fixed_atom_indices(root) | {atom_index}
        started = perf_counter()
        try:
            result = minimize_partial_energy(
                molecule, settings.uff_live_steps, settings.uff_max_step,
                temporary_fixed, _distance_constraints(root), _angle_constraints(root),
                _dihedral_constraints(root),
                gradient_tolerance=settings.uff_gradient_tolerance,
                energy_tolerance=settings.uff_energy_tolerance,
                algorithm=settings.uff_optimizer,
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, f"Interactive UFF step stopped; moved coordinates kept: {exc}")
            return {"CANCELLED"}
        if not result.ready:
            message = result.messages[0] if result.messages else "UFF step failed"
            self.report({"ERROR"}, f"{message}; moved coordinates kept")
            return {"CANCELLED"}
        elapsed_ms = (perf_counter() - started) * 1000.0
        root["blend_mol_data"] = _serialize_molecule(result.molecule)
        root["blend_mol_uff_live_last"] = json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "initial_energy_kcal_mol": result.initial_energy_kcal_mol,
            "final_energy_kcal_mol": result.final_energy_kcal_mol,
            "steps": result.iterations,
            "held_atom": atom_index,
            "elapsed_ms": elapsed_ms,
            "gradient_tolerance": settings.uff_gradient_tolerance,
            "energy_tolerance": settings.uff_energy_tolerance,
            "optimizer": settings.uff_optimizer,
        }, separators=(",", ":"))
        display_started = perf_counter()
        if not _update_explicit_display_geometry(root, collection, result.molecule, settings):
            _clear_molecule_display(root, collection)
            _render_molecule(root, collection, result.molecule, settings)
            replacement = next(
                (obj for obj in collection.objects
                 if obj.get("blend_mol_kind") == "atom"
                 and obj.get("blend_mol_atom_index") == atom_index),
                None,
            )
            if replacement is not None:
                replacement.select_set(True)
                context.view_layer.objects.active = replacement
        display_elapsed_ms = (perf_counter() - display_started) * 1000.0
        live_log = json.loads(root["blend_mol_uff_live_last"])
        live_log["display_update_ms"] = display_elapsed_ms
        live_log["total_step_ms"] = elapsed_ms + display_elapsed_ms
        root["blend_mol_uff_live_last"] = json.dumps(live_log, separators=(",", ":"))
        self.report(
            {"INFO"},
            f"Held atom {atom_index + 1}; partial UFF {result.initial_energy_kcal_mol:.3f} → "
            f"{result.final_energy_kcal_mol:.3f} kcal/mol",
        )
        return {"FINISHED"}


class BLENDMOL_OT_benchmark_uff_step(Operator):
    bl_idname = "blendmol.benchmark_uff_step"
    bl_label = "Benchmark 1 UFF Step"
    bl_description = "Measure one partial UFF step without changing the molecule coordinates"
    bl_options = {"REGISTER"}

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            self.report({"WARNING"}, "Select a Blend Mol atom, bond, or molecule root first")
            return {"CANCELLED"}
        try:
            molecule = _sync_atom_positions(root, collection)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot prepare UFF benchmark: {exc}")
            return {"CANCELLED"}
        fixed_atoms = _fixed_atom_indices(root)
        started = perf_counter()
        try:
            result = minimize_partial_energy(
                molecule, 1, context.scene.blend_mol_settings.uff_max_step,
                fixed_atoms, _distance_constraints(root), _angle_constraints(root),
                _dihedral_constraints(root),
                gradient_tolerance=context.scene.blend_mol_settings.uff_gradient_tolerance,
                energy_tolerance=context.scene.blend_mol_settings.uff_energy_tolerance,
                algorithm=context.scene.blend_mol_settings.uff_optimizer,
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, f"UFF benchmark failed: {exc}")
            return {"CANCELLED"}
        elapsed_ms = (perf_counter() - started) * 1000.0
        if not result.ready:
            message = result.messages[0] if result.messages else "UFF benchmark failed"
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        moving_atoms = len(molecule.atoms) - len(fixed_atoms)
        numeric_atoms = len(numeric_gradient_atom_indices(molecule) - fixed_atoms)
        compute_backend = backend_status()
        root["blend_mol_uff_benchmark"] = json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "atom_count": len(molecule.atoms),
            "moving_atoms": moving_atoms,
            "elapsed_ms": elapsed_ms,
            "estimated_hz": 1000.0 / elapsed_ms if elapsed_ms > 0.0 else None,
            "energy_evaluations": numeric_atoms * 3 + 1,
            "numeric_gradient_atoms": numeric_atoms,
            "gradient_method": (
                "analytic bond/angle/vdw + Rust parallel forward torsion/inversion"
                if compute_backend["active"] == "rust"
                else "analytic bond/angle/vdw + Python forward torsion/inversion"
            ),
            "compute_backend": compute_backend["active"],
            "native_available": compute_backend["native_available"],
            "gradient_tolerance": context.scene.blend_mol_settings.uff_gradient_tolerance,
            "energy_tolerance": context.scene.blend_mol_settings.uff_energy_tolerance,
            "optimizer": context.scene.blend_mol_settings.uff_optimizer,
        }, separators=(",", ":"))
        self.report({"INFO"}, f"One partial UFF step: {elapsed_ms:.1f} ms ({len(molecule.atoms)} atoms)")
        return {"FINISHED"}


_LIVE_UFF_OPERATOR = None
_AUTO_LIVE_UFF_TIMER_PENDING = False
_AUTO_LIVE_UFF_TARGET = None
# How long, after an auto-started Live UFF session ends in error (e.g. the
# collinear-angle error), to hold off auto-restarting it for the same atom.
# Without this, continuing to drag an atom through a degenerate geometry
# re-triggers the depsgraph auto-start handler on every further micro-
# movement, each attempt failing again -- a flood of repeated error reports
# that fights the user's own grab/cancel/confirm instead of just naming the
# problem once. See _AUTO_LIVE_UFF_ERROR_COOLDOWN below.
_AUTO_LIVE_UFF_ERROR_COOLDOWN_SECONDS = 1.5
_AUTO_LIVE_UFF_ERROR_COOLDOWN: dict[tuple[str, int], float] = {}


def _explicit_atom_has_moved(root, atom_object, atom_index):
    molecule = _deserialize_molecule(root["blend_mol_data"])
    position = atom_object.matrix_local.translation
    return any(abs(position[axis] - molecule.atoms[atom_index].position[axis]) > 1e-7
               for axis in range(3))


@persistent
def _invalidate_live_uff_before_history_change(*_args):
    """Discard pending and in-flight work before Undo, Redo, or file loading."""
    global _AUTO_LIVE_UFF_TIMER_PENDING, _AUTO_LIVE_UFF_TARGET
    if bpy.app.timers.is_registered(_start_auto_live_uff):
        bpy.app.timers.unregister(_start_auto_live_uff)
    _AUTO_LIVE_UFF_TIMER_PENDING = False
    _AUTO_LIVE_UFF_TARGET = None
    _AUTO_LIVE_UFF_ERROR_COOLDOWN.clear()
    if _LIVE_UFF_OPERATOR is not None:
        _LIVE_UFF_OPERATOR._finish(bpy.context, save_session=False)


def _start_auto_live_uff() -> None:
    """Start Live UFF outside the depsgraph callback after an atom transform."""
    global _AUTO_LIVE_UFF_TIMER_PENDING, _AUTO_LIVE_UFF_TARGET
    target = _AUTO_LIVE_UFF_TARGET
    _AUTO_LIVE_UFF_TARGET = None
    _AUTO_LIVE_UFF_TIMER_PENDING = False
    if _LIVE_UFF_OPERATOR is not None:
        return None
    context = bpy.context
    scene = getattr(context, "scene", None)
    if scene is None or not scene.blend_mol_settings.uff_live_auto_start:
        return None
    selected = _selected_explicit_atom(context)
    if selected is None:
        return None
    root, _collection, atom_index = selected
    if target != (root.name, context.active_object.name, atom_index):
        return None
    last_error_time = _AUTO_LIVE_UFF_ERROR_COOLDOWN.get((root.name, atom_index))
    if last_error_time is not None and perf_counter() - last_error_time < _AUTO_LIVE_UFF_ERROR_COOLDOWN_SECONDS:
        return None
    try:
        if not _explicit_atom_has_moved(root, context.active_object, atom_index):
            return None
        bpy.ops.blendmol.toggle_live_uff(
            "INVOKE_DEFAULT", start_immediately=True, auto_stop_when_ready=True,
        )
    except (RuntimeError, KeyError, IndexError, TypeError, ValueError, ReferenceError):
        # Blender can legitimately have no suitable window context while loading
        # a file or changing workspaces. The next transform queues a fresh start.
        pass
    return None


@persistent
def _queue_auto_live_uff_from_transform(scene, depsgraph) -> None:
    """Defer automatic Live UFF startup until Blender finishes a transform update."""
    global _AUTO_LIVE_UFF_TIMER_PENDING, _AUTO_LIVE_UFF_TARGET
    if _LIVE_UFF_OPERATOR is not None or _AUTO_LIVE_UFF_TIMER_PENDING:
        return
    settings = getattr(scene, "blend_mol_settings", None)
    if settings is None or not settings.uff_live_auto_start:
        return
    active = bpy.context.view_layer.objects.active if bpy.context.view_layer else None
    if active is None or active.get("blend_mol_kind") != "atom" or active.get("blend_mol_implicit"):
        return
    for update in depsgraph.updates:
        # ``update.id`` is the depsgraph's evaluated copy, which never equals
        # the original data-block even without modifiers; compare ``.original``.
        updated_id = getattr(update.id, "original", update.id)
        if updated_id == active and update.is_updated_transform:
            selected = _selected_explicit_atom(bpy.context)
            if selected is None:
                return
            root, _collection, atom_index = selected
            try:
                if not _explicit_atom_has_moved(root, active, atom_index):
                    return
            except (KeyError, IndexError, TypeError, ValueError, ReferenceError):
                return
            _AUTO_LIVE_UFF_TARGET = (root.name, active.name, atom_index)
            _AUTO_LIVE_UFF_TIMER_PENDING = True
            bpy.app.timers.register(_start_auto_live_uff, first_interval=0.0)
            return


class BLENDMOL_OT_toggle_live_uff(Operator):
    bl_idname = "blendmol.toggle_live_uff"
    bl_label = "Toggle Live UFF"
    bl_description = "Start or stop repeated partial UFF relaxation while the selected atom is moved"

    _timer = None
    _root_name = ""
    _atom_index = -1
    _last_position = None
    _stop_requested = False
    _worker = None
    _topology_revision = 0
    _topology_at_start = None
    _prepared_uff = None
    _initial_positions = None
    _initial_live_log = None
    _settle_updates = 0
    _applied_results = 0
    _cancel_event = None
    _adaptive_steps = 1
    _optimizer_state = None

    start_immediately: BoolProperty(options={"HIDDEN"}, default=False)
    auto_stop_when_ready: BoolProperty(options={"HIDDEN"}, default=False)

    def _set_live_error(self, root: bpy.types.Object, message: str) -> None:
        """Record a Live UFF failure on root AND surface it to the user.

        Storing the message on root alone (the previous behaviour) left
        Live UFF failing silently: nothing in the Live UFF panel ever read
        that property, and this modal operator's own error paths never
        called self.report, so a failure (e.g. the collinear-angle error)
        just stopped the session with no visible sign anything went wrong.
        """
        root["blend_mol_live_uff_error"] = message
        self.report({"ERROR"}, message)
        if self._root_name and self._atom_index >= 0:
            _AUTO_LIVE_UFF_ERROR_COOLDOWN[(self._root_name, self._atom_index)] = perf_counter()

    def _cancel_and_restore(self, context):
        """Invalidate work first, then restore this session's coordinate snapshot."""
        self._finish(context, save_session=False)
        root = bpy.data.objects.get(self._root_name)
        if root is None or self._initial_positions is None:
            return
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
            if (_molecule_revision(root) != self._topology_revision
                    or _topology_signature(molecule) != self._topology_at_start):
                self._set_live_error(root, "Cannot restore Live UFF: molecule structure changed")
                return
            collection = _molecule_collection(root)
            if collection is None:
                return
            restored = _apply_serialized_positions(molecule, self._initial_positions)
            # Save before display changes: transform notifications must see the
            # restored coordinates, rather than queueing a fresh auto session.
            root["blend_mol_data"] = _serialize_molecule(restored)
            if not _update_explicit_display_geometry(
                root, collection, restored, context.scene.blend_mol_settings,
            ):
                self._set_live_error(root, "Cannot restore all Live UFF display objects")
                return
            if self._initial_live_log is None:
                if "blend_mol_uff_live_last" in root:
                    del root["blend_mol_uff_live_last"]
            else:
                root["blend_mol_uff_live_last"] = self._initial_live_log
            root["blend_mol_live_uff_phase"] = "cancelled"
        except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._set_live_error(root, f"Live UFF restore failed: {exc}")

    def _submit(self, molecule, root, settings, phase="calculating"):
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = Event()
        root["blend_mol_live_uff_phase"] = phase
        return self._worker.submit((
            molecule,
            self._adaptive_steps,
            settings.uff_max_step,
            frozenset(_fixed_atom_indices(root) | {self._atom_index}),
            _distance_constraints(root),
            _angle_constraints(root),
            _dihedral_constraints(root),
            self._prepared_uff,
            settings.uff_gradient_tolerance,
            self._cancel_event,
            settings.uff_energy_tolerance,
            settings.uff_optimizer,
            self._optimizer_state,
        ))

    def _finish(self, context, save_session=True):
        global _LIVE_UFF_OPERATOR
        if self._stop_requested and self._timer is None and self._worker is None:
            return {"CANCELLED"}
        self._stop_requested = True
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._cancel_event = None
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._prepared_uff = None
        self._optimizer_state = None
        root = bpy.data.objects.get(self._root_name)
        if root is not None:
            root["blend_mol_live_uff_running"] = False
            root["blend_mol_live_uff_phase"] = "stopped"
            if save_session and self._applied_results and self._initial_positions is not None:
                try:
                    _store_live_uff_session(
                        root, self._initial_positions, self._topology_at_start,
                        self._atom_index, self._applied_results,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
        # A queued timer may finish this operator again. Never save a cancelled
        # session (or save a completed snapshot twice) on that second call.
        self._applied_results = 0
        if _LIVE_UFF_OPERATOR is self:
            _LIVE_UFF_OPERATOR = None
        return {"CANCELLED"}

    def invoke(self, context, event):
        global _LIVE_UFF_OPERATOR
        if _LIVE_UFF_OPERATOR is not None:
            return _LIVE_UFF_OPERATOR._finish(context)
        selected = _selected_explicit_atom(context)
        if selected is None:
            self.report({"WARNING"}, "Select one explicit atom to manipulate first")
            return {"CANCELLED"}
        root, collection, atom_index = selected
        self._root_name = root.name
        self._atom_index = atom_index
        self._last_position = (
            None if self.start_immediately
            else tuple(context.active_object.matrix_local.translation)
        )
        self._stop_requested = False
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Stored molecule data is invalid: {exc}")
            return {"CANCELLED"}
        molecule = _ensure_hydrogens_explicit_for_uff(
            root, collection, molecule, context.scene.blend_mol_settings,
        )
        self._topology_revision = _molecule_revision(root)
        self._topology_at_start = _topology_signature(molecule)
        self._prepared_uff = prepare_uff_context(molecule)
        self._initial_positions = _serialize_positions(molecule)
        self._initial_live_log = root.get("blend_mol_uff_live_last")
        self._settle_updates = 0
        self._applied_results = 0
        self._cancel_event = None
        self._adaptive_steps = context.scene.blend_mol_settings.uff_live_steps
        self._optimizer_state = None
        if "blend_mol_live_uff_error" in root:
            del root["blend_mol_live_uff_error"]
        self._worker = LatestOnlyWorker(
            lambda job: minimize_partial_energy(
                job[0], job[1], job[2], job[3], job[4], job[5], job[6], job[7], job[8],
                job[9].is_set, job[10], job[11], job[12],
            )
        )
        interval = context.scene.blend_mol_settings.uff_live_interval
        self._timer = context.window_manager.event_timer_add(interval, window=context.window)
        context.window_manager.modal_handler_add(self)
        root["blend_mol_live_uff_running"] = True
        root["blend_mol_live_uff_phase"] = "ready"
        _LIVE_UFF_OPERATOR = self
        self.report({"INFO"}, f"Live UFF started with atom {atom_index + 1} held")
        return {"RUNNING_MODAL"}

    def execute(self, context):
        return self.invoke(context, None)

    def modal(self, context, event):
        if self._stop_requested:
            return self._finish(context)
        if event.type == "ESC" or (event.type == "RIGHTMOUSE" and event.value == "PRESS"):
            self._cancel_and_restore(context)
            # Let Blender's active transform also receive its cancel event.
            return {"CANCELLED", "PASS_THROUGH"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._worker is None:
            return self._finish(context)
        root = bpy.data.objects.get(self._root_name)
        collection = _molecule_collection(root) if root else None
        if root is None or collection is None:
            return self._finish(context)
        try:
            stored_molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._set_live_error(root, f"Stored molecule changed: {exc}")
            return self._finish(context)
        if (
            _molecule_revision(root) != self._topology_revision
            or _topology_signature(stored_molecule) != self._topology_at_start
        ):
            self._set_live_error(root, "Live UFF stopped because the molecule structure changed")
            return self._finish(context)
        atom_object = next(
            (obj for obj in collection.objects
             if obj.get("blend_mol_kind") == "atom"
             and not obj.get("blend_mol_implicit")
             and obj.get("blend_mol_atom_index") == self._atom_index),
            None,
        )
        if atom_object is None:
            return self._finish(context)
        current_position = tuple(atom_object.matrix_local.translation)
        moved = self._last_position is None or not all(
            abs(current_position[axis] - self._last_position[axis]) < 1e-7 for axis in range(3)
        )
        settings = context.scene.blend_mol_settings
        try:
            if moved:
                self._last_position = current_position
                molecule = _sync_atom_positions(root, collection)
                self._settle_updates = 0
                self._optimizer_state = None
                self._submit(molecule, root, settings)
            completion = self._worker.poll_latest()
            if completion is None:
                return {"PASS_THROUGH"}
            if completion.error is not None:
                raise completion.error
            result = completion.value
            self._optimizer_state = result.optimizer_state
            compute_ms = completion.elapsed_ms
            completed_chunk_steps = self._adaptive_steps
            self._adaptive_steps = adapt_chunk_iterations(
                self._adaptive_steps, compute_ms, settings.uff_live_target_ms,
            )
            if not result.ready:
                self._set_live_error(root, result.messages[0] if result.messages else "UFF step failed")
                return self._finish(context)
            root["blend_mol_data"] = _serialize_molecule(result.molecule)
            self._applied_results += 1
            display_started = perf_counter()
            if not _update_explicit_display_geometry(root, collection, result.molecule, settings):
                self._set_live_error(root, "Live display requires a fully explicit current mesh")
                return self._finish(context)
            display_ms = (perf_counter() - display_started) * 1000.0
            root["blend_mol_uff_live_last"] = json.dumps({
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "initial_energy_kcal_mol": result.initial_energy_kcal_mol,
                "final_energy_kcal_mol": result.final_energy_kcal_mol,
                "steps": result.iterations,
                "held_atom": self._atom_index,
                "elapsed_ms": compute_ms,
                "display_update_ms": display_ms,
                "total_step_ms": compute_ms + display_ms,
                "generation": completion.generation,
                "background": True,
                "settle_update": self._settle_updates,
                "gradient_tolerance": settings.uff_gradient_tolerance,
                "energy_tolerance": settings.uff_energy_tolerance,
                "optimizer": settings.uff_optimizer,
                "requested_steps": completed_chunk_steps,
                "next_adaptive_steps": self._adaptive_steps,
                "compute_budget_ms": settings.uff_live_target_ms,
            }, separators=(",", ":"))
            continue_settling = should_continue_relaxation(
                result.iterations, result.termination,
                self._settle_updates, settings.uff_live_settle_updates,
            )
            if not continue_settling and self._settle_updates < settings.uff_live_settle_updates:
                # A string-based termination reason ("energy tolerance", "line
                # search stalled") can be satisfied well before the gradient
                # itself is anywhere near uff_gradient_tolerance -- confirmed
                # by headless measurement after the 0.1.67 fix. Only the
                # gradient is authoritative, so check it directly before
                # trusting "ready" while settle budget remains.
                residual_fixed = frozenset(_fixed_atom_indices(root) | {self._atom_index})
                _residual_objective, residual_gradient, residual_messages = (
                    partial_objective_energy_gradient(
                        result.molecule, self._prepared_uff, residual_fixed,
                        _distance_constraints(root), _angle_constraints(root),
                        _dihedral_constraints(root), 1e-4,
                    )
                )
                if not residual_messages:
                    residual_max_component, _residual_max_norm = _gradient_step_metrics(
                        residual_gradient, residual_fixed
                    )
                    if residual_max_component >= settings.uff_gradient_tolerance:
                        continue_settling = True
            if continue_settling:
                self._settle_updates += 1
                self._submit(result.molecule, root, settings, "settling")
            else:
                root["blend_mol_live_uff_phase"] = "ready"
                if self.auto_stop_when_ready:
                    return self._finish(context)
        except (ArithmeticError, KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._set_live_error(root, str(exc))
            return self._finish(context)
        return {"PASS_THROUGH"}


class BLENDMOL_OT_restore_uff_coordinates(Operator):
    bl_idname = "blendmol.restore_uff_coordinates"
    bl_label = "Restore UFF Coordinates"
    bl_description = "Restore coordinates from immediately before or after the last UFF minimization"
    bl_options = {"REGISTER", "UNDO"}

    state: EnumProperty(
        name="State",
        items=(("BEFORE", "Before", "Before the last minimization"),
               ("AFTER", "After", "Result of the last minimization")),
        default="BEFORE",
    )

    def execute(self, context):
        root = _selected_molecule_root(context)
        collection = _molecule_collection(root) if root else None
        property_name = (
            "blend_mol_uff_before_positions" if self.state == "BEFORE"
            else "blend_mol_uff_after_positions"
        )
        if root is None or collection is None or property_name not in root:
            self.report({"WARNING"}, "Select a molecule with a saved UFF comparison first")
            return {"CANCELLED"}
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
            updated = _apply_serialized_positions(molecule, root[property_name])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot restore UFF coordinates: {exc}")
            return {"CANCELLED"}
        root["blend_mol_data"] = _serialize_molecule(updated)
        _clear_molecule_display(root, collection)
        _render_molecule(root, collection, updated, context.scene.blend_mol_settings)
        label = "before" if self.state == "BEFORE" else "after"
        self.report({"INFO"}, f"Restored coordinates from {label} the last UFF minimization")
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


class BLENDMOL_OT_set_distance_constraint(Operator):
    bl_idname = "blendmol.set_distance_constraint"
    bl_label = "Set Selected Distance Constraint"
    bl_description = "Add or update a harmonic distance restraint between two selected atoms"
    bl_options = {"REGISTER", "UNDO"}

    use_current: BoolProperty(name="Use Current Distance", default=False)

    def execute(self, context):
        selected = _selected_explicit_atoms(context)
        if selected is None:
            self.report({"WARNING"}, "Select exactly two non-implicit atoms in one molecule")
            return {"CANCELLED"}
        root, collection, (first, second) = selected
        try:
            molecule = _sync_atom_positions(root, collection)
            constraints = list(_distance_constraints(root))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot create distance constraint: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        if self.use_current:
            delta = tuple(
                molecule.atoms[first].position[axis] - molecule.atoms[second].position[axis]
                for axis in range(3)
            )
            target = math.sqrt(sum(component * component for component in delta))
            settings.distance_constraint_target = target
        else:
            target = settings.distance_constraint_target
        term = (first, second, target, settings.distance_constraint_strength)
        constraints = [item for item in constraints if item[:2] != term[:2]]
        constraints.append(term)
        _store_distance_constraints(root, tuple(sorted(constraints)))
        self.report({"INFO"}, f"Constrained atoms {first + 1}-{second + 1} to {target:.3f} Å")
        return {"FINISHED"}


class BLENDMOL_OT_remove_distance_constraint(Operator):
    bl_idname = "blendmol.remove_distance_constraint"
    bl_label = "Remove Distance Constraint"
    bl_options = {"REGISTER", "UNDO"}

    remove_all: BoolProperty(name="Remove All", default=False)

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol molecule first")
            return {"CANCELLED"}
        try:
            constraints = list(_distance_constraints(root))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot read distance constraints: {exc}")
            return {"CANCELLED"}
        if self.remove_all:
            removed = len(constraints)
            constraints = []
        else:
            selected = _selected_explicit_atoms(context)
            if selected is None or selected[0] != root:
                self.report({"WARNING"}, "Select the two constrained atoms first")
                return {"CANCELLED"}
            pair = selected[2]
            remaining = [item for item in constraints if item[:2] != pair]
            removed = len(constraints) - len(remaining)
            constraints = remaining
        if removed == 0:
            self.report({"INFO"}, "No matching distance constraint")
            return {"CANCELLED"}
        _store_distance_constraints(root, tuple(constraints))
        self.report({"INFO"}, f"Removed {removed} distance constraint(s)")
        return {"FINISHED"}


class BLENDMOL_OT_set_angle_constraint(Operator):
    bl_idname = "blendmol.set_angle_constraint"
    bl_label = "Set Selected Angle Constraint"
    bl_description = "Use the active atom as center of a three-atom angle restraint"
    bl_options = {"REGISTER", "UNDO"}

    use_current: BoolProperty(name="Use Current Angle", default=False)

    def execute(self, context):
        selected = _selected_angle_atoms(context)
        if selected is None:
            self.report({"WARNING"}, "Select three atoms; make the center atom active")
            return {"CANCELLED"}
        root, collection, (first, center, second) = selected
        try:
            molecule = _sync_atom_positions(root, collection)
            constraints = list(_angle_constraints(root))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot create angle constraint: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        if self.use_current:
            first_vector = Vector(molecule.atoms[first].position) - Vector(molecule.atoms[center].position)
            second_vector = Vector(molecule.atoms[second].position) - Vector(molecule.atoms[center].position)
            if min(first_vector.length, second_vector.length) < 1e-10:
                self.report({"ERROR"}, "Angle atoms cannot occupy the same position")
                return {"CANCELLED"}
            target = first_vector.angle(second_vector)
            settings.angle_constraint_target = math.degrees(target)
        else:
            target = math.radians(settings.angle_constraint_target)
        key = (first, center, second)
        term = key + (target, settings.angle_constraint_strength)
        constraints = [item for item in constraints if item[:3] != key]
        constraints.append(term)
        _store_angle_constraints(root, tuple(sorted(constraints)))
        self.report(
            {"INFO"},
            f"Constrained angle {first + 1}-{center + 1}-{second + 1} to {math.degrees(target):.2f}°",
        )
        return {"FINISHED"}


class BLENDMOL_OT_remove_angle_constraint(Operator):
    bl_idname = "blendmol.remove_angle_constraint"
    bl_label = "Remove Angle Constraint"
    bl_options = {"REGISTER", "UNDO"}

    remove_all: BoolProperty(name="Remove All", default=False)

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol molecule first")
            return {"CANCELLED"}
        try:
            constraints = list(_angle_constraints(root))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot read angle constraints: {exc}")
            return {"CANCELLED"}
        if self.remove_all:
            removed = len(constraints)
            constraints = []
        else:
            selected = _selected_angle_atoms(context)
            if selected is None or selected[0] != root:
                self.report({"WARNING"}, "Select three atoms; make the center atom active")
                return {"CANCELLED"}
            key = selected[2]
            remaining = [item for item in constraints if item[:3] != key]
            removed = len(constraints) - len(remaining)
            constraints = remaining
        if removed == 0:
            self.report({"INFO"}, "No matching angle constraint")
            return {"CANCELLED"}
        _store_angle_constraints(root, tuple(constraints))
        self.report({"INFO"}, f"Removed {removed} angle constraint(s)")
        return {"FINISHED"}


class BLENDMOL_OT_set_dihedral_constraint(Operator):
    bl_idname = "blendmol.set_dihedral_constraint"
    bl_label = "Set Selected Dihedral Constraint"
    bl_description = "Infer a bonded four-atom path and add or update its dihedral restraint"
    bl_options = {"REGISTER", "UNDO"}

    use_current: BoolProperty(name="Use Current Dihedral", default=False)

    def execute(self, context):
        selected = _selected_dihedral_atoms(context)
        if selected is None:
            self.report({"WARNING"}, "Select four atoms forming one unambiguous bonded path")
            return {"CANCELLED"}
        root, collection, path = selected
        try:
            molecule = _sync_atom_positions(root, collection)
            constraints = list(_dihedral_constraints(root))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot create dihedral constraint: {exc}")
            return {"CANCELLED"}
        settings = context.scene.blend_mol_settings
        if self.use_current:
            target = _dihedral_angle(*(molecule.atoms[index].position for index in path))
            if target is None:
                self.report({"ERROR"}, "Selected path has a degenerate dihedral")
                return {"CANCELLED"}
            settings.dihedral_constraint_target = math.degrees(target)
        else:
            target = math.radians(settings.dihedral_constraint_target)
        term = path + (target, settings.dihedral_constraint_strength)
        constraints = [item for item in constraints if item[:4] != path]
        constraints.append(term)
        _store_dihedral_constraints(root, tuple(sorted(constraints)))
        self.report(
            {"INFO"},
            f"Constrained dihedral {'-'.join(str(index + 1) for index in path)} "
            f"to {math.degrees(target):.2f}°",
        )
        return {"FINISHED"}


class BLENDMOL_OT_remove_dihedral_constraint(Operator):
    bl_idname = "blendmol.remove_dihedral_constraint"
    bl_label = "Remove Dihedral Constraint"
    bl_options = {"REGISTER", "UNDO"}

    remove_all: BoolProperty(name="Remove All", default=False)

    def execute(self, context):
        root = _selected_molecule_root(context)
        if root is None:
            self.report({"WARNING"}, "Select a Blend Mol molecule first")
            return {"CANCELLED"}
        constraints = list(_dihedral_constraints(root))
        if self.remove_all:
            removed = len(constraints)
            constraints = []
        else:
            selected = _selected_dihedral_atoms(context)
            if selected is None or selected[0] != root:
                self.report({"WARNING"}, "Select four atoms forming one unambiguous bonded path")
                return {"CANCELLED"}
            path = selected[2]
            remaining = [item for item in constraints if item[:4] != path]
            removed = len(constraints) - len(remaining)
            constraints = remaining
        if removed == 0:
            self.report({"INFO"}, "No matching dihedral constraint")
            return {"CANCELLED"}
        _store_dihedral_constraints(root, tuple(constraints))
        self.report({"INFO"}, f"Removed {removed} dihedral constraint(s)")
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
    """Root panel: the one always-visible entry point and current-selection status.

    Everything else lives in collapsible sub-panels below, grouped by
    workflow stage (file I/O, editing, constraints, UFF, display), so the
    N-panel isn't one long flat list of equally-weighted buttons. No
    operator or property was removed in this split -- every control from
    the previous single-panel layout is still reachable, just regrouped.
    """

    bl_label = "Blend Mol"
    bl_idname = "BLENDMOL_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        new_row = layout.row()
        new_row.scale_y = 1.5
        new_row.operator(BLENDMOL_OT_new_molecule.bl_idname, icon="FILE_NEW")
        root = _selected_molecule_root(context)
        if root:
            layout.label(text=f"Editing: {root.name}", icon="OBJECT_DATA")
        else:
            layout.label(text="Select an atom/bond/root, or create or import a molecule", icon="INFO")


class BLENDMOL_PT_file_io(Panel):
    bl_label = "Import / Export"
    bl_idname = "BLENDMOL_PT_file_io"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        layout.label(text="Import")
        import_row = layout.row(align=True)
        import_row.operator(BLENDMOL_OT_import_v2000.bl_idname, text="MOL", icon="IMPORT")
        import_row.operator(BLENDMOL_OT_import_xyz.bl_idname, text="XYZ", icon="IMPORT")
        import_row.operator(BLENDMOL_OT_import_pdb.bl_idname, text="PDB", icon="IMPORT")
        import_row.operator(BLENDMOL_OT_import_sdf.bl_idname, text="SDF", icon="IMPORT")
        layout.operator(BLENDMOL_OT_import_smiles.bl_idname, icon="MODIFIER")
        layout.separator()
        layout.label(text="Export selected molecule")
        export_row = layout.row(align=True)
        export_row.operator(BLENDMOL_OT_export_v2000.bl_idname, text="MOL", icon="EXPORT")
        export_row.operator(BLENDMOL_OT_export_xyz.bl_idname, text="XYZ", icon="EXPORT")
        export_row.operator(BLENDMOL_OT_export_pdb.bl_idname, text="PDB", icon="EXPORT")
        export_row.operator(BLENDMOL_OT_export_sdf.bl_idname, text="SDF", icon="EXPORT")
        layout.operator(BLENDMOL_OT_export_uff_report.bl_idname, icon="FILE_TEXT")


class BLENDMOL_PT_edit(Panel):
    bl_label = "Edit Atoms & Bonds"
    bl_idname = "BLENDMOL_PT_edit"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_order = 2

    @classmethod
    def poll(cls, context):
        return _selected_molecule_root(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        root = _selected_molecule_root(context)
        layout.label(text="Move atom objects, then synchronize")
        layout.operator(BLENDMOL_OT_sync_atom_positions.bl_idname, icon="CON_TRANSFORM")
        layout.separator()
        layout.prop(settings, "edit_element")
        layout.prop(settings, "formal_charge")
        layout.operator(BLENDMOL_OT_add_atom.bl_idname, icon="ADD")
        layout.operator(BLENDMOL_OT_make_hydrogens_explicit.bl_idname, icon="MODIFIER")
        selected_atom = _selected_explicit_atom(context)
        if selected_atom:
            atom_box = layout.box()
            atom_box.prop(settings, "bond_order")
            atom_box.operator(BLENDMOL_OT_add_bonded_atom.bl_idname, icon="CON_TRACKTO")
            fixed_atoms = _fixed_atom_indices(root)
            state = "fixed" if selected_atom[2] in fixed_atoms else "movable"
            molecule = _deserialize_molecule(root["blend_mol_data"])
            charge = molecule.atoms[selected_atom[2]].formal_charge
            atom_box.label(text=f"Selected atom: {selected_atom[2] + 1} ({state}, {charge:+d})")
            atom_box.operator(BLENDMOL_OT_change_atom_element.bl_idname, icon="MODIFIER")
            atom_box.operator(BLENDMOL_OT_set_formal_charge.bl_idname, icon="ADD")
            atom_box.operator(BLENDMOL_OT_toggle_fixed_atom.bl_idname, icon="PINNED")
            atom_box.operator(BLENDMOL_OT_delete_atom.bl_idname, icon="TRASH")
        layout.separator()
        layout.label(text="Select two atoms to create or change a bond")
        layout.operator(BLENDMOL_OT_set_bond.bl_idname, icon="CON_TRACKTO")
        selected_bond = _selected_explicit_bond(context)
        if selected_bond:
            layout.label(text=f"Selected bond: {selected_bond[2] + 1}")
            layout.operator(BLENDMOL_OT_delete_bond.bl_idname, icon="TRASH")
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
            validation_messages = _molecule_validation_messages(molecule)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            validation_messages = ["Stored molecule data cannot be validated"]
        validation_box = layout.box()
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


class BLENDMOL_PT_constraints(Panel):
    bl_label = "UFF Constraints"
    bl_idname = "BLENDMOL_PT_constraints"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    @classmethod
    def poll(cls, context):
        return _selected_molecule_root(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        root = _selected_molecule_root(context)

        constraint_box = layout.box()
        constraint_box.label(text="Distance: select two atoms")
        constraint_box.prop(settings, "distance_constraint_target")
        constraint_box.prop(settings, "distance_constraint_strength")
        selected_pair = _selected_explicit_atoms(context)
        row = constraint_box.row(align=True)
        set_target = row.operator(
            BLENDMOL_OT_set_distance_constraint.bl_idname, text="Set Target"
        )
        set_target.use_current = False
        set_current = row.operator(
            BLENDMOL_OT_set_distance_constraint.bl_idname, text="Use Current"
        )
        set_current.use_current = True
        if selected_pair:
            pair = selected_pair[2]
            matching = next(
                (item for item in _distance_constraints(root) if item[:2] == pair), None
            )
            if matching:
                constraint_box.label(
                    text=f"Atoms {pair[0] + 1}-{pair[1] + 1}: {matching[2]:.3f} Å"
                )
                remove = constraint_box.operator(
                    BLENDMOL_OT_remove_distance_constraint.bl_idname,
                    text="Remove Selected Constraint", icon="X",
                )
                remove.remove_all = False
        constraints = _distance_constraints(root)
        constraint_box.label(text=f"Stored constraints: {len(constraints)}")
        if constraints:
            remove_all = constraint_box.operator(
                BLENDMOL_OT_remove_distance_constraint.bl_idname,
                text="Remove All Constraints", icon="TRASH",
            )
            remove_all.remove_all = True

        angle_box = layout.box()
        angle_box.label(text="Angle: select 3 atoms; active atom is the center")
        angle_box.prop(settings, "angle_constraint_target")
        angle_box.prop(settings, "angle_constraint_strength")
        row = angle_box.row(align=True)
        set_angle = row.operator(
            BLENDMOL_OT_set_angle_constraint.bl_idname, text="Set Target"
        )
        set_angle.use_current = False
        current_angle = row.operator(
            BLENDMOL_OT_set_angle_constraint.bl_idname, text="Use Current"
        )
        current_angle.use_current = True
        selected_angle = _selected_angle_atoms(context)
        if selected_angle:
            key = selected_angle[2]
            matching_angle = next(
                (item for item in _angle_constraints(root) if item[:3] == key), None
            )
            if matching_angle:
                angle_box.label(
                    text=(
                        f"Atoms {key[0] + 1}-{key[1] + 1}-{key[2] + 1}: "
                        f"{math.degrees(matching_angle[3]):.2f}°"
                    )
                )
                remove_angle = angle_box.operator(
                    BLENDMOL_OT_remove_angle_constraint.bl_idname,
                    text="Remove Selected Constraint", icon="X",
                )
                remove_angle.remove_all = False
        angle_constraints = _angle_constraints(root)
        angle_box.label(text=f"Stored constraints: {len(angle_constraints)}")
        if angle_constraints:
            remove_all_angles = angle_box.operator(
                BLENDMOL_OT_remove_angle_constraint.bl_idname,
                text="Remove All Constraints", icon="TRASH",
            )
            remove_all_angles.remove_all = True

        dihedral_box = layout.box()
        dihedral_box.label(text="Dihedral: select 4 atoms forming one bonded path")
        dihedral_box.prop(settings, "dihedral_constraint_target")
        dihedral_box.prop(settings, "dihedral_constraint_strength")
        row = dihedral_box.row(align=True)
        set_dihedral = row.operator(
            BLENDMOL_OT_set_dihedral_constraint.bl_idname, text="Set Target"
        )
        set_dihedral.use_current = False
        current_dihedral = row.operator(
            BLENDMOL_OT_set_dihedral_constraint.bl_idname, text="Use Current"
        )
        current_dihedral.use_current = True
        selected_dihedral = _selected_dihedral_atoms(context)
        if selected_dihedral:
            path = selected_dihedral[2]
            matching_dihedral = next(
                (item for item in _dihedral_constraints(root) if item[:4] == path), None
            )
            if matching_dihedral:
                dihedral_box.label(
                    text=(
                        f"Atoms {'-'.join(str(index + 1) for index in path)}: "
                        f"{math.degrees(matching_dihedral[4]):.2f}°"
                    )
                )
                remove_dihedral = dihedral_box.operator(
                    BLENDMOL_OT_remove_dihedral_constraint.bl_idname,
                    text="Remove Selected Constraint", icon="X",
                )
                remove_dihedral.remove_all = False
        dihedral_constraints = _dihedral_constraints(root)
        dihedral_box.label(text=f"Stored constraints: {len(dihedral_constraints)}")
        if dihedral_constraints:
            remove_all_dihedrals = dihedral_box.operator(
                BLENDMOL_OT_remove_dihedral_constraint.bl_idname,
                text="Remove All Constraints", icon="TRASH",
            )
            remove_all_dihedrals.remove_all = True


class BLENDMOL_PT_uff(Panel):
    bl_label = "Partial UFF"
    bl_idname = "BLENDMOL_PT_uff"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_order = 4

    @classmethod
    def poll(cls, context):
        return _selected_molecule_root(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        root = _selected_molecule_root(context)
        try:
            molecule = _deserialize_molecule(root["blend_mol_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            layout.alert = True
            layout.label(text="Stored molecule data cannot be read", icon="ERROR")
            return
        try:
            uff_report = type_molecule(molecule)
        except (IndexError, TypeError, ValueError):
            uff_report = None
        uff_box = layout
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
            uff_box.prop(settings, "show_uff_energy_breakdown", toggle=True, icon="ALIGN_JUSTIFY")
            if settings.show_uff_energy_breakdown:
                breakdown_box = uff_box.box()
                stretch = bond_stretch_energy(molecule)
                if stretch.ready:
                    breakdown_box.label(
                        text=f"Bond stretch: {stretch.energy_kcal_mol:.3f} kcal/mol ({stretch.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="Bond stretch: needs explicit hydrogens", icon="INFO")
                angle = angle_bend_energy(molecule)
                if angle.ready:
                    breakdown_box.label(
                        text=f"Angle bend: {angle.energy_kcal_mol:.3f} kcal/mol ({angle.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="Angle bend: needs explicit hydrogens", icon="INFO")
                torsion = torsion_energy(molecule)
                if torsion.ready:
                    breakdown_box.label(
                        text=f"Torsion: {torsion.energy_kcal_mol:.3f} kcal/mol ({torsion.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="Torsion: needs explicit hydrogens", icon="INFO")
                inversion = out_of_plane_energy(molecule)
                if inversion.ready:
                    breakdown_box.label(
                        text=f"Out of plane: {inversion.energy_kcal_mol:.3f} kcal/mol ({inversion.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="Out of plane: needs explicit hydrogens", icon="INFO")
                vdw = van_der_waals_energy(molecule)
                if vdw.ready:
                    breakdown_box.label(
                        text=f"van der Waals: {vdw.energy_kcal_mol:.3f} kcal/mol ({vdw.terms} terms)"
                    )
                else:
                    breakdown_box.label(text="van der Waals: needs explicit hydrogens", icon="INFO")
            total = partial_energy(molecule)
            if total.ready:
                uff_box.label(text=f"Partial UFF total: {total.energy_kcal_mol:.3f} kcal/mol")
                settings_box = uff_box.box()
                settings_box.prop(settings, "uff_iterations")
                settings_box.prop(settings, "uff_max_step")
                settings_box.prop(settings, "uff_gradient_tolerance")
                settings_box.prop(settings, "uff_energy_tolerance")
                settings_box.prop(settings, "uff_optimizer")
                settings_box.prop(settings, "uff_backend")
                compute_status = backend_status()
                settings_box.label(
                    text=f"Compute backend: {compute_status['active'].title()}",
                    icon="CHECKMARK" if compute_status["native_available"] else "INFO",
                )
                minimize_row = uff_box.row()
                minimize_row.scale_y = 1.3
                minimize_row.operator(BLENDMOL_OT_minimize_partial_uff.bl_idname, icon="MOD_SMOOTH")
                live_box = uff_box.box()
                live_box.label(text="Interactive UFF relaxation", icon="TRACKING")
                live_running = bool(
                    _LIVE_UFF_OPERATOR is not None
                    and _LIVE_UFF_OPERATOR._root_name == root.name
                )
                live_row = live_box.row()
                live_row.scale_y = 1.5
                live_row.operator(
                    BLENDMOL_OT_toggle_live_uff.bl_idname,
                    text="Stop Live UFF" if live_running else "Start Live UFF",
                    icon="PAUSE" if live_running else "PLAY",
                )
                if live_running:
                    phase = root.get("blend_mol_live_uff_phase", "ready")
                    live_box.label(text=f"Live state: {phase.title()}")
                if "blend_mol_live_uff_error" in root:
                    # Live UFF failures (e.g. the collinear-angle error) are
                    # also reported via self.report when they happen, but
                    # that status-bar message is easy to miss mid-drag; keep
                    # it visible here until the next Start Live UFF clears it.
                    live_box.label(text=root["blend_mol_live_uff_error"], icon="ERROR")
                    error_row = live_box.row()
                    error_row.prop(settings, "jitter_magnitude")
                    error_row.operator(BLENDMOL_OT_jitter_atom_positions.bl_idname, icon="MOD_NOISE")
                live_box.prop(settings, "uff_live_auto_start")
                if "blend_mol_uff_live_last" in root:
                    try:
                        live_run = json.loads(root["blend_mol_uff_live_last"])
                        live_box.label(
                            text=(
                                f"Atom {live_run['held_atom'] + 1}: "
                                f"{live_run['initial_energy_kcal_mol']:.3f} → "
                                f"{live_run['final_energy_kcal_mol']:.3f} kcal/mol"
                            )
                        )
                        if "display_update_ms" in live_run:
                            live_box.label(
                                text=(
                                    f"Compute {live_run['elapsed_ms']:.1f} ms + display "
                                    f"{live_run['display_update_ms']:.1f} ms = "
                                    f"{live_run['total_step_ms']:.1f} ms"
                                )
                            )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        live_box.label(text="Last interactive step log is unreadable", icon="ERROR")
                live_box.operator(BLENDMOL_OT_relax_around_selected_atom.bl_idname, icon="TRACKING")
                live_settings_box = live_box.box()
                live_settings_box.label(text="Live UFF settings")
                live_settings_box.prop(settings, "uff_live_steps")
                live_settings_box.prop(settings, "uff_live_target_ms")
                live_settings_box.prop(settings, "uff_live_settle_updates")
                live_settings_box.prop(settings, "uff_live_interval")
                live_settings_box.separator()
                live_settings_box.operator(BLENDMOL_OT_benchmark_uff_step.bl_idname, icon="TIME")
                if "blend_mol_uff_benchmark" in root:
                    try:
                        benchmark = json.loads(root["blend_mol_uff_benchmark"])
                        live_settings_box.label(
                            text=(
                                f"{benchmark['atom_count']} atoms / {benchmark['moving_atoms']} moving: "
                                f"{benchmark['elapsed_ms']:.1f} ms"
                            )
                        )
                        live_settings_box.label(
                            text=f"Estimated compute rate: {benchmark['estimated_hz']:.1f} Hz"
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        live_settings_box.label(text="UFF benchmark log is unreadable", icon="ERROR")
            if "blend_mol_uff_last_run" in root:
                try:
                    last_run = json.loads(root["blend_mol_uff_last_run"])
                    log_box = uff_box.box()
                    log_box.label(text="Last partial UFF minimization")
                    log_box.label(text=last_run["timestamp"])
                    log_box.label(
                        text=(
                            f"{last_run['initial_energy_kcal_mol']:.3f} → "
                            f"{last_run['final_energy_kcal_mol']:.3f} kcal/mol"
                        )
                    )
                    if (
                        "blend_mol_uff_before_positions" in root
                        and "blend_mol_uff_after_positions" in root
                    ):
                        row = log_box.row(align=True)
                        before = row.operator(
                            BLENDMOL_OT_restore_uff_coordinates.bl_idname,
                            text="Show Before", icon="LOOP_BACK",
                        )
                        before.state = "BEFORE"
                        after = row.operator(
                            BLENDMOL_OT_restore_uff_coordinates.bl_idname,
                            text="Show After", icon="CHECKMARK",
                        )
                        after.state = "AFTER"
                    log_box.label(
                        text=(
                            f"{last_run['iterations']} accepted steps; "
                            f"{last_run['termination']}; {last_run['fixed_atoms']} fixed"
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    uff_box.label(text="Last UFF log is unreadable", icon="ERROR")
            uff_box.label(text="No electrostatic terms; minimizer uses implemented terms only")
        else:
            uff_box.alert = True
            uff_box.label(text="Not ready for UFF", icon="ERROR")
            for message in uff_report.messages[:3]:
                uff_box.label(text=message)
            if len(uff_report.messages) > 3:
                uff_box.label(text=f"… and {len(uff_report.messages) - 3} more")
            uff_box.alert = False


class BLENDMOL_PT_relax(Panel):
    bl_label = "Display Relaxation & Snapshot"
    bl_idname = "BLENDMOL_PT_relax"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    @classmethod
    def poll(cls, context):
        return _selected_molecule_root(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
        root = _selected_molecule_root(context)
        layout.label(text="Simple display relaxation (not UFF)")
        layout.prop(settings, "relax_iterations")
        layout.operator(BLENDMOL_OT_relax_display_geometry.bl_idname, icon="MOD_SMOOTH")
        layout.operator(BLENDMOL_OT_save_position_snapshot.bl_idname, icon="BOOKMARKS")
        if "blend_mol_position_snapshot" in root:
            layout.operator(BLENDMOL_OT_restore_position_snapshot.bl_idname, icon="LOOP_BACK")
        layout.separator()
        layout.label(text="Stuck on an exactly-degenerate geometry (e.g. collinear)?")
        jitter_row = layout.row()
        jitter_row.prop(settings, "jitter_magnitude")
        jitter_row.operator(BLENDMOL_OT_jitter_atom_positions.bl_idname, icon="MOD_NOISE")


class BLENDMOL_PT_display(Panel):
    bl_label = "Display & Mesh Settings"
    bl_idname = "BLENDMOL_PT_display"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blend Mol"
    bl_parent_id = "BLENDMOL_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 6

    def draw(self, context):
        layout = self.layout
        settings = context.scene.blend_mol_settings
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
        layout.prop(settings, "show_implicit_hydrogens")
        layout.operator(BLENDMOL_OT_rebuild_display.bl_idname, icon="FILE_REFRESH")
        layout.separator()
        layout.label(text="Material (applies immediately)")
        layout.prop(settings, "roughness")
        layout.prop(settings, "metallic")
        layout.separator()
        mesh_box = layout.box()
        mesh_box.label(text="Mesh detail")
        mesh_box.label(text="Defaults for next import")
        mesh_box.prop(settings, "atom_segments")
        mesh_box.prop(settings, "atom_rings")
        mesh_box.prop(settings, "bond_sides")
        mesh_box.prop(settings, "bond_length_segments")
        mesh_box.separator()
        mesh_box.prop(settings, "subdivision_levels")
        mesh_box.operator(BLENDMOL_OT_add_subdivision.bl_idname, icon="MOD_SUBSURF")


CLASSES = (
    BlendMolElementColor,
    BlendMolSettings,
    BLENDMOL_OT_import_v2000,
    BLENDMOL_OT_import_xyz,
    BLENDMOL_OT_import_pdb,
    BLENDMOL_OT_import_sdf,
    BLENDMOL_OT_import_smiles,
    BLENDMOL_OT_export_v2000,
    BLENDMOL_OT_export_xyz,
    BLENDMOL_OT_export_pdb,
    BLENDMOL_OT_export_sdf,
    BLENDMOL_OT_export_uff_report,
    BLENDMOL_OT_new_molecule,
    BLENDMOL_OT_rebuild_display,
    BLENDMOL_OT_sync_atom_positions,
    BLENDMOL_OT_add_atom,
    BLENDMOL_OT_add_bonded_atom,
    BLENDMOL_OT_make_hydrogens_explicit,
    BLENDMOL_OT_change_atom_element,
    BLENDMOL_OT_set_formal_charge,
    BLENDMOL_OT_delete_atom,
    BLENDMOL_OT_set_bond,
    BLENDMOL_OT_delete_bond,
    BLENDMOL_OT_relax_display_geometry,
    BLENDMOL_OT_jitter_atom_positions,
    BLENDMOL_OT_minimize_partial_uff,
    BLENDMOL_OT_relax_around_selected_atom,
    BLENDMOL_OT_benchmark_uff_step,
    BLENDMOL_OT_toggle_live_uff,
    BLENDMOL_OT_restore_uff_coordinates,
    BLENDMOL_OT_toggle_fixed_atom,
    BLENDMOL_OT_set_distance_constraint,
    BLENDMOL_OT_remove_distance_constraint,
    BLENDMOL_OT_set_angle_constraint,
    BLENDMOL_OT_remove_angle_constraint,
    BLENDMOL_OT_set_dihedral_constraint,
    BLENDMOL_OT_remove_dihedral_constraint,
    BLENDMOL_OT_save_position_snapshot,
    BLENDMOL_OT_restore_position_snapshot,
    BLENDMOL_OT_reset_element_colors,
    BLENDMOL_OT_add_subdivision,
    BLENDMOL_PT_panel,
    BLENDMOL_PT_file_io,
    BLENDMOL_PT_edit,
    BLENDMOL_PT_constraints,
    BLENDMOL_PT_uff,
    BLENDMOL_PT_relax,
    BLENDMOL_PT_display,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.blend_mol_settings = PointerProperty(type=BlendMolSettings)
    if _queue_auto_live_uff_from_transform not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_queue_auto_live_uff_from_transform)
    for handlers in (bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre, bpy.app.handlers.load_pre):
        if _invalidate_live_uff_before_history_change not in handlers:
            handlers.append(_invalidate_live_uff_before_history_change)


def unregister():
    global _AUTO_LIVE_UFF_TIMER_PENDING, _LIVE_UFF_OPERATOR
    if _LIVE_UFF_OPERATOR is not None:
        _LIVE_UFF_OPERATOR._finish(bpy.context)
    _invalidate_live_uff_before_history_change()
    for handlers in (bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre, bpy.app.handlers.load_pre):
        if _invalidate_live_uff_before_history_change in handlers:
            handlers.remove(_invalidate_live_uff_before_history_change)
    if _queue_auto_live_uff_from_transform in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_queue_auto_live_uff_from_transform)
    if bpy.app.timers.is_registered(_start_auto_live_uff):
        bpy.app.timers.unregister(_start_auto_live_uff)
    _AUTO_LIVE_UFF_TIMER_PENDING = False
    del bpy.types.Scene.blend_mol_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
