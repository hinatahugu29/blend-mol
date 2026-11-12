"""Scene settings and the per-element colour palette properties."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
    IntProperty, StringProperty,
)
from bpy.types import PropertyGroup

from .display import _material
from .elements import ELEMENT_SYMBOLS, UNKNOWN_COLOR
from .native_backend import set_backend_preference


ELEMENT_ITEMS = tuple((element, element, "") for element in ELEMENT_SYMBOLS)


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
        default=UNKNOWN_COLOR,
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
        default=True,
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
