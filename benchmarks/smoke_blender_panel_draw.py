"""Exercise every BLENDMOL_PT_* panel's draw() without a real UI context.

Blender's --background mode has no area/region, so bpy.types.UILayout
cannot be constructed the way an interactive redraw provides it. This
substitutes a minimal fake layout implementing the small subset of the
UILayout API the panels actually call (label, operator, prop, box, row,
separator, the `alert` attribute), so draw() runs for real and any
AttributeError/TypeError from the 2026-10 panel restructuring (splitting
one flat panel into BLENDMOL_PT_file_io/edit/constraints/uff/relax/display)
surfaces immediately, across a range of selection states that exercise most
conditional branches. It does not check visual layout, only that draw()
completes without raising.
"""

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

addon = importlib.import_module("blend_mol")
from blend_mol.molfile import read_v2000


class FakeOperatorProps:
    def __setattr__(self, name, value):
        self.__dict__[name] = value


class FakeLayout:
    def __init__(self):
        self.alert = False

    def label(self, text="", icon="NONE"):
        pass

    def separator(self):
        pass

    def operator(self, idname, text=None, icon="NONE"):
        return FakeOperatorProps()

    def prop(self, data, name, text=None, icon="NONE", toggle=None):
        pass

    def box(self):
        return FakeLayout()

    def row(self, align=False):
        return FakeLayout()

    def column(self, align=False):
        return FakeLayout()


class FakePanel:
    def __init__(self, panel_cls):
        self._cls = panel_cls
        self.layout = FakeLayout()

    def draw(self, context):
        self._cls.draw(self, context)


def draw_all(context, label):
    panels = (
        addon.BLENDMOL_PT_panel,
        addon.BLENDMOL_PT_file_io,
        addon.BLENDMOL_PT_edit,
        addon.BLENDMOL_PT_constraints,
        addon.BLENDMOL_PT_uff,
        addon.BLENDMOL_PT_relax,
        addon.BLENDMOL_PT_display,
    )
    for panel_cls in panels:
        if hasattr(panel_cls, "poll") and not panel_cls.poll(context):
            continue
        FakePanel(panel_cls).draw(context)
    print(f"  drew all applicable panels for: {label}")


addon.register()
try:
    context = addon.bpy.context
    settings = context.scene.blend_mol_settings

    # --- State 1: nothing selected -----------------------------------
    addon.bpy.ops.object.select_all(action="DESELECT")
    context.view_layer.objects.active = None
    draw_all(context, "no selection")

    # --- State 2: molecule root selected, no atom/bond -----------------
    molecule = read_v2000(ROOT / "217-59-4(トリフェニレン)2.mol")
    root = addon._create_molecule(molecule, settings, "panel_smoke")
    collection = addon._molecule_collection(root)
    addon.bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    context.view_layer.objects.active = root
    draw_all(context, "root selected, no atom")

    def atom_object(index):
        return next(
            obj for obj in collection.objects
            if obj.get("blend_mol_kind") == "atom"
            and not obj.get("blend_mol_implicit")
            and obj.get("blend_mol_atom_index") == index
        )

    def bond_object(index):
        return next(
            obj for obj in collection.objects
            if obj.get("blend_mol_kind") == "bond"
            and obj.get("blend_mol_bond_index") == index
        )

    # --- State 3: one explicit atom selected ----------------------------
    addon.bpy.ops.object.select_all(action="DESELECT")
    atom_object(0).select_set(True)
    context.view_layer.objects.active = atom_object(0)
    draw_all(context, "one atom selected")

    # --- State 4: two atoms selected (bond/distance constraint branch) -
    addon.bpy.ops.object.select_all(action="DESELECT")
    atom_object(0).select_set(True)
    atom_object(1).select_set(True)
    context.view_layer.objects.active = atom_object(1)
    draw_all(context, "two atoms selected")

    # --- State 5: a bond selected ----------------------------------------
    addon.bpy.ops.object.select_all(action="DESELECT")
    bond_object(0).select_set(True)
    context.view_layer.objects.active = bond_object(0)
    draw_all(context, "bond selected")

    # --- State 6: with a stored distance/angle/dihedral constraint ------
    addon._store_distance_constraints(root, ((0, 1, 1.5, 500.0),))
    addon._store_angle_constraints(root, ((0, 1, 2, 1.9, 200.0),))
    addon._store_dihedral_constraints(root, ((0, 1, 2, 3, 0.5, 100.0),))
    addon.bpy.ops.object.select_all(action="DESELECT")
    atom_object(0).select_set(True)
    atom_object(1).select_set(True)
    context.view_layer.objects.active = atom_object(1)
    draw_all(context, "with stored constraints, two atoms selected (distance match)")

    addon.bpy.ops.object.select_all(action="DESELECT")
    for index in (0, 1, 2):
        atom_object(index).select_set(True)
    context.view_layer.objects.active = atom_object(1)
    draw_all(context, "with stored constraints, three atoms selected (angle match)")

    addon.bpy.ops.object.select_all(action="DESELECT")
    for index in (0, 1, 2, 3):
        atom_object(index).select_set(True)
    context.view_layer.objects.active = atom_object(1)
    draw_all(context, "with stored constraints, four atoms selected (dihedral match)")

    addon._store_distance_constraints(root, ())
    addon._store_angle_constraints(root, ())
    addon._store_dihedral_constraints(root, ())

    # --- State 7: energy breakdown toggled on/off -----------------------
    addon.bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    context.view_layer.objects.active = root
    settings.show_uff_energy_breakdown = True
    draw_all(context, "energy breakdown expanded")
    settings.show_uff_energy_breakdown = False
    draw_all(context, "energy breakdown collapsed")

    # --- State 8: after a real partial UFF minimization (last-run log) -
    settings.uff_iterations = 3
    minimize_result = addon.bpy.ops.blendmol.minimize_partial_uff()
    assert minimize_result == {"FINISHED"}, minimize_result
    draw_all(context, "after minimize_partial_uff (last-run log + restore buttons)")

    # --- State 9: a position snapshot exists -----------------------------
    assert addon.bpy.ops.blendmol.save_position_snapshot() == {"FINISHED"}
    draw_all(context, "with a saved position snapshot")

    # --- State 10: fixed atom toggled on ----------------------------------
    addon.bpy.ops.object.select_all(action="DESELECT")
    atom_object(0).select_set(True)
    context.view_layer.objects.active = atom_object(0)
    assert addon.bpy.ops.blendmol.toggle_fixed_atom() == {"FINISHED"}
    draw_all(context, "fixed atom selected")
    addon.bpy.ops.blendmol.toggle_fixed_atom()

    print(
        "BLEND_MOL_PANEL_DRAW_SMOKE_OK",
        "root_panel_ok", "file_io_panel_ok", "edit_panel_ok",
        "constraints_panel_ok", "uff_panel_ok", "relax_panel_ok", "display_panel_ok",
        "ten_selection_states_ok",
    )
finally:
    addon.unregister()
