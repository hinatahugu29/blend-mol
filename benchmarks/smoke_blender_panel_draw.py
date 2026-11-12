"""Exercise every BLENDMOL_PT_* panel's draw() without a real UI context.

Blender's --background mode has no area/region, so bpy.types.UILayout
cannot be constructed the way an interactive redraw provides it. This
substitutes a minimal fake layout implementing the small subset of the
UILayout API the panels actually call (label, operator, prop, box, row,
separator, the `alert` attribute), so draw() runs for real and any
AttributeError/TypeError from the 2026-10 panel restructuring (rearranging the workflow panels)
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

    def split(self, factor=0.5, align=False):
        return FakeLayout()


class FakePanel:
    def __init__(self, panel_cls):
        self._cls = panel_cls
        self.layout = FakeLayout()

    def draw(self, context):
        self._cls.draw(self, context)


class PreparationLayout(FakeLayout):
    """Record the actionable hydrogen button, including its visual state."""
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.enabled = True

    def operator(self, idname, text=None, icon="NONE"):
        self.events.append((idname, text, self.enabled, self.alert))
        return super().operator(idname, text, icon)

    def row(self, align=False):
        return PreparationLayout(self.events)

    def column(self, align=False):
        return PreparationLayout(self.events)


def assert_preparation(context, enabled, alert):
    events = []
    panel = FakePanel(addon.BLENDMOL_PT_edit)
    panel.layout = PreparationLayout(events)
    panel.draw(context)
    button = next(event for event in events if event[0] == "blendmol.make_hydrogens_explicit")
    assert button[2:] == (enabled, alert), button


def draw_all(context, label):
    panels = tuple(cls for cls in addon.CLASSES if cls.__name__.startswith("BLENDMOL_PT_"))
    for panel_cls in panels:
        if hasattr(panel_cls, "poll") and not panel_cls.poll(context):
            continue
        FakePanel(panel_cls).draw(context)
    print(f"  drew all applicable panels for: {label}")


addon.register()
try:
    context = addon.bpy.context
    settings = context.scene.blend_mol_settings
    assert settings.uff_live_auto_start is True
    # Other smoke cases exercise data changes without starting a modal session.
    settings.uff_live_auto_start = False

    # --- State 1: nothing selected -----------------------------------
    addon.bpy.ops.object.select_all(action="DESELECT")
    context.view_layer.objects.active = None
    draw_all(context, "no selection")
    assert_preparation(context, False, False)

    # --- State 2: molecule root selected, no atom/bond -----------------
    molecule = read_v2000(ROOT / "tests" / "fixtures" / "217-59-4(トリフェニレン)2.mol")
    root = addon.display._create_molecule(molecule, settings, "panel_smoke")
    collection = addon.scene_data._molecule_collection(root)
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

    pending = addon.panels._pending_hydrogens(root)
    assert pending == 0, pending

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
    addon.scene_data._store_distance_constraints(root, ((0, 1, 1.5, 500.0),))
    addon.scene_data._store_angle_constraints(root, ((0, 1, 2, 1.9, 200.0),))
    addon.scene_data._store_dihedral_constraints(root, ((0, 1, 2, 3, 0.5, 100.0),))
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

    addon.scene_data._store_distance_constraints(root, ())
    addon.scene_data._store_angle_constraints(root, ())
    addon.scene_data._store_dihedral_constraints(root, ())

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
    assert addon.panels._pending_hydrogens(root) == 0
    assert addon.panels._pending_hydrogens(None) is None

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

    # Hydrogen preparation must work even when implicit hydrogens are hidden.
    from blend_mol.molfile import read_smiles
    implicit_root = addon.display._create_molecule(read_smiles("C"), settings, "pending_hydrogens")
    addon.bpy.ops.object.select_all(action="DESELECT")
    implicit_root.select_set(True)
    context.view_layer.objects.active = implicit_root
    settings.show_implicit_hydrogens = False
    assert addon.panels._pending_hydrogens(implicit_root) == 4
    assert_preparation(context, True, True)
    draw_all(context, "implicit hydrogens hidden but preparation still needed")
    assert addon.bpy.ops.blendmol.make_hydrogens_explicit() == {"FINISHED"}
    assert addon.panels._pending_hydrogens(implicit_root) == 0
    assert_preparation(context, False, False)
    draw_all(context, "hydrogen preparation complete")
    atom_object(0).select_set(True)
    context.view_layer.objects.active = atom_object(0)

    # --- State 11: a stored Live UFF error must be visible in the panel ---
    # Previously blend_mol_live_uff_error was written but never read back
    # anywhere, so a Live UFF failure (e.g. the collinear-angle error) left
    # no visible trace once the modal session stopped.
    class RecordingFakeLayout(FakeLayout):
        def __init__(self, labels):
            super().__init__()
            self.labels = labels

        def label(self, text="", icon="NONE"):
            self.labels.append(text)

        def box(self):
            return RecordingFakeLayout(self.labels)

        def row(self, align=False):
            return RecordingFakeLayout(self.labels)

        def column(self, align=False):
            return RecordingFakeLayout(self.labels)

    root["blend_mol_live_uff_error"] = "UFF: collinear atoms have no stable angle gradient; moved coordinates kept"
    recorded_labels = []
    panel = FakePanel(addon.BLENDMOL_PT_uff)
    panel.layout = RecordingFakeLayout(recorded_labels)
    panel.draw(context)
    assert "collinear atoms have no stable angle gradient" in " ".join(recorded_labels), (
        "the stored Live UFF error must be shown in the Partial UFF panel", recorded_labels
    )
    del root["blend_mol_live_uff_error"]

    print(
        "BLEND_MOL_PANEL_DRAW_SMOKE_OK",
        "root_panel_ok", "file_io_panel_ok", "edit_panel_ok",
        "constraints_panel_ok", "uff_panel_ok", "relax_panel_ok", "display_panel_ok",
        "ten_selection_states_ok", "live_uff_error_shown_ok",
    )
finally:
    addon.unregister()
