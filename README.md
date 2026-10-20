# Blend Mol

Blender 5.2+ extension prototype for importing V2000 MOL files and displaying
their atom and bond graph as a ball-and-stick model.

## Current scope

- Imports atom coordinates, single/double/triple bonds, and formal charges
  (including `M  CHG` records) from MDL V2000 MOL files.
- Draws atoms with an element colour and bonds as cylinders.
- Adds implicit hydrogens for the common neutral valences used by the supplied
  triphenylene example.
- Lists the elements present in Blend Mol objects in the 3D Viewport sidebar,
  where their colours can be changed immediately for existing molecules.
- Provides import-time mesh-density defaults and a non-destructive Subdivision
  Surface command for existing atoms and bonds.
- Stores the imported atom and bond graph on the molecule root, so the selected
  molecule can be rebuilt non-destructively from the **Blend Mol** sidebar.
- Supports adding an atom at the 3D cursor, changing an explicit atom's element
  or formal charge, and deleting an explicit atom together with its connected
  bonds.
- Supports creating, changing, and deleting single/double/triple bonds, with
  valence warnings for common neutral elements.
- Includes a non-UFF display-geometry relaxation tool and position snapshots
  for returning to a structure before relaxation.
- Lets users pin atoms so they remain fixed during display-geometry relaxation;
  these fixed-atom constraints are retained for future UFF optimization.
- Includes an in-panel UFF preparation check. It currently assigns conservative
  UFF types for H/C/N/O/F and reports atoms outside that supported subset.
  For structures with all hydrogens explicit, it also shows the partial UFF
  bond-stretch, angle-bend, proper-torsion and van der Waals energies;
  inversion, electrostatic terms and optimization are still pending.
- Keeps two-dimensional MOL coordinates planar. Three-dimensional structure
  generation is deliberately not part of this first prototype.

## Install in Blender 5.2

Package this directory as a ZIP containing `blender_manifest.toml` and
`__init__.py`. In Blender, use **Edit → Preferences → Get Extensions → Install
from Disk**, then select that ZIP. The **Blend Mol** panel appears in the 3D
Viewport sidebar.

## Test data

`217-59-4(トリフェニレン).mol` is the first import test case. It is a 2D V2000
Molfile containing 18 carbon atoms and 21 bonds; Blend Mol displays 12 implicit
hydrogens from its bond orders.

## Roadmap

The implementation roadmap, including the planned built-in UFF optimizer, is in
[`ROADMAP.md`](ROADMAP.md).
