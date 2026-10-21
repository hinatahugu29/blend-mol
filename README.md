# Blend Mol

Blend Mol is a Blender 5.2+ molecular editing and visualization add-on with a
dependency-free Python UFF reference path and an optional parallel Windows
Rust backend.

The installable add-on lives entirely in [`blend_mol`](blend_mol). Zip that
directory so `blender_manifest.toml` is at the archive root, or use the
versioned archive generated under the local `dist` directory.

Current capabilities include starting from an empty scene, importing and
exporting common molecular formats, atom/bond editing, SMILES stereochemistry,
partial UFF energy and minimization, constraints, fixed atoms, and background
Live UFF relaxation. Version 0.1.49 adds stateful L-BFGS and makes it the
default after direct convergence and timing comparisons with FIRE2 and the
steepest-descent reference. Version 0.1.50 fixes safe handling of a late timer
event immediately after Live UFF is stopped. Version 0.1.51 fixes the packaged
path of the Windows Rust DLL. Version 0.1.52 can start Live UFF automatically
when an explicit atom is moved, then stop after the surrounding atoms settle.
Version 0.1.53 invalidates background work before Undo, Redo, or file loading,
preserves Blender's transform-cancel events, and validates deferred auto-start targets.
Version 0.1.54 restores the Live UFF session's starting coordinates on Escape
or right-click, including the surrounding atoms, and discards cancelled results.
Version 0.1.55 checks valence before adding bonded atoms or creating/changing
bonds, preventing explicit hydrogens left after UFF preparation from being
silently overbonded by subsequent edits. Version 0.1.56 fixes the depsgraph
comparison that silently prevented automatic Live UFF from ever starting on a
real atom move, since every Blender depsgraph update carries an evaluated
copy of the moved object, never the original data-block it was compared
against. Version 0.1.57 fixes the UFF electronegativity parameter for
chlorine, bromine, and iodine, which held the published table's hardness
column instead, shifting equilibrium bond lengths for carbon/nitrogen/
oxygen-halogen bonds; it is cross-checked against Open Babel 3.1.1's own
UFF parameter table in an automated regression test. Version 0.1.58 fixes
atom typing for cumulated double-bond systems (CO2, ketenes, allenes,
isocyanates): the central carbon/nitrogen previously typed sp2 from its
highest bond order alone, bending to about 120° instead of the correct
180° linear sp geometry, which a user noticed in a live UFF minimization.
Version 0.1.59 fixes minimization aborting with a "collinear" error for any
linear (sp) atom with a further substituent, including plain nitriles like
acetonitrile: the analytic angle gradient treated the 0/0 limit at an sp
center's own 180° equilibrium as a real degeneracy, and torsions were
defined through sp centers even though no torsional barrier exists there.
Version 0.1.60 adds silicon, boron, and hypervalent phosphorus UFF typing,
and widens neutral tetravalent P=O and sulfoxide/sulfone typing to match
Open Babel's own fallback behavior (confirmed against a standalone Open
Babel 3.1.1 install); it also fixes sulfur's torsion treatment missing the
same Group 6 special case oxygen already had. Version 0.1.61 locks the
Cl/Br/I electronegativity fix in with a permanent halomethane regression
test, and adds a GitHub Actions workflow that runs the full dependency-free
and Blender-based test suites automatically on every push. Version 0.1.62
adds PDB export (coordinates and connectivity; PDB has no bond-order field,
so re-importing loses bond order the same way any PDB import already does).
Version 0.1.63 speeds up the plain-Python van der Waals energy check that
minimization's line search calls on every trial step, roughly halving the
300-atom case and bringing it to about the 100 ms/10 Hz interactive target.
Version 0.1.64 fixes torsion energy through a sulfur single bond: a type
name's last character cannot distinguish P/S hybridisation from oxidation
state, and a membership check let sp2-sp3 pairs fall into the sp3-sp3
branch; both were confirmed against Open Babel 3.1.1's own C++ source
(forcefielduff.cpp), closing the sulfoxide/sulfone torsion gap to within
0.1 kJ/mol. Version 0.1.65 adds SDF export (a single V2000 record plus
`$$$$`), completing the PDB/SDF export pair alongside existing MOL/XYZ
export. Version 0.1.66 splits the single flat N-panel into seven
collapsible, workflow-grouped sub-panels (no operator or property removed),
emphasizes the three primary actions, and collapses the UFF energy
breakdown and less-used sections by default.

- User and installation details: [`blend_mol/README.md`](blend_mol/README.md)
- Development roadmap and performance criteria: [`ROADMAP.md`](ROADMAP.md)
- AvogadroLibs design review: [`AVOGADRO_REVIEW.md`](AVOGADRO_REVIEW.md)
- Rust backend design and fallback behavior: [`RUST_BACKEND.md`](RUST_BACKEND.md)
- Reproducible validation commands: [`benchmarks/README.md`](benchmarks/README.md)

Run the dependency-free test suite from the repository root:

```powershell
python -m unittest discover -s tests
```
