# Blend Mol

Blender 5.2+ extension prototype for importing V2000 MOL files and displaying
their atom and bond graph as a ball-and-stick model.

## Current scope

- Imports atom coordinates, single/double/triple bonds, and formal charges
  (including `M  CHG` records) from MDL V2000 MOL files.
- Exports the selected molecule, including its current explicit atom positions,
  bond orders, and formal charges, as an MDL V2000 MOL file.
- Exports atom elements and Cartesian coordinates as a standard XYZ file when
  only coordinate exchange is needed.
- Exports a JSON validation report containing partial UFF atom types, energy
  terms, fixed atoms, and the most recent minimization log.
- Starts a new molecule from an empty Blender scene, placing its first atom at
  the 3D cursor without importing a MOL file.
- Imports XYZ atom coordinates without inventing bonds or implicit hydrogens;
  bonds can then be created in Blend Mol.
- Imports PDB `ATOM` / `HETATM` coordinates and single bonds recorded in
  `CONECT` entries, without inferring additional hydrogens.
- Imports the first V2000 structure record in an SDF file.
- Creates a basic organic molecule directly from SMILES, including common
  elements, formal charges, branches, ring closures, and single/double/triple
  bonds, plus common aromatic notation such as `c1ccccc1`. Aromatic bonds are
  converted to editable alternating single/double bonds. It supplies a 2D
  starting layout. Tetrahedral `@` / `@@` markers are retained on atoms and
  produce opposite-handed 3D starting arrangements. Directional `/` / `\\`
  markers around double bonds are also retained and produce opposite-side or
  matching-side starting arrangements.
- Draws atoms with an element colour and bonds as cylinders.
- Adds implicit hydrogens for the common neutral valences used by the supplied
  triphenylene example.
- Converts displayed implicit hydrogens into editable atoms and bonds when a
  fully explicit structure is needed for partial UFF energy evaluation.
- Lists the elements present in Blend Mol objects in the 3D Viewport sidebar,
  where their colours can be changed immediately for existing molecules.
- Provides import-time mesh-density defaults and a non-destructive Subdivision
  Surface command for existing atoms and bonds, including configurable
  lengthwise support segments for explicit bonds.
- Stores the imported atom and bond graph on the molecule root, so the selected
  molecule can be rebuilt non-destructively from the **Blend Mol** sidebar.
- Supports adding an atom at the 3D cursor, changing an explicit atom's element
  or formal charge, and deleting an explicit atom together with its connected
  bonds.
- Can add a new atom and create its bond to the selected atom in one step,
  making carbon-chain construction possible without a separate two-atom selection.
- Supports creating, changing, and deleting single/double/triple bonds, with
  valence warnings for common neutral elements.
- Includes a non-UFF display-geometry relaxation tool and position snapshots
  for returning to a structure before relaxation.
- Lets users pin atoms so they remain fixed during display-geometry relaxation;
  fixed-atom constraints are applied by the UFF optimizer. Version 0.1.34 adds
  saved harmonic distance restraints between selected atom pairs. A target can
  be entered in angstroms or captured from the current geometry; the restraint
  is applied by normal minimization, manual interactive steps, and Live UFF.
  Version 0.1.35 adds the corresponding three-atom angle restraint. Select
  three atoms with the intended center atom active, then enter a degree target
  or capture the current angle. Angle restraints use an analytic gradient and
  are also shared by every minimization path.
  Version 0.1.36 adds four-atom dihedral restraints. For four selected atoms,
  Blend Mol infers one unambiguous bonded A-B-C-D path, stores a signed degree
  target, and uses the shortest periodic difference across the ±180° boundary.
  Its gradient is centrally differentiated only for the four participating
  atoms and is available in full, manual, and Live UFF minimization.
  Version 0.1.37 adds a prepared UFF calculation context inspired by the
  calculator/optimizer separation in AvogadroLibs. Atom typing, residual term
  definitions, and numerical-gradient atom membership are prepared once per
  topology and reused across every step and across Live UFF drag updates.
  Version 0.1.38 keeps scheduling background updates after each movement while
  accepted steps continue, stopping on convergence, a stalled line search, or
  a configurable update cap. A newer movement supersedes that settling chain.
  Stopping Live UFF saves the session's before/after coordinates for comparison.
  Version 0.1.39 exposes the Cartesian gradient convergence tolerance used by
  the optimizer. Full, manual, benchmark, and Live UFF paths share the same
  setting, and run logs retain the selected tolerance.
  Version 0.1.40 adds cooperative cancellation. New drag input signals the
  older minimization to stop at its next iteration or line-search boundary;
  stopping Live UFF does the same before invalidating the result generation.
  Version 0.1.41 extends the prepared UFF context with immutable bonded-neighbor
  and van der Waals pair tables. Coordinate-only Live UFF updates reuse these
  topology-derived tables for energy, analytic-gradient, and line-search
  evaluation while retaining the Python path as the reference implementation.
  Version 0.1.42 adopts Avogadro's wall-clock chunk adaptation for interactive
  optimization. The configured interactive step count is the initial chunk;
  subsequent Live UFF updates smoothly adapt between one and ten steps to fit
  the user-selected compute budget.
  Version 0.1.43 adds a shared energy-change convergence tolerance alongside
  the gradient tolerance. Full, manual, benchmark, and Live UFF paths use and
  record the same setting, and settling stops once an accepted improvement is
  smaller than the configured kcal/mol threshold.
  Version 0.1.44 introduces one calculation-facing objective/gradient boundary
  inspired by Avogadro's EnergyCalculator. It accumulates UFF and restraint
  terms together and projects frozen-atom gradients to zero before optimization,
  preventing fixed atoms from distorting convergence and step normalization.
  Version 0.1.45 makes convergence and movement independent of molecule size:
  the gradient tolerance applies to the largest movable Cartesian component,
  and UFF Max Atom Step caps each atom's displacement rather than the norm of
  the whole molecular displacement vector.
  Version 0.1.46 corrects inversion evaluation to the three UFF out-of-plane
  permutations per trigonal center, adds pyramidal N_3 and the carbonyl-carbon
  barrier exception, and validates six molecular classes against Avogadro's
  bundled Open Babel UFF with a 1 kJ/mol total-energy gate.
  Version 0.1.47 adds Open Babel's short-angle repulsion for planar UFF centers
  to both energy and analytic derivative evaluation. The six-molecule external
  gate is tightened to 0.1 kJ/mol; its observed maximum is about 0.032 kJ/mol.
  Version 0.1.48 adds a selectable stateful FIRE2 optimizer. Its velocity,
  adaptive time step, and mixing state continue across Live UFF background
  chunks instead of restarting at every update. Moving the held atom resets
  momentum, while fixed atoms remain fixed.
  Version 0.1.49 adds stateful **L-BFGS**, including continuation of its limited
  curvature history across Live UFF chunks. It converged faster and more
  reliably than both FIRE2 and steepest descent in the recorded three-case
  aromatic benchmark, so **L-BFGS** is now the default. All three remain
  selectable under **UFF Optimizer** for normal, manual, benchmark, and Live
  UFF operations.
  Version 0.1.50 fixes a Live UFF shutdown race: a timer event already queued
  when the user stops the tool now exits safely instead of accessing the
  worker after it has been released.
  Version 0.1.51 restores the `native` directory in the distribution ZIP so
  the Windows Rust DLL is found after Blender installation. The loader also
  accepts the accidentally flattened DLL location used by 0.1.49 and 0.1.50.
  Version 0.1.52 adds an opt-in **Start Live UFF When Moving an Atom** switch.
  It detects a transform of the active explicit atom, begins a background Live
  UFF relaxation, and stops after the configured settling sequence completes.
  Version 0.1.53 cancels pending and running work before Undo, Redo, and file
  loading. Escape and right-click also reach Blender's transform operator.
  Deferred automatic startup verifies the same atom is still selected and has
  actually moved; calculation failures terminate the session. Full rollback of
  surrounding atoms on transform cancellation remains pending.
- Includes an in-panel UFF preparation check. It currently assigns conservative
  UFF types for H/C/N/O/P/S/F/Cl/Br/I in conservative bonding states and reports atoms outside that supported subset.
  For structures with all hydrogens explicit, it also shows the partial UFF
  bond-stretch, angle-bend, proper-torsion, inversion and van der Waals
  energies, and can minimize that explicitly labelled subset with fixed atoms.
  Each minimization saves its start/end energy, accepted steps and stop reason
  with the molecule in the Blender file. The coordinates immediately before
  and after the last run are retained for one-click comparison and are included
  in the JSON report. Failed runs leave the original coordinates untouched.
  As a bridge toward live manipulation, a selected atom can be moved and held
  temporarily while a configurable small number of partial UFF steps relaxes
  the surrounding atoms. Explicit atom and bond objects are updated in place,
  preserving mesh identity and selection instead of rebuilding the display.
  A timer-driven Live UFF mode watches the selected atom and relaxes its
  surroundings only when that atom moves; its update interval is configurable.
  Its optional automatic mode starts on an atom move and stops after settling.
  Version 0.1.32 runs this calculation on a dedicated background worker. Drag
  updates replace any pending older request, and generation numbers prevent a
  stale result from overwriting newer coordinates. Blender objects remain
  strictly on the main thread, and stopping Live UFF invalidates in-flight work.
  Version 0.1.33 also tracks a molecule revision and an Undo-safe topology
  signature. Atom, element, charge, or bond edits during Live UFF stop the
  session before an older calculation can be applied.
  A one-step benchmark measures elapsed milliseconds and estimated compute
  frequency without applying the calculated coordinates.
  The UFF compute backend can be set to **Auto**, **Python**, or **Rust**.
  Version 0.1.28 includes a Windows Rust DLL for the van der Waals energy and
  gradient kernel. Auto uses it when available and otherwise returns to the
  reference Python path; Python always bypasses the DLL. Large pair lists are
  divided across available CPU threads, while small lists remain single-threaded
  to avoid parallel startup overhead.
  Torsion and inversion forward differences are also sent as one native batch;
  their independent coordinate perturbations run in parallel without repeated
  Python/DLL crossings. Python remains the reference path for comparison and
  explicit fallback.
  Bond-stretch, angle-bend and van der Waals forces use analytic gradients. The
  remaining torsion and inversion terms reuse their current energy with forward
  finite differences and share atom typing within each perturbed evaluation.
  Atoms that participate in neither remaining term are omitted from numerical
  perturbation entirely.
  The triphenylene reference has also been compared with Avogadro 2.0.0's
  bundled Open Babel 3 UFF: initial total energy differs by about 0.289%, and
  the 500-step optimized structures differ by about 0.000241 Å RMS across
  all pairwise distances.
  Electrostatic terms and a complete UFF optimizer are still pending.
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
