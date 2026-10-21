# AvogadroLibs review for Blend Mol

Reviewed upstream: <https://github.com/OpenChemistry/avogadrolibs>

## Useful upstream areas

- `avogadro/calc/uff.*` and `uffdata.h`: independent UFF calculator and data.
- `avogadro/calc/energycalculator.*`: stable energy/gradient abstraction.
- `avogadro/calc/energyoptimizer.*`: optimization separated from the UI.
- `avogadro/qtplugins/autoopt/autoopt.*`: direct atom/selection manipulation and
  interactive molecule state.
- `avogadro/qtplugins/forcefield/*`: backend selection, frozen atoms,
  constraints, tolerances, and background calculation integration.

AvogadroLibs is BSD-3-Clause licensed. Formulae, parameter interpretation,
test fixtures, and small appropriately attributed portions can therefore be
studied or reused, subject to retaining the required notice. Any imported code
must be identified explicitly instead of being silently copied.

## Decisions for Blend Mol

1. Keep Blender registration, selection, undo, timers, and mesh updates in
   Python. Do not link the Qt rendering or plugin layers into Blender.
2. Keep the Rust DLL C ABI small and independent of Blender/Python objects,
   matching Avogadro's separation between calculation and GUI layers.
3. Introduce a prepared calculation context so atom typing and topology terms
   are built once, then reused by energy, gradient, and optimization calls.
4. Move the complete optimization loop off Blender's UI callback before Live
   UFF is considered production-ready. Exchange immutable input snapshots and
   accept only the newest completed result.
5. Add cancellation/generation tokens so stale calculations cannot overwrite
   coordinates after another drag update, molecule edit, or tool stop.
6. Expand constraints beyond fixed atoms only after the basic background
   worker is safe: distance, angle, and torsion constraints are the next useful
   categories.
7. Continue using Avogadro/Open Babel outputs as external regression oracles,
   while Python remains Blend Mol's internal reference implementation.

## Not adopted directly

- Qt event, widget, rendering, and plugin classes: Blender already owns these
  responsibilities and bundling Qt would greatly increase compatibility risk.
- Avogadro's C++ molecule object as the DLL boundary: this would couple the
  add-on to a C++ ABI and its dependency graph. Plain contiguous arrays remain
  safer for a Windows Blender extension.
- Open Babel as a required runtime: it remains an optional external validation
  tool, not an installation requirement.

## Concrete follow-up gates

- Background worker never calls Blender APIs outside the main thread.
- A moved atom remains constrained to the latest user position.
- Results carry a molecule revision and are discarded when stale.
- Start/stop/edit/undo and Blender shutdown cancel safely.
- 50-cycle and longer soak tests show no stale writes or worker leakage.
- Python/Rust and Avogadro comparison suites remain green after the worker
  architecture changes.

## Implemented in Blend Mol 0.1.32

- Live UFF calculations run on a dedicated daemon worker and never access
  Blender data from that thread.
- Immutable molecule snapshots cross the UI/calculation boundary.
- Pending drag calculations are coalesced: only the latest waiting snapshot is
  retained while one calculation is running.
- Every request has a monotonically increasing generation. Results from older
  generations, stopped sessions, or a disabled add-on are discarded.
- A Blender 5.2.2 background smoke test exercises 50 calculation cycles and
  verifies that the worker exits without leaving its thread behind.

The remaining safety step is a stored molecule revision so topology edits and
Undo can invalidate a Live UFF session even if they occur through another
operator.

Blend Mol 0.1.33 completes that step: structural edit operators increment a
stored revision, while a topology signature independently catches revisions
restored by Blender Undo. Live UFF validates both before accepting any worker
completion.

Blend Mol 0.1.34 begins the Forcefield-style constraint layer with persistent
two-atom distance restraints. They use a harmonic penalty with an analytic
gradient, remain independent of the UFF energy report, and participate in full,
manual, and background Live UFF minimization.

Blend Mol 0.1.35 extends this layer to persistent three-atom angle restraints.
The active selected atom defines the center, targets are stored in degrees for
readability, and calculation uses radians with an analytic gradient.

Blend Mol 0.1.36 completes the first constraint set with persistent four-atom
dihedral restraints. Bond connectivity determines the path order, signed
targets wrap correctly at ±180°, and only the four participating atoms enter
the restraint's central-difference gradient.

Blend Mol 0.1.37 implements the prepared calculation context from decision 3.
Atom typing, native torsion/inversion term arrays, and the residual-gradient
atom set are immutable and reused while coordinates change. A topology key
rejects accidental reuse after structural edits. On the 30-atom reference,
100 repeated one-step calls improved from about 8.66 to 7.78 ms per call.

Blend Mol 0.1.38 changes Live UFF from a single response per movement into an
AutoOpt-style settling chain. Accepted work resubmits through the same latest-
only worker until convergence, line-search stall, or a user-set update cap.
New movement invalidates the older chain, and stopping records before/after
coordinates plus the number of applied updates.

Blend Mol 0.1.39 adopts the forcefield plugin's user-visible convergence
tolerance concept. The Cartesian gradient threshold is validated, shared by
all optimization paths, and stored with normal and Live UFF run metadata.

Blend Mol 0.1.40 extends cancellation beyond stale-result rejection. Every
Live UFF request owns a cancellation event; replacement input and tool stop
signal the old event, and the optimizer checks it at iteration and line-search
boundaries. A native kernel already in progress finishes safely, but no further
optimization work is scheduled for that stale request.

Blend Mol 0.1.41 further separates topology preparation from coordinate
evaluation. Bonded-neighbor lists and non-1-2/non-1-3 van der Waals pair
parameters are now immutable prepared-context data, reused by energy,
analytic-gradient, and line-search evaluation. Regression tests compare cached
and uncached paths after a coordinate displacement so the optimization does not
trade reference-path agreement for speed.

Blend Mol 0.1.42 adopts the wall-clock chunk adaptation exposed by Avogadro's
current `EnergyOptimizer`. Live UFF starts from the configured interactive step
count, then adjusts subsequent chunks multiplicatively toward a configurable
compute-time budget. Smoothing and hard 1–10 step bounds prevent timing noise
from destabilizing interaction, while each run log records the completed and
next requested chunk sizes.

Blend Mol 0.1.43 adds the energy-change convergence criterion exposed by
force-field optimization tools. It complements the Cartesian gradient test:
an accepted step whose objective improvement is at or below the configured
kcal/mol threshold terminates normally. Full, manual, benchmark, and Live UFF
paths share the value and retain it in their run metadata.

Blend Mol 0.1.44 consolidates optimizer-facing evaluation behind one objective
and Cartesian-gradient function, following Avogadro's `EnergyCalculator`
boundary. UFF terms, distance/angle/dihedral restraints, native or Python
residual gradients, and frozen-atom projection are now assembled in one place.
This also fixes fixed atoms contributing to the convergence norm even though
their coordinates could not move.

Blend Mol 0.1.45 follows Avogadro's optimizer conventions for size-independent
interactive control. Convergence now uses the largest absolute gradient
component over movable atoms, and step scaling caps the displacement of each
atom separately. Larger molecules therefore no longer receive progressively
smaller per-atom moves merely because more Cartesian coordinates are present.

Blend Mol 0.1.46 expands decision 7 from one aromatic reference to methane,
ethane, ethanol, ammonia, water, and benzene generated by Avogadro's bundled
Open Babel. The new matrix exposed a 13.62 kJ/mol ammonia discrepancy: the old
single central-plane approximation omitted UFF's three `N_3` out-of-plane
terms. Implementing the three permutations and carbonyl-carbon barrier exception
reduces the matrix's maximum total-energy difference to about 0.153 kJ/mol.

Blend Mol 0.1.47 resolves most of that remaining maximum. Open Babel adds a
small exponential repulsion to planar angle terms to penalize near-zero bond
angles. Applying the same documented compatibility behavior, including its
analytic derivative and kJ-to-kcal conversion, reduces benzene's total-energy
difference to about 0.032 kJ/mol and triphenylene's angle difference from
0.516 to 0.014 kJ/mol. No Open Babel source code is copied into the add-on.

Blend Mol 0.1.48 adds an independently implemented FIRE2 optimizer based on
the algorithmic behavior exposed by Avogadro's `EnergyOptimizer`. Velocity,
time step, mixing coefficient, and the positive-power counter are explicit
immutable state and continue across Live UFF chunks. Moving the held atom,
changing topology, stopping the tool, or supplying incompatible state resets
that momentum safely. Per-atom displacement remains capped by Blend Mol's
existing setting, and fixed atoms have both gradient and velocity projected
to zero. The former backtracking steepest-descent path remains selectable as
the conservative reference while FIRE2 is evaluated before any hybrid
FIRE/L-BFGS default is adopted.

The new validation covers chunked-versus-continuous FIRE2 equivalence, fixed
atoms, cancellation, state mismatch recovery, Blender 5.2.2 background use,
and Python/Rust parity. Across three randomized ten-step fixed-atom cases, the
largest Python/Rust difference was about `3.3e-12 kcal/mol` in energy and
`6.1e-14 Å` in any coordinate.
