# Rust UFF backend boundary

## Decision

The Python implementation remains the reference and fallback backend. A
Windows-first Rust prototype is justified by the 2026-10-21 scaling benchmark:
the torsion/inversion-heavy 120-atom case takes about 1.13 seconds per step,
well above the 100 ms interactive ceiling.

## Ownership boundary

Python keeps Blender registration, panels, selection, timers, cancellation,
undo/snapshots, molecule serialization, and mesh updates. Rust receives only
plain molecular data and performs UFF energy, gradient, and bounded local
minimization.

The first bridge should be a C-compatible DLL loaded with `ctypes`. Its inputs
are contiguous arrays for atomic types, coordinates, bonds and bond orders,
formal charges, and fixed/movable masks. Its outputs are coordinates, energy
components, iteration counts, and an explicit status/error code. No Blender or
Python object may cross the boundary.

## Required behavior

- Loading failure selects the Python backend automatically.
- A cancellation flag is checked between optimization iterations.
- The caller owns all buffers; the DLL must not retain their addresses.
- Panics never cross the FFI boundary.
- Backend selection and result status are visible in the Blender panel/report.

## First implemented kernel (0.1.28)

The van der Waals energy and Cartesian gradient are the first native kernel.
It is selected with `Auto / Python / Rust` in the UFF panel. `Auto` falls back
to Python if the DLL cannot load; `Rust` treats an unavailable or failed DLL as
an error. The Rust kernel divides sufficiently large pair lists into independent
chunks using the standard library's available CPU parallelism, then reduces
per-worker energies and gradients without shared mutable buffers.

This is intentionally not a claim that the whole optimizer is faster yet.
Python/Rust equality tests and isolated/full-step benchmarks decide whether each
kernel stays native. The currently dominant torsion/inversion residual remains
Python work and is the next measured migration candidate.

## Batched residual gradient (0.1.29)

Torsion and inversion term definitions are prepared once in Python, then their
base energy and every active Cartesian forward difference are evaluated in one
DLL call. Coordinate perturbations are independent and are divided across the
available CPU threads. Each worker reuses one coordinate buffer and writes only
to its private result list, followed by a deterministic indexed merge.

The direct full-step comparison on the development machine uses the same
torsion/inversion-heavy structures and three-run medians:

| Atoms | Python | Rust | Speedup |
| ---: | ---: | ---: | ---: |
| 30 | 62.55 ms | 7.19 ms | 8.70x |
| 120 | 878.32 ms | 40.21 ms | 21.84x |
| 300 | 5512.51 ms | 186.80 ms | 29.51x |

An additional energy-only van der Waals migration was measured and rejected:
it increased the 300-atom full step from about 186 ms to 207 ms because Python
array preparation and FFI transfer outweighed the native calculation. The code
path was removed. This measured rollback is part of the backend policy: native
code is retained only when the end-to-end path benefits.

## Validation status (0.1.30)

Randomized parity now covers ten distorted one-step structures and five
distorted twenty-step structures with two fixed atoms. Maximum observed
Python/Rust differences were `7.82e-13 kcal/mol` in final energy and
`1.51e-14 Å` in coordinates. Fifty repeated Rust runs produced bit-for-bit
identical reported energies and coordinates on the development machine.

The loader is tested for automatic Python fallback when the DLL is missing and
for an explicit error when Rust is forced. Native tests reject invalid atom
indices and degenerate torsion geometry. Blender 5.2.2 completed fifty Live UFF
start/move/evaluate/stop cycles without changing the object count or leaving a
running/error state behind.

The native path remains numerically aligned after correcting the reference
Python UFF formulas against Avogadro 2 / Open Babel 3. The optimized
triphenylene structures differ by only `0.000241 Å` RMS across all pairwise
distances after 500 requested steps.

## Acceptance gate

- Energy components, gradients, and optimized coordinates agree with the
  Python reference within documented tolerances.
- Existing unit tests run against both backends from the same fixtures.
- The heavy 120-atom benchmark completes one step in under 100 ms on the
  development machine.
- Blender 5.2.2 remains responsive, cancellation is safe, and repeated
  start/stop cycles do not leak or crash.
- 30 / 120 / 300 atom results and DLL/fallback behavior are recorded before
  the native backend becomes the default.
