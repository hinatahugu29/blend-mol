# Blend Mol UFF benchmarks

Check bonded-atom additions and bond-order edits after explicit hydrogen
preparation and UFF minimization:

```powershell
& 'D:\SteamLibrary\steamapps\common\Blender\blender.exe' --background --factory-startup --python-exit-code 1 --python benchmarks\smoke_blender_bond_valence.py
```

This verifies rejected edits preserve data, revision, display object identity,
and UFF history, while valid additions, charged nitrogen and incremental repair
of overbonded imports remain possible.

Validate the release ZIP with Steam Blender's own Python, without installing
the extension or changing preferences:

```powershell
& 'D:\SteamLibrary\steamapps\common\Blender\blender.exe' --background --factory-startup --python-exit-code 1 --python benchmarks\validate_distribution.py -- dist\blend_mol-0.1.52.zip
```

This checks the packaged DLL path, an actual native kernel evaluation against
its reference formula, explicit Python bypass, Auto fallback for a missing DLL,
and the forced-Rust error. It uses a temporary extraction directory and releases
the Windows DLL before cleanup. This gate targets Windows AMD64; it does not
establish compatibility with other CPUs or Blender versions.

The Blender background smoke suite also checks move detection, pending timer
cleanup and cancellation before history changes, and transform-cancel event
propagation. Run it with `--python-exit-code 1` to report assertion failures:

```powershell
& 'D:\SteamLibrary\steamapps\common\Blender\blender.exe' --background --factory-startup --python-exit-code 1 --python benchmarks\smoke_blender_background_worker.py
```

Run from the repository root:

```powershell
python benchmarks\benchmark_uff_scaling.py
```

The script measures one partial-UFF step without changing the input molecule.
It reports medians for two deliberately different workloads:

- separated explicit methane fragments, representing a light case with few
  residual numerical-gradient terms;
- replicated explicit triphenylene, representing a torsion/inversion-heavy
  aromatic case.

The 30 / roughly 100 / 300 atom sizes are the release decision points recorded
in `ROADMAP.md`. Times do not include Blender mesh updates, so they are a lower
bound for interactive latency.

To compare the first Rust kernel directly with its Python reference:

```powershell
python benchmarks\benchmark_native_vdw.py
```

This reports numerical agreement and speedup separately. A speedup in this
isolated kernel does not by itself prove that the complete optimizer is faster.

For a direct end-to-end comparison of one complete partial-UFF step using the
same torsion/inversion-heavy inputs:

```powershell
python benchmarks\benchmark_backend_comparison.py
```

For randomized one-step and multi-step parity, fixed-atom behavior, and Rust
repeatability:

```powershell
python benchmarks\validate_backends.py
```

This validation also compares ten stateful FIRE2 steps on Python and Rust with
two fixed atoms. The Blender background smoke test separately continues FIRE2
state across two chunks to cover the Live UFF execution boundary.

To compare steepest descent, FIRE2, and L-BFGS on the same three distorted
30-atom aromatic structures:

```powershell
python benchmarks\benchmark_optimizers.py
```

The output records every run plus median wall time and final energy. L-BFGS
state continuation is also covered by the unit and Blender smoke suites.

With Avogadro 2.0.0 installed in its default Windows location, compare energy
components against its bundled Open Babel 3 UFF implementation:

```powershell
python benchmarks\compare_avogadro_uff.py
```

To run the broader six-molecule external regression gate (requires Avogadro 2
in its default Windows location):

```powershell
python benchmarks\compare_avogadro_matrix.py
```

The matrix covers alkanes, an alcohol, pyramidal nitrogen, water, and an
aromatic ring. It fails if any total UFF energy differs by more than 0.1 kJ/mol.
