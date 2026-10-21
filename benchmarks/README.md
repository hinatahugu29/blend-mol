# Blend Mol UFF benchmarks

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
