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

- User and installation details: [`blend_mol/README.md`](blend_mol/README.md)
- Development roadmap and performance criteria: [`ROADMAP.md`](ROADMAP.md)
- AvogadroLibs design review: [`AVOGADRO_REVIEW.md`](AVOGADRO_REVIEW.md)
- Rust backend design and fallback behavior: [`RUST_BACKEND.md`](RUST_BACKEND.md)
- Reproducible validation commands: [`benchmarks/README.md`](benchmarks/README.md)

Run the dependency-free test suite from the repository root:

```powershell
python -m unittest discover -s tests
```
