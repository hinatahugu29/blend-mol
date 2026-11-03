"""Compare UFF energies from the 2D SMILES layout and the 3D embedding (Blender).

Hydrogen placement uses Blender's mathutils, so run inside Blender:

    blender --background --factory-startup --python benchmarks/compare_smiles_layout_uff.py

For each SMILES, both starting geometries get explicit hydrogens and the
same L-BFGS partial UFF minimization (300 iterations). The table shows the
starting and final energies and how the run ended.
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
addon = importlib.import_module("blend_mol")

MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "beta-glucose": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "reported pyranose": "OC[C@@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O",
    "chain polyol": "OC[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)CO",
    "menthol": "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O",
    "cholesterol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
}


def run(smiles: str, embed_3d: bool):
    molecule = addon.molfile.read_smiles(smiles, "test", embed_3d=embed_3d)
    explicit, _added = addon.ops_edit._with_hydrogens_explicit(molecule)
    result = addon.uff.minimize_partial_energy(explicit, 300, 0.05, algorithm="LBFGS")
    if not result.ready:
        return None, None, "; ".join(result.messages)[:60]
    return result.initial_energy_kcal_mol, result.final_energy_kcal_mol, result.termination


print(f"{'molecule':18s} {'2D start':>12s} {'2D final':>10s} {'3D start':>10s} {'3D final':>10s}")
worse = 0
for name, smiles in MOLECULES.items():
    start_2d, final_2d, end_2d = run(smiles, False)
    start_3d, final_3d, end_3d = run(smiles, True)

    def fmt(value):
        return f"{value:.4g}" if value is not None else "failed"

    print(f"{name:18s} {fmt(start_2d):>12s} {fmt(final_2d):>10s} {fmt(start_3d):>10s} {fmt(final_3d):>10s}"
          f"   [{end_2d} | {end_3d}]")
    if final_3d is None or (final_2d is not None and final_3d > final_2d + 1.0):
        worse += 1
print("3D_WORSE_THAN_2D:", worse)
print("BLEND_MOL_SMILES_LAYOUT_UFF_COMPARE_DONE")
