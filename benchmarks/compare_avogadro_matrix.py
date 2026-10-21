"""Cross-check several molecular classes with Avogadro's bundled Open Babel UFF."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blend_mol"))

from molfile import read_v2000
from uff import (
    angle_bend_energy, bond_stretch_energy, partial_energy, torsion_energy,
    van_der_waals_energy,
)


AVOGADRO_BIN = Path(r"C:\Program Files\Avogadro2\bin")
KCAL_TO_KJ = 4.184
CASES = {
    "methane": "C",
    "ethane": "CC",
    "ethanol": "CCO",
    "ammonia": "N",
    "water": "O",
    "benzene": "c1ccccc1",
}
PATTERNS = {
    "bond": r"BOND STRETCHING ENERGY\s*=\s*([-+0-9.]+)",
    "angle": r"ANGLE BENDING ENERGY\s*=\s*([-+0-9.]+)",
    "torsion": r"TORSIONAL ENERGY\s*=\s*([-+0-9.]+)",
    "vdw": r"VAN DER WAALS ENERGY\s*=\s*([-+0-9.]+)",
    "total": r"TOTAL ENERGY\s*=\s*([-+0-9.]+)",
}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["BABEL_DATADIR"] = str(AVOGADRO_BIN / "data")
    return environment


def _generate(smiles: str, output: Path) -> None:
    subprocess.run(
        [str(AVOGADRO_BIN / "obabel.exe"), f"-:{smiles}", "-O", str(output), "--gen3d", "-h"],
        check=True, capture_output=True, text=True, env=_environment(),
    )


def _openbabel_energy(path: Path) -> dict[str, float]:
    completed = subprocess.run(
        [str(AVOGADRO_BIN / "obenergy.exe"), "-ff", "UFF", str(path)],
        check=True, capture_output=True, text=True, env=_environment(),
    )
    values = {}
    for name, pattern in PATTERNS.items():
        match = re.search(pattern, completed.stdout)
        if match is None:
            raise RuntimeError(f"Cannot parse Open Babel {name} energy for {path.name}")
        values[name] = float(match.group(1))
    return values


def _blend_mol_energy(path: Path) -> dict[str, float]:
    molecule = read_v2000(path)
    reports = {
        "bond": bond_stretch_energy(molecule),
        "angle": angle_bend_energy(molecule),
        "torsion": torsion_energy(molecule),
        "vdw": van_der_waals_energy(molecule),
        "total": partial_energy(molecule),
    }
    if any(not report.ready for report in reports.values()):
        messages = [message for report in reports.values() for message in report.messages]
        raise RuntimeError(f"Blend Mol UFF failed for {path.name}: {messages}")
    return {name: report.energy_kcal_mol * KCAL_TO_KJ for name, report in reports.items()}


def compare_matrix() -> dict:
    results = []
    with tempfile.TemporaryDirectory(prefix="blend_mol_avogadro_matrix_") as directory:
        temporary = Path(directory)
        for name, smiles in CASES.items():
            path = temporary / f"{name}.mol"
            _generate(smiles, path)
            blend_mol = _blend_mol_energy(path)
            openbabel = _openbabel_energy(path)
            differences = {key: blend_mol[key] - openbabel[key] for key in blend_mol}
            results.append({
                "name": name,
                "smiles": smiles,
                "blend_mol_kj_mol": blend_mol,
                "openbabel_kj_mol": openbabel,
                "difference_kj_mol": differences,
                "absolute_total_difference_kj_mol": abs(differences["total"]),
            })
    maximum = max(item["absolute_total_difference_kj_mol"] for item in results)
    return {
        "comparison": "Blend Mol vs Avogadro 2.0.0 bundled Open Babel 3 UFF matrix",
        "cases": results,
        "maximum_absolute_total_difference_kj_mol": maximum,
        "gate_kj_mol": 0.1,
        "gate_passed": maximum <= 0.1,
    }


if __name__ == "__main__":
    report = compare_matrix()
    print(json.dumps(report, indent=2))
    if not report["gate_passed"]:
        raise SystemExit("Avogadro/Open Babel UFF comparison exceeded the 0.1 kJ/mol gate")
