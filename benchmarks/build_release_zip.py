"""Build the Blend Mol release ZIP, preserving the native/ subfolder.

PowerShell's Compress-Archive flattens any subfolder given as an individual
-Path entry (it drops the folder, keeping only the file's base name at the
archive root). That flattened the native/blend_mol_uff.dll path in the
0.1.49/0.1.50 archives; this script avoids it by writing each member with
an explicit arcname via Python's zipfile module.

Usage: run from Blender's own bundled Python (or any CPython, no bpy
needed), from the repository root:

    & 'D:\\SteamLibrary\\steamapps\\common\\Blender\\5.2\\python\\bin\\python.exe' benchmarks\\build_release_zip.py
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "blend_mol"
MANIFEST = SOURCE / "blender_manifest.toml"

FILES = (
    "blender_manifest.toml",
    "__init__.py",
    "live_worker.py",
    "molfile.py",
    "native_backend.py",
    "uff.py",
    "README.md",
    "native/blend_mol_uff.dll",
)


def _read_version() -> str:
    text = MANIFEST.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Cannot find version in {MANIFEST}")
    return match.group(1)


def build(destination: Path | None = None) -> Path:
    version = _read_version()
    if destination is None:
        destination = ROOT / "dist" / f"blend_mol-{version}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            source_path = SOURCE / relative
            if not source_path.is_file():
                raise RuntimeError(f"Missing required release file: {source_path}")
            archive.write(source_path, arcname=relative)
    return destination


def verify(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    missing = set(FILES) - names
    if missing:
        raise RuntimeError(f"Built archive is missing: {sorted(missing)}")
    if "native/blend_mol_uff.dll" not in names:
        raise RuntimeError("native/ subfolder was flattened in the built archive")


if __name__ == "__main__":
    output = build(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    verify(output)
    print(f"Built {output} ({output.stat().st_size} bytes)")
