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
    "display.py",
    "elements.py",
    "embed.py",
    "live_uff.py",
    "live_worker.py",
    "molfile.py",
    "native_backend.py",
    "ops_constraints.py",
    "ops_display.py",
    "ops_edit.py",
    "ops_io.py",
    "ops_uff.py",
    "panels.py",
    "properties.py",
    "scene_data.py",
    "uff.py",
    "README.md",
    "LICENSE",
)

NATIVE_LIBRARIES = (
    "native/blend_mol_uff.dll",
    "native/libblend_mol_uff.so",
    "native/libblend_mol_uff.dylib",
)


def _read_version() -> str:
    text = MANIFEST.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Cannot find version in {MANIFEST}")
    return match.group(1)


def build(destination: Path | None = None, require_all_native: bool = False) -> Path:
    version = _read_version()
    if destination is None:
        destination = ROOT / "dist" / f"blend_mol-{version}.zip"
    unlisted = sorted(
        path.name for path in SOURCE.glob("*.py") if path.name not in FILES
    )
    if unlisted:
        # A module missing from FILES would install fine here but fail to
        # import from the release zip.
        raise RuntimeError(f"Add-on modules missing from FILES: {unlisted}")
    present_native = [rel for rel in NATIVE_LIBRARIES if (SOURCE / rel).is_file()]
    if require_all_native and len(present_native) != len(NATIVE_LIBRARIES):
        missing_native = sorted(set(NATIVE_LIBRARIES) - set(present_native))
        raise RuntimeError(f"Missing required native libraries: {missing_native}")
    if not present_native:
        raise RuntimeError("No native UFF library found under blend_mol/native/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in (*FILES, *present_native):
            source_path = SOURCE / relative
            if not source_path.is_file():
                raise RuntimeError(f"Missing required release file: {source_path}")
            archive.write(source_path, arcname=relative)
    return destination


def verify(path: Path, require_all_native: bool = False) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    missing = set(FILES) - names
    if missing:
        raise RuntimeError(f"Built archive is missing: {sorted(missing)}")
    present_native = set(NATIVE_LIBRARIES) & names
    if not present_native:
        raise RuntimeError("No native/ library found in the built archive (or subfolder was flattened)")
    if require_all_native and present_native != set(NATIVE_LIBRARIES):
        raise RuntimeError(f"Built archive is missing native libraries: {sorted(set(NATIVE_LIBRARIES) - present_native)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    require_all = "--require-all-native" in args
    positional = [arg for arg in args if arg != "--require-all-native"]
    output = build(Path(positional[0]) if positional else None, require_all_native=require_all)
    verify(output, require_all_native=require_all)
    print(f"Built {output} ({output.stat().st_size} bytes)")
