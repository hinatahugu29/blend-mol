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

import argparse
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

PLATFORM_CONFIGS = {
    "windows": ("windows-x64", ["windows-x64"], ("native/blend_mol_uff.dll",)),
    "linux": ("linux-x64", ["linux-x64"], ("native/libblend_mol_uff.so",)),
    "macos": ("macos-universal", ["macos-arm64", "macos-x64"], ("native/libblend_mol_uff.dylib",)),
}


def _read_version() -> str:
    text = MANIFEST.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Cannot find version in {MANIFEST}")
    return match.group(1)


def _manifest_for_platforms(platforms: list[str]) -> str:
    text = MANIFEST.read_text(encoding="utf-8")
    rendered = ", ".join(f'"{item}"' for item in platforms)
    updated, count = re.subn(
        r'^platforms\s*=\s*\[[^\]]*\]',
        f"platforms = [{rendered}]",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Cannot update platforms in {MANIFEST}")
    return updated


def build(
    destination: Path | None = None,
    require_all_native: bool = False,
    target_platform: str = "all",
) -> Path:
    version = _read_version()
    unlisted = sorted(
        path.name for path in SOURCE.glob("*.py") if path.name not in FILES
    )
    if unlisted:
        # A module missing from FILES would install fine here but fail to
        # import from the release zip.
        raise RuntimeError(f"Add-on modules missing from FILES: {unlisted}")

    if target_platform == "all":
        present_native = [rel for rel in NATIVE_LIBRARIES if (SOURCE / rel).is_file()]
        if require_all_native and len(present_native) != len(NATIVE_LIBRARIES):
            missing_native = sorted(set(NATIVE_LIBRARIES) - set(present_native))
            raise RuntimeError(f"Missing required native libraries: {missing_native}")
        if not present_native:
            raise RuntimeError("No native UFF library found under blend_mol/native/")
        manifest_override = None
        default_name = f"blend_mol-{version}.zip"
    else:
        suffix, platforms, required_libs = PLATFORM_CONFIGS[target_platform]
        missing_native = [rel for rel in required_libs if not (SOURCE / rel).is_file()]
        if missing_native:
            raise RuntimeError(f"Missing native library for {target_platform}: {missing_native}")
        present_native = list(required_libs)
        manifest_override = _manifest_for_platforms(platforms)
        default_name = f"blend_mol-{version}-{suffix}.zip"

    if destination is None:
        destination = ROOT / "dist" / default_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in (*FILES, *present_native):
            if relative == "blender_manifest.toml" and manifest_override is not None:
                archive.writestr("blender_manifest.toml", manifest_override.encode("utf-8"))
                continue
            source_path = SOURCE / relative
            if not source_path.is_file():
                raise RuntimeError(f"Missing required release file: {source_path}")
            archive.write(source_path, arcname=relative)
    return destination


def verify(
    path: Path,
    require_all_native: bool = False,
    target_platform: str = "all",
) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    missing = set(FILES) - names
    if missing:
        raise RuntimeError(f"Built archive is missing: {sorted(missing)}")
    present_native = set(NATIVE_LIBRARIES) & names
    if target_platform == "all":
        if not present_native:
            raise RuntimeError("No native/ library found in the built archive (or subfolder was flattened)")
        if require_all_native and present_native != set(NATIVE_LIBRARIES):
            raise RuntimeError(
                f"Built archive is missing native libraries: {sorted(set(NATIVE_LIBRARIES) - present_native)}"
            )
    else:
        expected = set(PLATFORM_CONFIGS[target_platform][2])
        if present_native != expected:
            raise RuntimeError(f"Expected native libraries {sorted(expected)}, got {sorted(present_native)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Blend Mol release ZIP")
    parser.add_argument("destination", nargs="?", type=Path, default=None)
    parser.add_argument("--require-all-native", action="store_true")
    parser.add_argument("--platform", choices=("all", "windows", "linux", "macos"), default="all")
    args = parser.parse_args()
    output = build(args.destination, require_all_native=args.require_all_native, target_platform=args.platform)
    verify(output, require_all_native=args.require_all_native, target_platform=args.platform)
    print(f"Built {output} ({output.stat().st_size} bytes)")
