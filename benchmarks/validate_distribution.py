"""Validate kernels from an extracted release ZIP using the current Python.

Run with Blender's --background --factory-startup --python, followed by -- ZIP.
The installed extension and Blender preferences are not modified.
"""

from pathlib import Path
import json
import math
import platform
import sys
import tempfile
import zipfile
from _ctypes import FreeLibrary


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if len(arguments) != 1:
        raise SystemExit("Usage: validate_distribution.py [--] release.zip")
    archive_path = Path(arguments[0]).resolve()
    with tempfile.TemporaryDirectory(prefix="blend_mol_release_") as directory:
        destination = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            required = {"blender_manifest.toml", "native/blend_mol_uff.dll", "native_backend.py", "uff.py"}
            if not required.issubset(archive.namelist()):
                raise RuntimeError("Release ZIP is missing required files")
            for name in archive.namelist():
                if not (destination / name).resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError("Unsafe archive member")
            archive.extractall(destination)
        sys.path.insert(0, str(destination))
        import native_backend as backend

        if Path(backend.__file__).resolve().parent != destination.resolve():
            raise RuntimeError("Validation must load the extracted distribution")
        coordinates = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        pairs = [(0, 1, 0.105, 3.851)]
        backend.set_backend_preference("rust")
        status = backend.backend_status()
        if not status["native_available"]:
            raise RuntimeError(f"Packaged Rust DLL failed to load: {status}")
        energy, gradient = backend.vdw_energy_gradient(coordinates, pairs)
        ratio6 = (pairs[0][3] / 4.0) ** 6
        expected = pairs[0][2] * (ratio6 * ratio6 - 2 * ratio6)
        if not math.isclose(energy, expected, abs_tol=1e-12):
            raise RuntimeError("Packaged kernel energy differs from reference")
        if not all(math.isfinite(component) for vector in gradient for component in vector):
            raise RuntimeError("Packaged gradient is not finite")
        backend.set_backend_preference("python")
        if backend.vdw_energy_gradient(coordinates, pairs) is not None:
            raise RuntimeError("Python selection did not bypass the DLL")

        # Simulate a missing DLL only in this isolated process, without touching
        # the packaged or installed library on disk.
        # All native calls have returned; release the Windows loader reference
        # before the temporary extraction directory is cleaned up.
        FreeLibrary(backend._LIBRARY._handle)
        backend._LIBRARY = None
        backend._LOAD_ERROR = None
        backend._library_path = lambda: destination / "missing.dll"
        backend.set_backend_preference("auto")
        if backend.vdw_energy_gradient(coordinates, pairs) is not None:
            raise RuntimeError("Auto did not fall back when the DLL was missing")
        fallback_status = backend.backend_status()
        backend.set_backend_preference("rust")
        try:
            backend.vdw_energy_gradient(coordinates, pairs)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Forced Rust did not reject a missing DLL")
        print(json.dumps({
            "archive": str(archive_path),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "native": status,
            "kernel_energy_error": abs(energy - expected),
            "python_bypass": True,
            "missing_dll_fallback": fallback_status["active"],
            "forced_rust_missing_dll_error": True,
        }, indent=2))


if __name__ == "__main__":
    main()
