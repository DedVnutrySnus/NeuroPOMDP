# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the NeuroPOMDP Windows desktop build."""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules


try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:  # PyInstaller executes spec files without __file__
    project_root = Path.cwd().resolve()
entry_script = project_root / "neuropomdp" / "launcher.py"
icon_path = project_root / "assets" / "neuropomdp.ico"


def _unique_extend(target: list, values: list) -> None:
    """Extend a list without duplicates while preserving order."""

    for value in values:
        if value not in target:
            target.append(value)


datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []

for package in ("streamlit", "matplotlib", "numpy", "scipy", "pandas", "pyarrow", "altair", "pydeck", "watchdog"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    _unique_extend(datas, package_datas)
    _unique_extend(binaries, package_binaries)
    _unique_extend(hiddenimports, package_hiddenimports)

for package in ("jax", "jaxlib"):
    _unique_extend(hiddenimports, collect_submodules(package))
    _unique_extend(binaries, collect_dynamic_libs(package))

_unique_extend(
    hiddenimports,
    [
        "neuropomdp.dashboard.app",
        "neuropomdp.dashboard.components",
        "neuropomdp.dashboard.formatting",
        "neuropomdp.dashboard.state",
    ],
)

a = Analysis(
    [str(entry_script)],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeuroPOMDP",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(icon_path),
    disable_windowed_traceback=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NeuroPOMDP",
)
