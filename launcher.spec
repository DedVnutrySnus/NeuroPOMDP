# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build for the NeuroPOMDP Windows launcher.

Build from the project root:
    pyinstaller --clean --noconfirm launcher.spec
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata


PROJECT_ROOT = Path(SPEC).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT / "neuropomdp"


# Streamlit uses dynamic imports and package metadata.
streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all("streamlit")
streamlit_hiddenimports += collect_submodules("streamlit")

# The launcher creates a temporary bootstrap script containing:
#     from neuropomdp.dashboard.app import main
# so PyInstaller cannot reliably see that import statically.
neuropomdp_hiddenimports = collect_submodules("neuropomdp")

datas = [
    *streamlit_datas,
    *copy_metadata("streamlit"),
]

binaries = [
    *streamlit_binaries,
]

hiddenimports = [
    *streamlit_hiddenimports,
    *neuropomdp_hiddenimports,
]


a = Analysis(
    [str(PACKAGE_ROOT / "launcher.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="NeuroPOMDP",
)
