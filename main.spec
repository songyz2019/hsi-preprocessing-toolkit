# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('gradio_client')
datas += collect_data_files('gradio')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['rasterio._path', 'rasterio.mask', 'rasterio.stack', 'rasterio._show_versions', 'rasterio.drivers', 'rasterio.merge', 'rasterio.tools', 'rasterio.dtypes', 'rasterio.path', 'rasterio.transform', 'rasterio.enums', 'rasterio.plot', 'rasterio.vrt', 'rasterio.env', 'rasterio.profiles', 'rasterio.warp', 'rasterio.errors', 'rasterio.rpc', 'rasterio.windows', 'rasterio.abc', 'rasterio.features', 'rasterio.sample', 'rasterio.control', 'rasterio.fill', 'rasterio.session', 'rasterio.coords', 'rasterio.io', 'rasterio._base', 'rasterio._features', 'rasterio._transform', 'rasterio.crs', 'rasterio._env', 'rasterio._filepath', 'rasterio._version', 'rasterio.shutil', 'rasterio._err', 'rasterio._fill', 'rasterio._vsiopener', 'rasterio._example', 'rasterio._io', 'rasterio._warp'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
