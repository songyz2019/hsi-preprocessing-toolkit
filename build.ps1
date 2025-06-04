#!/bin/bash

uv run pyi-makespec -F     --hidden-import=rasterio._path     --hidden-import=rasterio.mask     --hidden-import=rasterio.stack     --hidden-import=rasterio._show_versions     --hidden-import=rasterio.drivers     --hidden-import=rasterio.merge     --hidden-import=rasterio.tools     --hidden-import=rasterio.dtypes     --hidden-import=rasterio.path     --hidden-import=rasterio.transform     --hidden-import=rasterio.enums     --hidden-import=rasterio.plot     --hidden-import=rasterio.vrt     --hidden-import=rasterio.env     --hidden-import=rasterio.profiles     --hidden-import=rasterio.warp     --hidden-import=rasterio.errors     --hidden-import=rasterio.rpc     --hidden-import=rasterio.windows     --hidden-import=rasterio.abc     --hidden-import=rasterio.features     --hidden-import=rasterio.sample     --hidden-import=rasterio.control     --hidden-import=rasterio.fill     --hidden-import=rasterio.session     --hidden-import=rasterio.coords     --hidden-import=rasterio.io     --hidden-import=rasterio._base     --hidden-import=rasterio._features     --hidden-import=rasterio._transform     --hidden-import=rasterio.crs     --hidden-import=rasterio._env     --hidden-import=rasterio._filepath     --hidden-import=rasterio._version     --hidden-import=rasterio.shutil     --hidden-import=rasterio._err     --hidden-import=rasterio._fill     --hidden-import=rasterio._vsiopener     --hidden-import=rasterio._example     --hidden-import=rasterio._io     --hidden-import=rasterio._warp     --collect-data=gradio_client --collect-data=gradio     main.py

# https://github.com/pyinstaller/pyinstaller/issues/8108

uv run pyinstaller main.spec

Start-Process -FilePath "dist/main.exe" -NoNewWindow -Wait



