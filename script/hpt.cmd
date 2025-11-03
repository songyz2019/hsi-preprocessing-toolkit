@echo off

REM CONFIGS
set "PACKAGE=hsi-preprocessing-toolkit==2.0.0"
set "EXECUTABLE=hsi_preprocessing_toolkit"
set "UV_PYTHON=3.13"
set "UV_MANAGED_PYTHON=1"
set "UV_INDEX_STRATEGY=unsafe-best-match"
set "UV_INDEX=https://mirrors.aliyun.com/pypi/simple"
set "UV_DEFAULT_INDEX=https://pypi.org/simple"

REM COMMANDS
where uv >nul 2>nul && goto :UV_INSTALLED
echo installing uv, make sure you have internet access to astral.sh and pypi.org
powershell -Command "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
REM RESTART TO MAKE PATH WORKS
start "" "%~f0"
exit /b
:UV_INSTALLED

uv tool install %PACKAGE%
uv tool run --from %PACKAGE% %EXECUTABLE%