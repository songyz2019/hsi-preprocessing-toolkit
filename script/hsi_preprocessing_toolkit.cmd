@echo off
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo "uv not found, installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)

uvx hsi_preprocessing_toolkit