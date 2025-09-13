#! /usr/bin/env bash

mkdir -p dist
zip -r dist/hsi-preprocessing-toolkit-win64.zip . -x ".venv/*" -x ".git/*" -x "script/*" -x "dist/*" -x "**/__pycache__/*"