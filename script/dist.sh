#! /usr/bin/env bash

mkdir -p dist
zip -r dist/dist.zip . -x ".venv/*" -x ".git/*" -x "script/*" -x "dist/*" -x "**/__pycache__/*" -x "**/__pycache__/*"