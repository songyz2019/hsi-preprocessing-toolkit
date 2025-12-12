# HSI Preprocessing Toolkit


[![PyPI - Version](https://img.shields.io/pypi/v/hsi-preprocessing-toolkit.svg)](https://pypi.org/project/hsi-preprocessing-toolkit)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/hsi-preprocessing-toolkit)](https://pypi.org/project/hsi-preprocessing-toolkit)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hsi-preprocessing-toolkit.svg)](https://pypi.org/project/hsi-preprocessing-toolkit)
![GitHub Created At](https://img.shields.io/github/created-at/songyz2019/hsi-preprocessing-toolkit)
![GitHub License](https://img.shields.io/github/license/songyz2019/hsi-preprocessing-toolkit)

![](asset/screenshot.jpg)

HSI Preprocessing Toolkit (HPT, formerly HDR2MAT) is a hyperspectral image preprocessing toolset that:
1. Read the raw data from the HSI camera, and convert it into `.mat` file
2. Read the `.mat` file
3. Preview HSI, and convert it to RGB `.png` file
4. Crop and rotate the HSI and preview in realtime
5. Select spectrals of interest visually and save them into a `.mat` file
6. Mix multiple HSI images with layers.
7. Some other utils


## Usage
### Manual
1. Download [hpt.cmd](https://github.com/songyz2019/hsi-preprocessing-toolkit/blob/main/script/hpt.cmd) from GitHub release
2. Double click the `hpt.cmd` file. 

> **TIPS**: Make sure you have internet access during the first start

### Install with uv
1. Install [uv](https://docs.astral.sh/uv/) directly or with pip: `pip install uv`
2. Install HPT: `uv tool install hsi-preprocessing-toolkit`
3. Start HPT: `hsi_preprocessing_toolkit`

### Install with pip
1. Install HPT: `pip install hsi-preprocessing-toolkit`
2. Start HPT: `hsi_preprocessing_toolkit`

> **TIPS**: It's not recommend to install CLI tools directly with pip, please use [uv tool](https://docs.astral.sh/uv/guides/tools/) or [pipx](https://pipx.pypa.io/) to install python applications in isolated environments.


## Credit
1. [uv](https://docs.astral.sh/uv/) for providing a new reliable solution for Python application distribution.  
2. `gradio` for modern Python Data Science UI 
3. `rasterio` for remote sensing data reading
4. `scipy`, `numpy`, `matplotlib` and `einops`
5. For more projects, see `pyproject.toml`


## License

```text
Copyright (C) 2025  songyz2019

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
