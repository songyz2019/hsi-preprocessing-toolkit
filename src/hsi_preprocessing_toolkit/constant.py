import gradio as gr

CONSTS = dict(
    homepage_url = "https://github.com/songyz2019/hsi-preprocessing-toolkit",
)

i18n = gr.I18n(**{
    'en': {
        "about.tab_title": "About",
        "about.title": "HSI Preprocessing Toolkit",
        "about.description": "A Hyperspectral Image Preprocessing Toolkit from HSI Camera to Machine Learning Dataset",
        "about.homepage": "主页",
        "about.license": """Copyright (C) 2025  songyz2019

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.""",

        "hsi_processing.tab_title": "HSI Processing",
        "hsi_processing.load": "Load",
        "hsi_processing.current_layer": "Selected Layer",
        "hsi_processing.upload_instructions": "**Upload one of the following formats:**\n1. One .hdr file + one raw data file without extension\n2. One .mat file",
        "hsi_processing.input_format": "Input Image Shape Format",
        "hsi_processing.data_files": "Data Files",
        "hsi_processing.manual_normalize": "Manual Normalize (affects preview only)",
        "hsi_processing.normalize_min": "Normalize Min",
        "hsi_processing.normalize_max": "Normalize Max",
        "hsi_processing.wavelength_start": "Wavelength Range Start",
        "hsi_processing.wavelength_end": "Wavelength Range End",
        "hsi_processing.processing": "Processing",
        "hsi_processing.crop": "Crop",
        "hsi_processing.top": "Top",
        "hsi_processing.bottom": "Bottom",
        "hsi_processing.left": "Left",
        "hsi_processing.right": "Right",
        "hsi_processing.rotate": "Rotate",
        "hsi_processing.rotate_degree": "Rotate Degree",
        "hsi_processing.preview": "Preview",
        "hsi_processing.apply_processing": "Apply Processing Effects",
        "hsi_processing.mat_data_type": "MAT Data Type",
        "hsi_processing.mat_format": "MAT Image Shape Format",
        "hsi_processing.mat_key": "Key of MAT file",
        "hsi_processing.compress_mat": "Produce Compressed MAT File",
        "hsi_processing.spectral_selection": "Spectral Selection",
        "hsi_processing.spectral_selection_help": "Click on the image to select pixels for spectral data extraction. The selected pixels will be plotted in the spectral plot below.",
        "hsi_processing.spectral_plot": "Spectral Plot",
        "hsi_processing.style": "Style",
        "hsi_processing.clear": "Clear",
        "hsi_processing.download": "Download",
        "hsi_processing.output_results": "Output Results",
        "hsi_processing.mat_file": "MAT File",
        "hsi_processing.info": "Info",
        "hsi_processing.same_as_input": "Same",
        "hsi_processing.auto_detect": "Auto Detect",

        "scanner_calc.tab_title": "Scanner Parameters"
    },
    'zh-CN' : {
        "about.tab_title": "关于",
        "about.title": "HPT高光谱处理工具箱",
        "about.description": "A Hyperspectral Image Preprocessing Toolkit from HSI Camera to Machine Learning Dataset",
        "about.homepage": "主页",
        "hsi_processing.tab_title": "高光谱图像处理",
        "hsi_processing.current_layer": "选中图层",
        "hsi_processing.load": "加载",
        "hsi_processing.upload_instructions": "**应上传以下两种格式中的一种**\n1. 同时上传一个.hdr文件 + 一个无后缀的数据文件\n2. 一个.mat文件",
        "hsi_processing.input_format": "输入数据形状",
        "hsi_processing.data_files": "数据文件",
        "hsi_processing.manual_normalize": "手动归一化(仅影响预览结果)",
        "hsi_processing.normalize_min": "归一化最小值",
        "hsi_processing.normalize_max": "归一化最大值",
        "hsi_processing.wavelength_start": "波长范围起始",
        "hsi_processing.wavelength_end": "波长范围结束",
        "hsi_processing.processing": "处理",
        "hsi_processing.crop": "裁切",
        "hsi_processing.top": "上",
        "hsi_processing.bottom": "下",
        "hsi_processing.left": "左",
        "hsi_processing.right": "右",
        "hsi_processing.rotate": "旋转",
        "hsi_processing.rotate_degree": "旋转角度",
        "hsi_processing.preview": "预览",
        "hsi_processing.apply_processing": "应用处理效果",
        "hsi_processing.mat_data_type": "mat文件数据类型",
        "hsi_processing.mat_format": "mat文件格式",
        "hsi_processing.mat_key": "mat文件的key",
        "hsi_processing.compress_mat": "启用mat文件压缩",
        "hsi_processing.spectral_selection": "光谱选择",
        "hsi_processing.spectral_selection_help": "点击预览图像图像中的像素进行光谱数据提取。选中的像素将在下方的光谱图中绘制。",
        "hsi_processing.spectral_plot": "光谱图",
        "hsi_processing.style": "样式",
        "hsi_processing.clear": "清空",
        "hsi_processing.download": "下载",
        "hsi_processing.output_results": "输出结果",
        "hsi_processing.mat_file": "MAT文件",
        "hsi_processing.info": "信息",
        "hsi_processing.same_as_input": "与输入相同",
        "hsi_processing.auto_detect": "自动检测",

        "scanner_calc.tab_title": "推扫仪计算"
    }
})

__all__ = ['i18n', 'CONSTS']