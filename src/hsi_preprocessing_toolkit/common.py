import gradio as gr
import logging
import logging.handlers
import os
import importlib.metadata
import platform
import secrets
import argparse

# 全局信息
APP_NAME='hsi-preprocessing-toolkit'
APP_VERSION = importlib.metadata.version(APP_NAME)

# 全局参数
def get_args():
    parser = argparse.ArgumentParser(APP_NAME)
    parser.add_argument('--debug',           default=False,      action='store_true',                   help='start in debug mode')
    parser.add_argument('--mode',            default='window',   choices=['window','browser','server'], help='start in a window, browser or as a server')
    parser.add_argument('--no-access-token', default=False,      action='store_true',                   help='do not use access token for security')
    parser.add_argument('--host',            default='localhost',                                       help='bind address')
    parser.add_argument('--port',            default=None,                                              help='bind port')
    arg = parser.parse_args()
    return arg
ARGS = get_args()


# I18N
from importlib import resources
from string import Template
def _load_about_md(lang :str):
    if lang.startswith('zh'):
        lang = 'zh'
    else:
        lang = 'en'
    text = resources.files("hsi_preprocessing_toolkit.asset.page").joinpath(f"about.{lang}.md").read_text(encoding="utf-8")
    return Template(text).safe_substitute(
        VERSION=APP_VERSION
    )

TRANSLATION = {
    'en': {
        "about.tab_title": "About",
        "about.title": "HSI Preprocessing Toolkit",
        "about.content": _load_about_md('en'),

        "hsi_processing.tab_title": "HSI Processing",
        "hsi_processing.load": "Load",
        "hsi_processing.loading": "Loading, this could take minutes...",
        "hsi_processing.loaded": "Loaded",
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
        "hsi_processing.translate_offset": "Traslate",
        "hsi_processing.translate_x": "X",
        "hsi_processing.translate_y": "Y",
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
        "hsi_processing.applying_transforms": "Applying Transforms...",
        "hsi_processing.applied_transforms": "Transforms Done",
        "hsi_processing.no_converted_data_for_clicking": "No transforms applied data, apply transforms first",
        "hsi_processing.spectral_profiles": "Spectral Profiles",
        "hsi_processing.spectral_profiles.generate": "Generate Spectral Profiles",
        "hsi_processing.spectral_profiles.x_lambda_plane": "x-λ",
        "hsi_processing.spectral_profiles.y_lambda_plane": "y-λ",
        "hsi_processing.spectral_profiles.hsi_cube": "HSI Cube",


        "scanner_calc.tab_title": "Scanner Parameters",
    },
    'zh-CN' : {
        "about.tab_title": "关于",
        "about.title": "HPT高光谱处理工具箱",
        "about.content": _load_about_md('zh'),

        "hsi_processing.tab_title": "高光谱图像处理",
        "hsi_processing.current_layer": "选中图层",
        "hsi_processing.load": "加载",
        "hsi_processing.loading": "加载中,这可能需要几分钟...",
        "hsi_processing.loaded": "加载完成",
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
        "hsi_processing.translate_offset": "平移",
        "hsi_processing.translate_x": "X轴",
        "hsi_processing.translate_y": "Y轴",
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
        "hsi_processing.applying_transforms": "应用变换中...",
        "hsi_processing.applied_transforms": "变换完成",
        "hsi_processing.no_converted_data_for_clicking": "没有应用变换后的数据,请先应用变换",
        "hsi_processing.spectral_profiles": "光谱剖面",
        "hsi_processing.spectral_profiles.generate": "生成光谱剖面",
        "hsi_processing.spectral_profiles.x_lambda_plane": "x-λ剖面",
        "hsi_processing.spectral_profiles.y_lambda_plane": "y-λ剖面",
        "hsi_processing.spectral_profiles.hsi_cube": "光谱立方体",

        "scanner_calc.tab_title": "推扫仪计算",
    }
}
i18n = gr.I18n(**TRANSLATION)

# 日志
LOGGER = logging.getLogger(TRANSLATION['en']['about.title']) # 全局唯一LOGGER
LOGGER_MEMORY_HANDLER :logging.handlers.MemoryHandler|None = None
LOGGER_MEMORY_HANDLER = logging.handlers.MemoryHandler(10_000, flushLevel=logging.WARNING) # 用于在UI中显示LOGGING信息
LOGGER.addHandler(LOGGER_MEMORY_HANDLER)

logging.basicConfig(
    level=logging.DEBUG if ARGS.debug else logging.INFO,
    format='[%(levelname)s %(asctime)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

ACCESS_TOKEN = None if ARGS.no_access_token else secrets.token_hex(16) 


# 初始化完成
LOGGER.info(f"{APP_NAME} v{APP_VERSION} initalized. DEBUG={ARGS.debug} Gradio=v{gr.__version__} Python=v{platform.python_version()} OS={platform.platform()}")