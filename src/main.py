from pathlib import Path
import shutil
from typing import Tuple, List
import einops
import numpy as np
import matplotlib.pyplot as plt
from rasterio import open as rasterio_open
from scipy.io import savemat, loadmat
import gradio as gr
import gradio.utils
from scipy.ndimage import rotate
from rs_fusion_datasets.util.hsi2rgb import _hsi2rgb, hsi2rgb
from jaxtyping import Float
from enum import Enum
from scanner_calc import scanner_calc_tab


plt.rcParams['font.family'] = 'SimHei'

i18n = gr.I18n(**{
    'en': {
        "title": "hsi-preprocessing-toolkit",
        "load": "Load",
        "upload_instructions": "**Upload one of the following formats:**\n1. One .hdr file + one raw data file without extension\n2. One .mat file",
        "input_format": "Input Image Shape Format",
        "data_files": "Data Files",
        "manual_normalize": "Manual Normalize (affects preview only)",
        "normalize_min": "Normalize Min",
        "normalize_max": "Normalize Max",
        "wavelength_start": "Wavelength Range Start",
        "wavelength_end": "Wavelength Range End",
        "processing": "Processing",
        "crop": "Crop",
        "top": "Top",
        "bottom": "Bottom",
        "left": "Left",
        "right": "Right",
        "rotate": "Rotate",
        "rotate_degree": "Rotate Degree",
        "preview": "Preview",
        "apply_processing": "Apply Processing Effects",
        "mat_data_type": "MAT Data Type",
        "mat_format": "MAT Image Shape Format",
        "mat_key": "Key of MAT file",
        "compress_mat": "Produce Compressed MAT File",
        "spectral_selection": "Spectral Selection",
        "spectral_selection_help": "Click on the image to select pixels for spectral data extraction. The selected pixels will be plotted in the spectral plot below.",
        "spectral_plot": "Spectral Plot",
        "style": "Style",
        "clear": "Clear",
        "download": "Download",
        "output_results": "Output Results",
        "mat_file": "MAT File",
        "info": "Info",
        "same_as_input": "Same",
        "auto_detect": "Auto Detect",
    },
    'zh-CN':{
        "title": "高光谱图像预处理工具箱",
        "load": "加载",
        "upload_instructions": "**应上传以下两种格式中的一种**\n1. 同时上传一个.hdr文件 + 一个无后缀的数据文件\n2. 一个.mat文件",
        "input_format": "输入数据形状",
        "data_files": "数据文件",
        "manual_normalize": "手动归一化(仅影响预览结果)",
        "normalize_min": "归一化最小值",
        "normalize_max": "归一化最大值",
        "wavelength_start": "波长范围起始",
        "wavelength_end": "波长范围结束",
        "processing": "处理",
        "crop": "裁切",
        "top": "上",
        "bottom": "下",
        "left": "左",
        "right": "右",
        "rotate": "旋转",
        "rotate_degree": "旋转角度",
        "preview": "预览",
        "apply_processing": "应用处理效果",
        "mat_data_type": "mat文件数据类型",
        "mat_format": "mat文件格式",
        "mat_key": "mat文件的key",
        "compress_mat": "启用mat文件压缩",
        "spectral_selection": "光谱选择",
        "spectral_selection_help": "点击预览图像图像中的像素进行光谱数据提取。选中的像素将在下方的光谱图中绘制。",
        "spectral_plot": "光谱图",
        "style": "样式",
        "clear": "清空",
        "download": "下载",
        "output_results": "输出结果",
        "mat_file": "MAT文件",
        "info": "信息",
        "same_as_input": "与输入相同",
        "auto_detect": "自动检测",
    }
})

class AppState(Enum):
    NOT_LOADED = 0
    LOADED = 2
    PREVIEWED = 3
    COVERTED = 4

# Void -> HWC
def load_data(
    dat_files :List[gr.utils.NamedString] | None,
    input_format, mat_key: str | None = None
) -> Tuple[np.ndarray | None, Path | None]:
    if dat_files is None or len(dat_files) == 0:
        raise gr.Error("No data file provided. 请上传数据文件. Please upload a data file.")

    dat_paths = [Path(f.name) for f in dat_files]
    mat_paths = [p for p in dat_paths if p.suffix == '.mat']
    hdr_paths = [p for p in dat_paths if p.suffix == '.hdr']
    raw_paths = [p for p in dat_paths if p.suffix == '']

    if len(mat_paths) > 0:
        data_path = mat_paths[0]
        mat_dat = loadmat(mat_paths[0], squeeze_me=True, mat_dtype=True, struct_as_record=False)
        mat_keys = [x for x in mat_dat.keys() if not x.startswith('__') and not x.endswith('__')]
        if mat_key is None or mat_key == '':
            mat_key = mat_keys[0]
            if len(mat_keys) >= 2:
                gr.Warning(f"Multiple keys found: {mat_keys}. Using: {mat_key}")
        else:
            if mat_key not in mat_dat:
                raise gr.Error(f"Key '{mat_key}' not found in the MAT file. Available keys: {mat_keys}")
        data = mat_dat[mat_key]
    
    elif len(hdr_paths) == 0 or len(raw_paths) == 0:
        raise gr.Error("Both .hdr and raw data files are required. Only one is provided.")
    elif len(hdr_paths) > 0 and len(raw_paths) > 0:
        data_path = raw_paths[0]
        hdr_path = hdr_paths[0]
        if hdr_path.parent != data_path.parent:
            shutil.copy(hdr_path, data_path.with_suffix('.hdr'))
        print(f"Loading {data_path}")
        with rasterio_open(data_path, 'r') as src:
            data = src.read()
        print(f"Loaded {data_path}")
    else:
        raise gr.Error("Unknown file format")

    if data is None:
        raise gr.Error("Data loading failed.")
    elif len(data.shape) != 3:
        raise gr.Error(f"Data shape {data.shape} is not valid. Expected 3D array.")

    if input_format == 'CHW':
        data = einops.rearrange(data, 'c h w -> h w c')

    return data, data_path


# HWC -> HWC
def process_img(img, crop_top, crop_left, crop_bottom, crop_right, rotate_deg):
    # Rotate
    if rotate_deg % 360 != 0:
        img = rotate(img, angle=rotate_deg, axes=(0, 1), reshape=True)

    # Crop
    if crop_top > 0 or crop_left > 0 or crop_bottom > 0 or crop_right > 0:
        crop_bottom = None if crop_bottom == 0 else -crop_bottom
        crop_right  = None if crop_right == 0  else -crop_right
        img = img[crop_top:crop_bottom, crop_left:crop_right, :]

    return img

# def update_ui(state_app_state: AppState):
#     gr.Info(f"App state changed to {state_app_state.name}")
    
#     preview_visible = state_app_state in [AppState.LOADED, AppState.PREVIEWED, AppState.COVERTED]
#     convert_visible = state_app_state in [AppState.PREVIEWED, AppState.COVERTED]
#     plot_visible = state_app_state in [AppState.COVERTED]
    
#     return (
#         gr.update(visible=preview_visible),
#         gr.update(visible=convert_visible),
#         gr.update(visible=plot_visible)
#     )

def gr_load(
        dat_files :List[gradio.utils.NamedString] | None,
        input_format, input_mat_key: str | None,
        manual_normalize: bool, normalize_min: float, normalize_max: float,
        wavelength_from: int, wavelength_to: int,
    ):
    data, data_path = load_data(dat_files, input_format, input_mat_key)
    if data is None:
        raise gr.Error("No data file provided or data loading failed.")
    gr.Info("Loading data...")
    if not manual_normalize:
        rgb = hsi2rgb(data, wavelength_range=(wavelength_from, wavelength_to), input_format='HWC', output_format='HWC', to_u8np=True)
    else:
        rgb = _hsi2rgb( (data-normalize_min)/(normalize_max-normalize_min), wavelength=np.linspace(wavelength_from, wavelength_to, data.shape[-1]))
        rgb = (rgb*255.0).astype(np.uint8)
    gr.Success(f"Data loaded")
    logging_text = str({
        "original_shape": str(data.shape),
        "original_data_type": str(data.dtype),
        "original_reflection_range": [float(data.min()), float(data.max())],
        "original_reflection_mean": float(data.mean()),
        "original_reflection_std": float(data.std()),
    })
    return rgb, data, data_path, AppState.LOADED, logging_text

def gr_preview(
        state_rgb :Float[np.ndarray, 'h w c'] | None,
        crop_top: int, crop_left: int, crop_bottom: int, crop_right: int, 
        rotate_deg: int
    ):
    if state_rgb is None:
        raise gr.Error("No data provided.")
    return process_img(
        state_rgb, crop_top, crop_left, crop_bottom, crop_right, rotate_deg
    ), AppState.PREVIEWED

def gr_download_selected_spectral(data, state_select_location, data_path):
    if not state_select_location or len(state_select_location) == 0:
        raise gr.Error("No spectral data selected. Please select at least one pixel to download")
    
    gr.Info("Converting ...")

    dat_path = Path(data_path.name)
    result = []
    result_location = []
    for (row, col) in state_select_location:
        result.append(data[row, col,:])
        result_location.append([row, col]) 
    
    result = np.array(result)
    result_location = np.array(result_location)

    mat_path = dat_path.with_stem(dat_path.stem + '_selected_spectral').with_suffix('.mat')
    savemat(mat_path, {'data': result, 'location': result_location}, do_compression=True, format='5')

    gr.Success("Done ...")
    
    return str(mat_path)

def gr_convert(
        data, data_path,
        crop_top: int, crop_left: int, crop_bottom: int, crop_right: int, 
        rotate_deg: int,
        mat_dtype, output_format, mat_key, compress_mat: bool,
        logging_text: str
    ):
    if output_format == 'same':
        output_format = input_format
    
    data = process_img(
        data, crop_top, crop_left, crop_bottom, crop_right, rotate_deg
    )
    
    # Convert to mat
    mat_file = data_path.with_stem(data_path.stem + '_converted').with_suffix('.mat')
    if mat_dtype == "default":
        data_processed = data
    else:
        data_processed = data.astype(mat_dtype)

    if output_format == 'CHW':
        mat_dat_sav = einops.rearrange(data_processed, 'h w c -> c h w')
    else:
        mat_dat_sav = data_processed
    savemat(mat_file, {mat_key: mat_dat_sav}, do_compression=compress_mat, format='5')  
    
    info = {
        'original_shape': str(data.shape),
        'original_data_type': str(data.dtype),
        'original_reflection_range': [float(data.min()), float(data.max())],
        'original_reflection_mean': float(data.mean()),
        'original_reflection_std': float(data.std()),
        'output_shape': str(mat_dat_sav.shape),
        'output_data_type': str(mat_dat_sav.dtype),
        'output_reflection_range': [float(mat_dat_sav.min()), float(mat_dat_sav.max())],
        'output_reflection_mean': float(mat_dat_sav.mean()),
        'output_reflection_std': float(mat_dat_sav.std()),
    }

    logging_text += "\n".join([f"{k}: {v}" for k, v in info.items()]) + "\n"

    # Gradio will handle the visibility of mat_file_output when a file path is returned
    return data_processed, str(mat_file), logging_text, AppState.COVERTED


def gr_on_img_clicked(evt: gr.SelectData, state_figure :tuple, data, state_select_location, wavelength_from: int, wavelength_to :int, plot_hint: str):
    """绘制选中像素的光谱曲线"""
    if data is None:
        raise gr.Error("No converted data provided")
    if state_select_location is None:
        state_select_location = []
    col, row = evt.index
    state_select_location.append((row, col)) # WHY?

    wavelengths = np.linspace(wavelength_from, wavelength_to, data.shape[-1])
    # spectral_plot = pd.DataFrame(
    #     {
    #         "Wavelength": wavelengths,
    #         "Reflectance": data[row, col, :]
    #     }
    # ) # typecheck: ignore

    # fig, ax = plt.subplots(figsize=(10, 6))
    if state_figure is None:
        state_figure = plt.subplots(figsize=(10, 6))
    fig, ax = state_figure
    # for (row, col) in state_select_location:
    #     ax.plot(wavelengths, data[row, col, :], plot_hint ,linewidth=2)
    ax.plot(wavelengths, data[row, col, :], plot_hint ,linewidth=2)
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Reflectance')
    plt.tight_layout()
    state_figure = (fig, ax)
    return fig, state_figure, state_select_location


if __name__ == "__main__":
    theme = gr.themes.Default(primary_hue='cyan').set(
        button_primary_background_fill='#39c5bb',
        button_primary_background_fill_hover="#30A8A0",
    )

    with gr.Blocks(title="hsi-preprocessing-toolkit", theme=theme) as demo:
        state_app_state = gr.State(value=AppState.NOT_LOADED)
        state_original_data = gr.State(value=None)
        state_processed_data = gr.State(value=None)
        state_data_path = gr.State(value=None,)
        state_selected_location = gr.State(value=[])
        state_original_rgb = gr.State(value=None)
        state_spectral_figure = gr.State(value=None)
        
        with gr.Tab('scanner') as tab:
            scanner_calc_tab()

        with gr.Tab(i18n("title"), id="hsi_preprocessing_toolkit"):
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(i18n("load")) as load_panel:
                        gr.Markdown(i18n("upload_instructions"))
                        with gr.Row():
                            input_format = gr.Radio(
                                label=i18n("input_format"),
                                choices=['HWC', 'CHW'],
                                value='CHW'
                            )
                            input_mat_key = gr.Textbox(
                                label=i18n("mat_key"),
                                value=None,
                                placeholder=i18n("auto_detect"),
                                visible=True,
                            )
                        dat_files = gr.File(
                            label=i18n("data_files"),
                            file_count="multiple",
                            type="filepath",
                        )
                        manual_normalize = gr.Checkbox(
                            label=i18n("manual_normalize"),
                            value=False,
                        )
                        with gr.Row():
                            normalize_min = gr.Number(
                                label=i18n("normalize_min"),
                                value=0,
                                precision=1,
                                visible=False,
                            )
                            normalize_max = gr.Number(
                                label=i18n("normalize_max"),
                                value=2**16-1,
                                precision=1,
                                visible=False,
                            )

                        def toggle_normalize_fields(manual_normalize):
                            return (
                                gr.update(visible=manual_normalize),
                                gr.update(visible=manual_normalize),
                            )
                        manual_normalize.change(
                            fn=toggle_normalize_fields,
                            inputs=[manual_normalize],
                            outputs=[normalize_min, normalize_max],
                        )
                        with gr.Row():
                            wavelength_from = gr.Number(
                                label=i18n("wavelength_start"),
                                value=400,
                                precision=1,
                            )
                            wavelength_to = gr.Number(
                                label=i18n("wavelength_end"),
                                value=1000,
                                precision=1,
                            )
                        reload_btn = gr.Button(i18n("load"), variant="primary")
                    
                    with gr.Accordion(i18n("processing"), visible=False) as preview_panel:
                        with gr.Column():
                            gr.Markdown(f"### {i18n('crop')}")
                            with gr.Row():
                                crop_top = gr.Slider(
                                    label=i18n("top"),
                                    minimum=0,
                                    maximum=0,
                                    step=1,
                                )
                                crop_bottom = gr.Slider(
                                    label=i18n("bottom"),
                                    minimum=0,
                                    maximum=0,
                                    step=1,
                                )
                            with gr.Row():
                                crop_left = gr.Slider(
                                    label=i18n("left"),
                                    minimum=0,
                                    maximum=0,
                                    step=1,
                                )
                                crop_right = gr.Slider(
                                    label=i18n("right"),
                                    minimum=0,
                                    maximum=0,
                                    step=1,
                                )

                        with gr.Column():
                            gr.Markdown(f"### {i18n('rotate')}")
                            rotate_deg = gr.Slider(
                                label=i18n("rotate_degree"),
                                minimum=-360,
                                maximum=360,
                                step=1,
                                value=0
                            )
                        # 这个按钮因为实时预览实际上不需要了，但是暂且隐藏备用
                        preview_btn  = gr.Button(i18n("preview"), variant="primary", visible=False)
                        

                    with gr.Accordion(i18n("apply_processing"), visible=False) as convert_panel:
                        mat_dtype = gr.Radio(
                            label=i18n("mat_data_type"),
                            choices=["same", "uint8", "uint16", "float32"],
                            value="float32"
                        )
                        output_format = gr.Radio(
                            label=i18n("mat_format"),
                            choices=[
                                'same', 
                                'HWC', 
                                'CHW'
                            ],
                            value='same'
                        )
                        mat_key = gr.Text(
                            label=i18n("mat_key"),
                            value='data',
                        )
                        compress_mat = gr.Checkbox(
                            label=i18n("compress_mat"),
                            value=True
                        )
                        convert_btn  = gr.Button(i18n("apply_processing"), variant="primary")
                    
                    with gr.Accordion(i18n("spectral_selection"),visible=False) as plot_panel:
                        gr.Markdown(i18n('spectral_selection_help'))
                        spectral_plot = gr.Plot(
                            label=i18n("spectral_plot"),
                            visible=True,
                            # x="Wavelength", y="Reflectance",
                            # height=400, width=600,
                        )
                        plot_hint = gr.Textbox(
                            label=i18n("style"),
                            value='b-',
                        )
                        with gr.Row():
                            clear_plot_btn = gr.Button(
                                i18n("clear"),
                                variant="secondary",
                            )
                            download_select_spectral = gr.DownloadButton(
                                "Download",
                                variant="primary",
                            )

                with gr.Column():
                    with gr.Column(variant="panel"):
                        gr.Markdown(f"## {i18n('output_results')}")   
                        preview_img = gr.Image(label=i18n("preview"), format="png", height="auto", width="auto", interactive=False)
                        mat_file_output = gr.File(
                            label=i18n("mat_file"),
                            type="filepath",
                            interactive=False
                        )
                        logging_text = gr.Textbox(
                            label=i18n("info"),
                        )
                    

            # dat_files.upload(
            #     fn=gr_on_file_upload,
            #     inputs=[dat_files, input_format, manual_normalize, normalize_min, normalize_max, wavelength_from, wavelength_to],
            #     outputs=[state_original_rgb, state_original_data, state_data_path, state_app_state]
            # )
            reload_btn.click(
                fn=gr_load,
                inputs=[dat_files, input_format, input_mat_key, manual_normalize, normalize_min, normalize_max, wavelength_from, wavelength_to],
                outputs=[state_original_rgb, state_original_data, state_data_path, state_app_state, logging_text]
            )
            convert_btn.click(
                fn=gr_convert,
                inputs=[
                    state_original_data, state_data_path,
                    crop_top, crop_left, crop_bottom, crop_right,
                    rotate_deg,
                    mat_dtype, output_format, mat_key, compress_mat,
                    logging_text
                ],
                outputs=[state_processed_data ,mat_file_output, logging_text, state_app_state]
            )
            for component in [crop_top, crop_left, crop_bottom, crop_right, rotate_deg]:
                component.change(
                    fn=gr_preview,
                    inputs=[state_original_rgb, crop_top, crop_left, crop_bottom, crop_right, rotate_deg],
                    outputs=[preview_img, state_app_state]
                )
            
            preview_btn.click(
                fn=gr_preview,
                inputs=[
                    state_original_rgb,
                    crop_top, crop_left, crop_bottom, crop_right,
                    rotate_deg,
                ],
                outputs=[preview_img, state_app_state]
            )
            preview_img.select(
                fn=gr_on_img_clicked,
                inputs=[state_spectral_figure, state_processed_data, state_selected_location, wavelength_from, wavelength_to, plot_hint],
                outputs=[spectral_plot, state_spectral_figure, state_selected_location]
            )
            clear_plot_btn.click(
                fn=lambda: (None, None, []),
                outputs=[spectral_plot, state_spectral_figure, state_selected_location]
            )
            download_select_spectral.click(
                fn=gr_download_selected_spectral,
                inputs=[state_processed_data, state_selected_location, state_data_path],
                outputs=[download_select_spectral]
            )
            state_original_data.change(
                fn=lambda x: (
                    gr.update(maximum=x.shape[1] if x is not None else 0),
                    gr.update(maximum=x.shape[1] if x is not None else 0),
                    gr.update(maximum=x.shape[0] if x is not None else 0),
                    gr.update(maximum=x.shape[0] if x is not None else 0),
                ),
                inputs=[state_original_data],
                outputs=[crop_left, crop_right, crop_top, crop_bottom]
            )
            state_original_rgb.change(
                fn=lambda x:x,
                inputs=[state_original_rgb],
                outputs=[preview_img]
            )
            state_app_state.change(
                fn=lambda x: (
                    gr.update(visible=True, open=(x==AppState.NOT_LOADED)),
                    gr.update(visible=x!=AppState.NOT_LOADED, open=(x in [AppState.LOADED, AppState.PREVIEWED])),
                    gr.update(visible=x!=AppState.NOT_LOADED, open=(x in [AppState.LOADED, AppState.PREVIEWED])),
                    gr.update(visible=x==AppState.COVERTED,   open=(x==AppState.COVERTED))
                ),
                inputs=[state_app_state],
                outputs=[load_panel, preview_panel, convert_panel, plot_panel]
            )

    demo.launch(share=False, inbrowser=True, i18n=i18n)

