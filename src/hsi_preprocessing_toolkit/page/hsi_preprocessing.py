from pathlib import Path
import shutil
from typing import Tuple, List
import einops
import numpy as np
import matplotlib.pyplot as plt
from rasterio import open as rasterio_open
import scipy.io
from scipy.io import savemat, loadmat
import gradio as gr
import gradio.utils
from rs_fusion_datasets.util.hsi2rgb import _hsi2rgb, hsi2rgb
from jaxtyping import Float
from enum import Enum
from ..algorithm import composite_img
from ..common import i18n, LOGGER, TRANSLATION
import logging

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
        LOGGER.info(f"Loading {data_path} as .mat files, {mat_key=}")
        mat_dat = loadmat(data_path, squeeze_me=True, mat_dtype=True, struct_as_record=False)
        mat_keys = [x for x in mat_dat.keys() if not x.startswith('__') and not x.endswith('__')]

        if not mat_keys:
            LOGGER.error("No keys found in .mat file")
            gr.Error("No keys found in .mat file")
        elif not mat_key:
            LOGGER.log(level=logging.INFO if len(mat_keys)==1 else logging.WARNING, msg = f"Auto-selected mat_key={mat_keys[0]} from {mat_keys}")
            data = mat_dat[mat_keys[0]]
        elif mat_key not in mat_dat.keys():
            LOGGER.error("mat_key not found in .mat files")
            gr.Error("mat_key not found in .mat files") # TODO: Maybe we can make gr.Error triggered when calling LOGGER.error by adding a handler?
        else:
            data = mat_dat[mat_key]
            
        # 处理mat_struct的情况
        if isinstance(data, scipy.io.matlab.mio5_params.mat_struct):
            # TODO: This should be more complex than this, we need mat_key and struct_key if user want, but gradio does not have messagebox
            LOGGER.info("mat_struct detected, processing...")
            if not data._fieldnames:
                gr.Error("_fieldnames not found for mat_struct")
            elif mat_key and mat_key in data._fieldnames: 
                data = getattr(data, mat_key)
            else:
                gr.Warning(f"Auto-selected struct_key={data._fieldnames[0]} from {data._fieldnames}")
                data = getattr(data, data._fieldnames[0])

    elif len(hdr_paths) == 0 or len(raw_paths) == 0:
        raise gr.Error("Both .hdr and raw data files are required. Only one is provided.")
    elif len(hdr_paths) > 0 and len(raw_paths) > 0:
        data_path = raw_paths[0]
        hdr_path = hdr_paths[0]
        if hdr_path.parent != data_path.parent:
            shutil.copy(hdr_path, data_path.with_suffix('.hdr'))
        LOGGER.info(f"Loading {data_path} as .hdr+raw files")
        with rasterio_open(data_path, 'r') as src:
            # LOGGER.info(src.profile)
            data = src.read()

    else:
        raise gr.Error("Unknown file format")

    if data is None:
        raise gr.Error("Data loading failed.")
    elif len(data.shape) != 3:
        raise gr.Error(f"Data shape {data.shape} is not valid. Expected 3D array.")

    if input_format == 'CHW':
        data = einops.rearrange(data, 'c h w -> h w c')

    return data, data_path


def gr_load(
        state_current_layer_index,
        state_transforms :list[dict], state_original_rgb, state_original_data, state_original_data_path,
        dat_files :List[gradio.utils.NamedString] | None,
        input_format, input_mat_key: str | None,
        manual_normalize: bool, normalize_min: float, normalize_max: float,
        wavelength_from: int, wavelength_to: int,
    ):
    LOGGER.info(f"gr_load {state_current_layer_index=} {len(state_original_rgb)=}")

    # dialog_title = ' '.join([Path(f.name).name for f in dat_files])
    gr.Info(TRANSLATION['en']["hsi_processing.loading"] + '\n' + TRANSLATION['zh-CN']["hsi_processing.loading"], duration=30)  # TODO: Wait the fix and support of i18n in gradio 6
    # gr.Info(i18n("hsi_processing.loading"), duration=30)
    data, data_path = load_data(dat_files, input_format, input_mat_key)
    if data is None:
        raise gr.Error("No data file provided or data loading failed.")

    if not manual_normalize:
        rgb = hsi2rgb(data, wavelength_range=(wavelength_from, wavelength_to), input_format='HWC', output_format='HWC', to_u8np=True)
    else:
        rgb = _hsi2rgb( (data-normalize_min)/(normalize_max-normalize_min), wavelength=np.linspace(wavelength_from, wavelength_to, data.shape[-1]))
        rgb = (rgb*255.0).astype(np.uint8)
    gr.Success(TRANSLATION['en']["hsi_processing.loaded"] + '\n' + TRANSLATION['zh-CN']["hsi_processing.loaded"], duration=30) # TODO: Wait the fix and support of i18n in gradio 6
    # gr.Success(i18n("hsi_processing.loaded"), duration=30)

    # print('\a') # This suppose that server and client are in the same PC

    LOGGER.info(str({
        "original_shape": str(data.shape),
        "original_data_type": str(data.dtype),
        "original_reflection_range": [float(data.min()), float(data.max())],
        "original_reflection_mean": float(data.mean()),
        "original_reflection_std": float(data.std()),
    }))

    # # FIXME: Do not use dirty fix
    while state_current_layer_index >= len(state_original_rgb):
        state_original_rgb.append(None)
        state_original_data.append(None)
        state_original_data_path.append(None)
        state_transforms.append(DEFAULT_TRANSFORM)

    state_original_rgb[state_current_layer_index]       = rgb
    state_original_data[state_current_layer_index]      = data
    state_original_data_path[state_current_layer_index] = data_path
    state_transforms[state_current_layer_index]         = DEFAULT_TRANSFORM

    LOGGER.info(f"gr_loaded {state_current_layer_index=} {len(state_original_rgb)=}")

    return state_transforms, state_original_rgb, state_original_data, state_original_data_path, AppState.LOADED

# def gr_transpose_original_data(input_format,state_transforms, state_current_layer_index, state_original_data, state_ui_state):
#     if input_format == "CHW":
#         pattern = 'c h w -> h w c'
#     else:
#         pattern = 'h w c -> c h w'
#     state_original_data = einops.rearrange(state_original_data, pattern)
#     state_transforms[state_current_layer_index] = DEFAULT_TRANSFORM
#     state_original_rgb[state_current_layer_index] = 
#     return state_transforms, state_original_rgb, state_original_data, AppState.LOADED

def gr_composite(
        state_original_rgb :Float[np.ndarray, 'h w c'] | None,
        state_transforms,
    ):
    LOGGER.info("compositing...")
    img = composite_img(state_original_rgb, state_transforms)
    LOGGER.info(f"composited {img.shape=} {type(img)=}")
    return img, AppState.PREVIEWED


def gr_download_selected_spectral(state_current_layer_index, data, state_select_location, data_path):
    if not state_select_location or len(state_select_location) == 0:
        raise gr.Error("No spectral data selected. Please select at least one pixel to download")
    
    gr.Info(i18n("hsi_processing.applying_transforms"))

    dat_path = Path(data_path[state_current_layer_index].name)
    result = []
    result_location = []
    for (row, col) in state_select_location:
        result.append(data[row, col,:])
        result_location.append([row, col]) 
    
    result = np.array(result)
    result_location = np.array(result_location)

    mat_path = dat_path.with_stem(dat_path.stem + '_selected_spectral').with_suffix('.mat')
    savemat(mat_path, {'data': result, 'location': result_location}, do_compression=True, format='5')

    gr.Success("hsi_processing.applied_transforms")
    
    return str(mat_path)

def gr_convert(
        state_current_layer_index,
        state_transforms,
        data, data_path,
        mat_dtype, output_format, mat_key, compress_mat: bool,
    ):
    LOGGER.info("gr_convert")
    
    data_path = data_path[state_current_layer_index]
    data = composite_img(
        data, state_transforms
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

    LOGGER.info( "\n".join([f"{k}: {v}" for k, v in info.items()]) )

    # Gradio will handle the visibility of mat_file_output when a file path is returned
    return data_processed, str(mat_file), AppState.COVERTED


def gr_on_img_clicked(evt: gr.SelectData, state_figure :tuple, data, state_select_location, wavelength_from: int, wavelength_to :int, plot_hint: str):
    """绘制选中像素的光谱曲线"""
    if data is None:
        gr.Error(TRANSLATION['en']["hsi_processing.no_converted_data_for_clicking"] + '\n' + TRANSLATION['zh-CN']["hsi_processing.no_converted_data_for_clicking"]) 
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

def gr_update_transforms(state_transforms, state_current_layer_index, crop_top, crop_left, crop_bottom, crop_right, rotate_deg, offset_x, offset_y):
    state_transforms[state_current_layer_index] = {
        'rotation': rotate_deg,
        'crop': [crop_top, crop_left, crop_bottom, crop_right],
        'location': [offset_x, offset_y]
    }
    return state_transforms

def gr_on_state_current_layer_index_changed(state_current_layer_index, state_transforms, state_original_data):
    if state_current_layer_index >= len(state_transforms):
        trans = DEFAULT_TRANSFORM
        shape = [0,0]
        ui_state = AppState.NOT_LOADED
        dat_files = gr.update(value=[])
    else:
        trans = state_transforms[state_current_layer_index]
        shape = state_original_data[state_current_layer_index].shape
        ui_state = AppState.LOADED
        dat_files = gr.skip()

    crop_top, crop_left, crop_bottom, crop_right = trans['crop']
    rotate_deg = trans['rotation']
    offset_x, offset_y = trans['location']
    
    max_h, max_w = shape[:2]
    LOGGER.info(f"layer_index_changed {state_current_layer_index=} {max_h=}, {max_w=}")
    update_crop_top = gr.update(value=crop_top, maximum=max_h, minimum=0)
    update_crop_bottom = gr.update(value=crop_bottom, maximum=max_h, minimum=0)
    update_crop_left = gr.update(value=crop_left, maximum=max_w, minimum=0)
    update_crop_right = gr.update(value=crop_bottom, maximum=max_w, minimum=0)

    return update_crop_top, update_crop_left, update_crop_bottom, update_crop_right, rotate_deg, offset_x, offset_y, ui_state, dat_files

def gr_on_state_data_path_changed(state_data_path, state_current_layer_index, current_layer_radio):
    choices = [f"{i} "+(p.name if p else "Empty") for i,p in enumerate(state_data_path+[None])]
    LOGGER.info(f"state_data_path_changed. {choices=}")
    # 更新UI
    current_layer_radio = gr.update( value=choices[state_current_layer_index], choices=choices )
    return current_layer_radio


DEFAULT_TRANSFORM = {
    'rotation': 0,
    'crop': [0,0,0,0],
    'location': [0,0]
}

def HSIProcessingTab():
    with gr.Tab(i18n("hsi_processing.tab_title"), id="hsi_preprocessing_toolkit"):
        # 应用整体状态
        state_ui_state           = gr.State(value=AppState.NOT_LOADED)
        # 输入的状态，支持多输出
        state_current_layer_index = gr.State(value=0)      # 已选中的图层的数组index
        state_data_path           = gr.State(value=[])     # 原数据文件路径
        state_original_data       = gr.State(value=[])     # 原数据
        state_original_rgb        = gr.State(value=[])     # 原数据的RGB代理
        state_transforms          = gr.State(value=[])     # 对数据的变换
        # 输出内容状态，保持单个
        state_processed_data      = gr.State(value=None)   # 处理后的数据
        state_selected_location   = gr.State(value=[])     # 已选择的光谱XY坐标点 
        state_spectral_figure     = gr.State(value=None)   # 已选择的光谱的绘图 

        with gr.Row():
            with gr.Column():
                with gr.Column():
                    # current_layer_index_slider = gr.Slider(
                    #     label=i18n("hsi_processing.current_layer"),
                    #     minimum=0,
                    #     maximum=3,
                    #     step=1,
                    #     value=0
                    # )
                    current_layer_radio = gr.Radio(
                        label=i18n("hsi_processing.current_layer"),
                        choices=["0 Empty",],
                        value="0 Empty",
                        type='index'
                    )
                    # with gr.Row(variant="compact"):
                    #     @gr.render(inputs=[current_layer_index_slider, state_data_path])
                    #     def render_layer_selector(current_layer, state_data_path):
                    #         for data_path in state_ui_state:
                    #             radio_button = gr.Radio()


                with gr.Accordion(i18n("hsi_processing.load")) as load_panel:
                    gr.Markdown(i18n("hsi_processing.upload_instructions"))
                    with gr.Row():
                        input_format = gr.Radio(
                            label=i18n("hsi_processing.input_format"),
                            choices=['HWC', 'CHW'],
                            value='CHW'
                        )
                        input_mat_key = gr.Textbox(
                            label=i18n("hsi_processing.mat_key"),
                            value=None,
                            placeholder=i18n("hsi_processing.auto_detect"),
                            visible=True,
                        )
                    dat_files = gr.File(
                        label=i18n("hsi_processing.data_files"),
                        file_count="multiple",
                        type="filepath"
                    )
                    manual_normalize = gr.Checkbox(
                        label=i18n("hsi_processing.manual_normalize"),
                        value=False,
                    )
                    with gr.Row():
                        normalize_min = gr.Number(
                            label=i18n("hsi_processing.normalize_min"),
                            value=0,
                            precision=1,
                            visible=False,
                        )
                        normalize_max = gr.Number(
                            label=i18n("hsi_processing.normalize_max"),
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
                            label=i18n("hsi_processing.wavelength_start"),
                            value=400,
                            precision=1,
                        )
                        wavelength_to = gr.Number(
                            label=i18n("hsi_processing.wavelength_end"),
                            value=1000,
                            precision=1,
                        )
                    with gr.Row():
                        reload_btn = gr.Button(i18n("hsi_processing.load"), variant="primary")
                        reset_loaded_btn = gr.Button(i18n("hsi_processing.clear"), variant="secondary")
                # TODO: set maxium to be shape[i]
                with gr.Accordion(i18n("hsi_processing.processing"), visible=False) as preview_panel:
                    with gr.Column():
                        gr.Markdown(f"### {i18n('hsi_processing.crop')}")
                        with gr.Row():
                            crop_top = gr.Slider(
                                label=i18n("hsi_processing.top"),
                                minimum=0,
                                maximum=9999,
                                step=1,
                            )
                            crop_bottom = gr.Slider(
                                label=i18n("hsi_processing.bottom"),
                                minimum=0,
                                maximum=9999,
                                step=1,
                            )
                        with gr.Row():
                            crop_left = gr.Slider(
                                label=i18n("hsi_processing.left"),
                                minimum=0,
                                maximum=9999,
                                step=1,
                            )
                            crop_right = gr.Slider(
                                label=i18n("hsi_processing.right"),
                                minimum=0,
                                maximum=9999,
                                step=1,
                            )

                    with gr.Column():
                        gr.Markdown(f"### {i18n('hsi_processing.rotate')}")
                        rotate_deg = gr.Slider(
                            label=i18n("hsi_processing.rotate_degree"),
                            minimum=-360,
                            maximum=360,
                            step=1,
                            value=0
                        )
                    
                    with gr.Column():
                        gr.Markdown(f"### {i18n('hsi_processing.translate_offset')}")
                        with gr.Row():
                            offset_x = gr.Slider(
                                label=i18n("hsi_processing.translate_x"),
                                minimum=0,
                                maximum=+9999,
                                step=1,
                            )
                            offset_y = gr.Slider(
                                label=i18n("hsi_processing.translate_y"),
                                minimum=0,
                                maximum=+9999,
                                step=1,
                            )

                with gr.Accordion(i18n("hsi_processing.apply_processing"), visible=False) as convert_panel:
                    mat_dtype = gr.Radio(
                        label=i18n("hsi_processing.mat_data_type"),
                        choices=["auto", "uint8", "uint16", "float32"],
                        value="float32"
                    )
                    output_format = gr.Radio(
                        label=i18n("hsi_processing.mat_format"),
                        choices=[
                            'HWC', 
                            'CHW'
                        ],
                        value='CHW'
                    )
                    mat_key = gr.Text(
                        label=i18n("hsi_processing.mat_key"),
                        value='data',
                        max_lines=1
                    )
                    compress_mat = gr.Checkbox(
                        label=i18n("hsi_processing.compress_mat"),
                        value=True
                    )
                    convert_btn  = gr.Button(i18n("hsi_processing.apply_processing"), variant="primary")
                
                with gr.Accordion(i18n("hsi_processing.spectral_selection"),visible=False) as plot_panel:
                    gr.Markdown(i18n('hsi_processing.spectral_selection_help'))
                    spectral_plot = gr.Plot(
                        label=i18n("hsi_processing.spectral_plot"),
                        visible=True,
                        # x="Wavelength", y="Reflectance",
                        # height=400, width=600,
                    )
                    plot_hint = gr.Textbox(
                        label=i18n("hsi_processing.style"),
                        value='b-',
                    )
                    with gr.Row():
                        clear_plot_btn = gr.Button(
                            i18n("hsi_processing.clear"),
                            variant="secondary",
                        )
                        download_select_spectral = gr.DownloadButton(
                            "Download",
                            variant="primary",
                        )

            with gr.Column(scale=2):
                with gr.Column(variant="panel"):
                    gr.Markdown(f"## {i18n('hsi_processing.output_results')}")   
                    preview_img = gr.Image(label=i18n("hsi_processing.preview"), format="png", height="auto", width="auto", interactive=False)
                    mat_file_output = gr.File(
                        label=i18n("hsi_processing.mat_file"),
                        type="filepath",
                        interactive=False
                    )

        # 回调函数   
        reload_btn.click(
            fn=gr_load,
            inputs=[state_current_layer_index, state_transforms, state_original_rgb, state_original_data, state_data_path, dat_files, input_format, input_mat_key, manual_normalize, normalize_min, normalize_max, wavelength_from, wavelength_to],
            outputs=[state_transforms, state_original_rgb, state_original_data, state_data_path, state_ui_state]
        )
        reset_loaded_btn.click(
            fn=lambda : ([],[],[],AppState.NOT_LOADED), # TODO: add logging info
            inputs=[],
            outputs=[state_original_rgb, state_original_data, state_data_path, state_ui_state]
        )
        convert_btn.click(
            fn=gr_convert,
            inputs=[
                state_current_layer_index,
                state_transforms,
                state_original_data, state_data_path,
                mat_dtype, output_format, mat_key, compress_mat
            ],
            outputs=[state_processed_data ,mat_file_output, state_ui_state]
        )

        # 实时更新state_transforms。
        for component in [crop_top, crop_left, crop_bottom, crop_right, rotate_deg, offset_x, offset_y]:
            component.change(
                fn = gr_update_transforms,
                inputs=[state_transforms, state_current_layer_index, crop_top, crop_left, crop_bottom, crop_right, rotate_deg, offset_x, offset_y],
                outputs=[state_transforms],
            )
        # state_transforms更新会触发重新合成
        state_transforms.change(
            fn=gr_composite,
            inputs=[state_original_rgb, state_transforms],
            outputs=[preview_img, state_ui_state],
        )
        # 绑定图层ID滑块
        state_data_path.change(
            fn = gr_on_state_data_path_changed,
            inputs=[state_data_path, state_current_layer_index, current_layer_radio],
            outputs=[current_layer_radio],
        )
        current_layer_radio.change(
            fn = lambda x:x,
            inputs=[current_layer_radio],
            outputs=[state_current_layer_index]
        )

        # ------------
        state_current_layer_index.change(
            fn=gr_on_state_current_layer_index_changed,
            inputs=[state_current_layer_index, state_transforms, state_original_data],
            outputs=[
                crop_top, crop_left, crop_bottom, crop_right, rotate_deg, offset_x, offset_y, state_ui_state, dat_files
            ],
        )
        state_original_data.change(
            fn=gr_on_state_current_layer_index_changed,
            inputs=[state_current_layer_index, state_transforms, state_original_data],
            outputs=[
                crop_top, crop_left, crop_bottom, crop_right, rotate_deg, offset_x, offset_y, state_ui_state, dat_files
            ]
        )
        # ------------
        
        preview_img.select(
            fn=gr_on_img_clicked,
            inputs=[state_spectral_figure, state_processed_data, state_selected_location, wavelength_from, wavelength_to, plot_hint],
            outputs=[spectral_plot, state_spectral_figure, state_selected_location],
        )
        clear_plot_btn.click(
            fn=lambda: (None, None, []),
            outputs=[spectral_plot, state_spectral_figure, state_selected_location],
        )
        download_select_spectral.click(
            fn=gr_download_selected_spectral,
            inputs=[state_current_layer_index, state_processed_data, state_selected_location, state_data_path],
            outputs=[download_select_spectral]
        )

        # 加载新图片后，重新合成
        state_original_rgb.change(
            fn=gr_composite,
            inputs=[state_original_rgb, state_transforms],
            outputs=[preview_img, state_ui_state]
        )
        state_ui_state.change(
            fn=lambda x: (
                gr.update(visible=True, open=(x==AppState.NOT_LOADED)),
                gr.update(visible=x!=AppState.NOT_LOADED, open=(x in [AppState.LOADED, AppState.PREVIEWED])),
                gr.update(visible=x!=AppState.NOT_LOADED, open=(x in [AppState.LOADED, AppState.PREVIEWED])),
                gr.update(visible=x!=AppState.NOT_LOADED,   open=(x==AppState.COVERTED)) # Should be x==AppState.COVERTED, just in case for potentional bugs
            ), 
            inputs=[state_ui_state],
            outputs=[load_panel, preview_panel, convert_panel, plot_panel]
        )

        # 转置
        # input_format.change(
        #     fn=gr_transpose_original_data,
        #     inputs=[state_current_layer_index, input_format, state_original_data],
        #     outputs=[state_transforms, state_original_rgb, state_original_data, AppState.LOADED]
        # )