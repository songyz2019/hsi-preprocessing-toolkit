from pathlib import Path
import shutil
from typing import Tuple, Union, List
import einops
import gradio.utils
import numpy as np
import matplotlib.pyplot as plt
from rasterio import open as rasterio_open
from scipy.io import savemat
import gradio
from scipy.ndimage import rotate
from rs_fusion_datasets.util.fileio import load_one_key_mat
from rs_fusion_datasets.util.hsi2rgb import _hsi2rgb, hsi2rgb
from jaxtyping import Float, Int
plt.rcParams['font.family'] = 'SimHei'

# Void -> HWC
def load_data(
    dat_files :List[gradio.utils.NamedString] | None,
    input_format,
) -> Tuple[np.ndarray | None, Path | None, str]:
    logging_text = ""
    if dat_files is None or len(dat_files) == 0:
        return None, None, "No data file provided."

    dat_paths = [Path(f.name) for f in dat_files]
    mat_paths = [p for p in dat_paths if p.suffix == '.mat']
    hdr_paths = [p for p in dat_paths if p.suffix == '.hdr']
    raw_paths = [p for p in dat_paths if p.suffix == '']

    if len(mat_paths) > 0:
        data_path = mat_paths[0]
        data = load_one_key_mat(mat_paths[0])
    elif len(hdr_paths) == 0 or len(raw_paths) == 0:
        return None, None, "请同时上传.hdr文件和无后缀的数据文件. Both .hdr and raw data files are required."
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
        return None, None, "无法识别数据文件格式. Unrecognized data file format."

    # Convert input format
    if input_format == 'CHW':
        data = einops.rearrange(data, 'c h w -> h w c') # type: ignore

    return data, data_path, "" # type: ignore


# HWC -> HWC
def process_img(img, crop_top, crop_left, crop_bottom, crop_right, rotate_deg, rotate_reshape):
    # Rotate
    if rotate_deg != 0:
        img = rotate(img, angle=rotate_deg, axes=(0, 1), reshape=rotate_reshape)

    # Crop
    if crop_top > 0 or crop_left > 0 or crop_bottom > 0 or crop_right > 0:
        crop_bottom = None if crop_bottom == 0 else -crop_bottom
        crop_right  = None if crop_right == 0  else -crop_right
        img = img[crop_top:crop_bottom, crop_left:crop_right, :]

    return img

def gr_on_file_upload(
        dat_files :List[gradio.utils.NamedString] | None,
        input_format, 
        manual_normalize: bool, normalize_min: float, normalize_max: float,
        wavelength_from: int, wavelength_to: int,
        logging_text: str
    ):
    data, data_path, _logging_text = load_data(dat_files, input_format)
    logging_text += _logging_text + "\n"
    if data is None:
        return None, None, None, logging_text+"\nupload error"
    if not manual_normalize:
        rgb = hsi2rgb(data, wavelength_range=(wavelength_from, wavelength_to), input_format='HWC', output_format='HWC', to_u8np=True)
    else:
        rgb = _hsi2rgb( (data-normalize_min)/(normalize_max-normalize_min), wavelength=np.linspace(wavelength_from, wavelength_to, data.shape[-1]))
        rgb = (rgb*255.0).astype(np.uint8)
    return rgb, data, data_path, logging_text

def gr_preview(
        preview_img,
        crop_top: int, crop_left: int, crop_bottom: int, crop_right: int, 
        rotate_deg: int, rotate_reshape: bool
    ):
    return process_img(
        preview_img, crop_top, crop_left, crop_bottom, crop_right, rotate_deg, rotate_reshape
    )

def gr_download_selected_spectral(data, state_select_location, data_path):
    if not state_select_location or len(state_select_location) == 0:
        return None
    
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
    
    return str(mat_path)  # Return string path for DownloadButton

def gr_convert(
        data, data_path,
        crop_top: int, crop_left: int, crop_bottom: int, crop_right: int, 
        rotate_deg: int, rotate_reshape: bool,
        mat_dtype, output_format, mat_key, compress_mat: bool,
        logging_text: str
    ):
    # Load `data`
    if output_format == 'Same as Input':
        output_format = input_format
    
    data = process_img(
        data, crop_top, crop_left, crop_bottom, crop_right, rotate_deg, rotate_reshape
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
    return data_processed, str(mat_file), logging_text


def gr_on_img_clicked(evt: gradio.SelectData, data, state_select_location, wavelength_from: int, wavelength_to :int, plot_hint: str):
    """绘制选中像素的光谱曲线"""
    if data is None:
        return None, state_select_location
    if state_select_location is None:
        state_select_location = []
    col, row = evt.index
    state_select_location.append((row, col)) # WHY?

    wavelengths = np.linspace(wavelength_from, wavelength_to, data.shape[-1])
    fig, ax = plt.subplots(figsize=(10, 6))
    for (row, col) in state_select_location:
        ax.plot(wavelengths, data[row, col, :], plot_hint ,linewidth=2)
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Reflectance')
    plt.tight_layout()
    return fig, state_select_location


if __name__ == "__main__":
    with gradio.Blocks(title="HDR TO MAT") as demo:
        with gradio.Row():
            with gradio.Column():
                with gradio.Column(variant="panel"):
                    gradio.Markdown("## 输入")
                    # Input Files
                    gradio.Markdown("**应上传以下两种格式中的一种**  \n1. 一个.hdr文件 + 一个无后缀的数据文件  \n2. 一个.mat文件")
                    input_format = gradio.Radio(
                        label="输入数据形状 Input Image Shape Format",
                        choices=['HWC', 'CHW'],
                        value='CHW'
                    )
                    dat_files = gradio.File(
                        label="数据文件 DAT File",
                        file_count="multiple",
                        type="filepath",
                    )

                    manual_normalize = gradio.Checkbox(
                        label="手动归一化(仅影响预览结果) Manual Normalize",
                        value=False,
                    )
                    with gradio.Row():
                        normalize_min = gradio.Number(
                            label="归一化最小值 Normalize Min",
                            value=0,
                            precision=1,
                            visible=False,
                        )
                        normalize_max = gradio.Number(
                            label="归一化最大值 Normalize Max",
                            value=2**16-1,
                            precision=1,
                            visible=False,
                        )

                    def toggle_normalize_fields(manual_normalize):
                        return (
                            gradio.update(visible=manual_normalize),
                            gradio.update(visible=manual_normalize),
                        )
                    manual_normalize.change(
                        fn=toggle_normalize_fields,
                        inputs=[manual_normalize],
                        outputs=[normalize_min, normalize_max],
                    )
                    with gradio.Row():
                        wavelength_from = gradio.Number(
                            label="波长范围起始 Wavelength Range Start",
                            value=400,
                            precision=1,
                        )
                        wavelength_to = gradio.Number(
                            label="波长范围结束 Wavelength Range End",
                            value=1000,
                            precision=1,
                        )
                    reload_btn = gradio.Button("重新加载", variant="primary")
                
                with gradio.Column(variant="panel"):
                    gradio.Markdown("## 处理")
                    with gradio.Column():
                        gradio.Markdown("### 裁切 Crop")
                        with gradio.Row():
                            crop_top = gradio.Number(
                                label="上方裁切像素 Crop Top",
                                value=0,
                                precision=1,
                            )
                            crop_bottom = gradio.Number(
                                label="下方裁切像素 Crop Bottom",
                                value=0,
                                precision=1,
                            )
                        with gradio.Row():
                            crop_left = gradio.Number(
                                label="左侧裁切像素 Crop Left",
                                value=0,
                                precision=1,
                            )
                            crop_right = gradio.Number(
                                label="右侧裁切像素 Crop Right",
                                value=0,
                                precision=1,
                            )

                    with gradio.Column():
                        gradio.Markdown("### 旋转 Rotate")
                        rotate_deg = gradio.Number(
                            label="旋转角度 Rotate Degree",
                            value=0,
                            precision=1,
                        )
                        rotate_reshape = gradio.Checkbox(
                            label="旋转时调整形状 Rotate Reshape",
                            value=True
                        )
                    preview_btn  = gradio.Button("预览", variant="primary")
                    


                # MAT Output Options
                with gradio.Column(variant="panel"):
                    gradio.Markdown("## 输出选项")
                    mat_dtype = gradio.Radio(
                        label="mat文件数据类型 MAT Data Type",
                        choices=["default", "uint8", "uint16", "float32"],
                        value="default"
                    )
                    output_format = gradio.Radio(
                        label="mat文件格式 MAT Image Shape Format",
                        choices=['Same as Input', 'HWC', 'CHW'],
                        value='Same as Input'
                    )
                    mat_key = gradio.Text(
                        label="mat文件的key Key of MAT file",
                        value='data',
                    )
                    compress_mat = gradio.Checkbox(
                        label="启用mat文件压缩 Produce Compressed MAT File",
                        value=True
                    )
                    convert_btn  = gradio.Button("应用处理效果", variant="primary")

            with gradio.Column():
                with gradio.Column(variant="panel"):
                    gradio.Markdown("## 输出结果 Output")   
                    preview_img = gradio.Image(label="预览 Preview", format="png", height="auto", width="auto", interactive=False)
                    mat_file_output = gradio.File(
                        label="MAT File",
                        type="filepath",
                        interactive=False
                    )
                
                with gradio.Column(variant="panel"):
                    gradio.Markdown("## 光谱选择 Spectral Selection")
                    spectral_plot = gradio.Plot(
                        label="光谱图 Spectral Plot",
                        visible=True,
                    )
                    plot_hint = gradio.Textbox(
                        label="样式 Style",
                        value='b-',
                    )
                    with gradio.Row():
                        clear_plot_btn = gradio.Button(
                            "清空 Clear",
                            variant="secondary",
                        )
                        download_select_spectral = gradio.DownloadButton(
                            "下载 Download",
                            variant="primary",
                        )

                with gradio.Column(variant="panel"):
                    gradio.Markdown("## 日志 Log")
                    logging_text = gradio.Textbox(
                        label="信息 Info",
                    )
    
        state_original_data = gradio.State(
            value=None,
        )
        state_processed_data = gradio.State(
            value=None,
        )
        state_data_path = gradio.State(
            value=None,
        )
        state_select_location = gradio.State(
            value=[],
        )
        state_rgb = gradio.State(
            value=None,
        )

        # Bind functions
        dat_files.upload(
            fn=gr_on_file_upload,
            inputs=[dat_files, input_format, manual_normalize, normalize_min, normalize_max, wavelength_from, wavelength_to, logging_text],
            outputs=[state_rgb, state_original_data, state_data_path, logging_text]
        )
        reload_btn.click(
            fn=gr_on_file_upload,
            inputs=[dat_files, input_format, manual_normalize, normalize_min, normalize_max, wavelength_from, wavelength_to, logging_text],
            outputs=[state_rgb, state_original_data, state_data_path, logging_text]
        )
        convert_btn.click(
            fn=gr_convert,
            inputs=[
                state_original_data, state_data_path,
                crop_top, crop_left, crop_bottom, crop_right,
                rotate_deg, rotate_reshape,
                mat_dtype, output_format, mat_key, compress_mat,
                logging_text
            ],
            outputs=[state_processed_data ,mat_file_output, logging_text]
        )
        preview_btn.click(
            fn=gr_preview,
            inputs=[
                state_rgb,
                crop_top, crop_left, crop_bottom, crop_right,
                rotate_deg, rotate_reshape,
            ],
            outputs=[preview_img]
        )
        preview_img.select(
            fn=gr_on_img_clicked,
            inputs=[state_processed_data, state_select_location, wavelength_from, wavelength_to, plot_hint],
            outputs=[spectral_plot, state_select_location]
        )
        clear_plot_btn.click(
            fn=lambda: (None, []),
            outputs=[spectral_plot, state_select_location]
        )
        download_select_spectral.click(
            fn=gr_download_selected_spectral,
            inputs=[state_processed_data, state_select_location, state_data_path],
            outputs=[download_select_spectral]
        )

    demo.launch(share=False, inbrowser=True)

