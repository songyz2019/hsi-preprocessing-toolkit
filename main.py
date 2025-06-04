from pathlib import Path
import shutil
from typing import Union
import einops
import gradio.utils
from rasterio import open as rasterio_open
from scipy.io import savemat
import gradio
from hsi2rgb import hsi2rgb

def gr_main(
        dat_file :Union[gradio.utils.NamedString, Path], hdr_file :Union[gradio.utils.NamedString, Path], input_format,
        mode, crop_top: int, crop_left: int, crop_bottom: int, crop_right: int,
        mat_dtype, output_format, mat_key, compress_mat: bool
    ):
    # Prepare
    if isinstance(hdr_file, gradio.utils.NamedString):
        hdr_file = Path(hdr_file.name)
    if isinstance(dat_file, gradio.utils.NamedString):
        dat_file = Path(dat_file.name)
    if hdr_file.parent != dat_file.parent:
        shutil.copy(hdr_file, dat_file.with_suffix('.hdr'))
    if output_format == 'Same as Input':
        output_format = input_format
    
    # Read
    print(f"Reading {dat_file}...", end=' ')
    with rasterio_open(dat_file, 'r') as src:
        data = src.read()
    print(f"Done")
    if input_format == 'CHW':
        data = einops.rearrange(data, 'c h w -> h w c')
    
    # Crop
    if crop_top > 0 or crop_left > 0 or crop_bottom > 0 or crop_right > 0:
        crop_bottom = None if crop_bottom == 0 else -crop_bottom
        crop_right  = None if crop_right == 0  else -crop_right
        data = data[crop_top:crop_bottom, crop_left:crop_right, :]
    
    # Convert to mat
    if mode == "转为Mat文件 Convert To Mat File":
        mat_file = dat_file.with_suffix('.mat')
        print(f"Converting {dat_file} to  {mat_file}...", end=' ')
        if mat_dtype == "default":
            mat_dat = data
        else:
            mat_dat = data.astype(mat_dtype)
        if output_format == 'CHW':
            mat_dat = einops.rearrange(mat_dat, 'h w c -> c h w')
        savemat(mat_file, {mat_key: data}, do_compression=compress_mat, format='5')  
        del mat_dat
        print(f"Done")
    else:
        mat_file = hdr_file # for debugging purposes, not actually saved


    # Generate RGB
    rgb = hsi2rgb(data, wavelength_range=(400, 1000),input_format='HWC', output_format='HWC')
    
    info = {}
    info['number_of_bands'] = data.shape[-1]
    info['width'] = data.shape[-2]
    info['height'] = data.shape[-3]
    info['input_data_type'] = str(data.dtype)
    info['wavelength_range'] = (400, 1000)
    info_str = "\n".join([f"{k}: {v}" for k, v in info.items()])

    return rgb, str(mat_file), info_str

if __name__ == "__main__":
    gradio.Interface(
        fn=gr_main,
        inputs=[
            # Input Files
            gradio.File(
                label="数据文件(无后缀) DAT File",
                file_count="single",
                type="filepath"
            ),
            gradio.File(
                label="hdr文件 HDR File",
                file_count="single",
                file_types=[".hdr"],
                type="filepath"
            ),

            gradio.Radio(
                label="输入数据形状 Input Image Shape Format",
                choices=['HWC', 'CHW'],
                value='CHW'
            ),

            # Processing Options
            gradio.Radio(
                label="任务 Mode",
                choices=["转为Mat文件 Convert To Mat File", "预览 Preview"],
                value="转为Mat文件 Convert To Mat File"
            ),

            gradio.Number(
                label="上方裁切像素 Crop Top",
                value=0,
                precision=1,
            ),
            gradio.Number(
                label="左侧裁切像素 Crop Left",
                value=0,
                precision=1,
            ),
            gradio.Number(
                label="下方裁切像素 Crop Bottom",
                value=0,
                precision=1,
            ),
            gradio.Number(
                label="右侧裁切像素 Crop Right",
                value=0,
                precision=1,
            ),

            # MAT Output Options
            gradio.Radio(
                label="mat文件数据类型 MAT Data Type",
                choices=["default", "uint8", "uint16", "float32"],
                value="default"
            ),
            gradio.Radio(
                label="mat文件格式 MAT Image Shape Format",
                choices=['Same as Input', 'HWC', 'CHW'],
                value='Same as Input'
            ),
            gradio.Text(
                label="mat文件的key Key of MAT file",
                value='data',
            ),
            gradio.Checkbox(
                label="启用mat文件压缩 Produce Compressed MAT File",
                value=True
            ),
        ],
        outputs=[
            gradio.Image(label="预览 Preview", format="png", height=320),
            gradio.File(
                label="MAT File",
                type="filepath"
            ),
            gradio.Textbox(
                label="信息 Info",
            ),
        ],
        title="HDR TO MAT",
        description="将hdr文件转为mat文件.预计时间1分钟/GB"
    ).launch(share=False, inbrowser=True)

