import gradio as gr
from ..constant import i18n

def _calculate_scanner_parameters(meta_pixel_size, n_wpixel, focal, h, delta_t):
    """All Unit should be in SI unit."""
    v = (meta_pixel_size * h) / (focal * delta_t)
    resolution = (meta_pixel_size * h) / focal
    return v, resolution

def calculate_scanner_parameters(meta_pixel_size, n_wpixel, focal, h, delta_t, pulse_per_mm):
    """ Use the provide unit"""
    meta_pixel_size /= 1e6  # Convert um to m
    focal /= 1000  # Convert mm to m
    delta_t /= 1000  # Convert ms to s
    v, resolution = _calculate_scanner_parameters(meta_pixel_size, n_wpixel, focal, h, delta_t)  # Convert ms to s
    v *= 1000  # Convert m/s to mm/s
    resolution *= 1000  # Convert m to mm
    pulse_freq = round(v) * pulse_per_mm
    return v, resolution, pulse_freq

def scanner_calc_tab():
    with gr.Tab(i18n("scanner_calc.tab_title")):
        with gr.Row():
            with gr.Column(variant="panel"):
                # unit_scale = gr.Radio(
                #     label="单位制",
                #     choices=[ 
                #         ("默认", "default"), 
                #         ("国际单位制", "si"),
                #     ],
                #     value="default",
                #     visible=True
                # )
                with gr.Accordion("相机参数", open=False):
                    meta_pixel_size = gr.Number(label="像元尺寸 (um)", value=7.40, precision=2)
                    n_wpixel = gr.Number(label="横向空间像素数量", value=640)
                    focal = gr.Number(label="焦距 Focal (mm)", value=8.0, precision=1)
                with gr.Accordion("电机参数", open=False):
                    pulse_per_mm = gr.Number(label="每毫米脉冲数", value=160, precision=2)
                h = gr.Number(label="物距 Distance (m)", value=0.432, precision=3)
                delta_t = gr.Number(label="帧间隔 Frame Period (ms)", value=100)
                btn = gr.Button("计算", variant="primary")
            with gr.Column(variant="panel"):
                gr.Markdown("### 计算结果")
                pulse_freq = gr.Number(label="脉冲频率 (Hz)", interactive=False, precision=1)
                v = gr.Number(label="推扫速度 (mm/s)", interactive=False, precision=2)
                resolution = gr.Number(label="分辨率 (mm/pixel)", interactive=False, precision=3)

            btn.click(
                calculate_scanner_parameters,
                inputs=[meta_pixel_size, n_wpixel, focal, h, delta_t, pulse_per_mm],
                outputs=[v, resolution, pulse_freq]
            )
