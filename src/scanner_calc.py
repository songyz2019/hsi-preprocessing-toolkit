import gradio as gr

def _calculate_scanner_parameters(meta_pixel_size, n_wpixel, focal, h, delta_t):
    """All Unit should be in SI unit."""
    v = (meta_pixel_size * h) / (focal * delta_t)
    resolution = (meta_pixel_size * h) / focal
    return v, resolution

def calculate_scanner_parameters(meta_pixel_size, n_wpixel, focal, h, delta_t):
    """ Use the provide unit"""
    meta_pixel_size /= 1e6  # Convert um to m
    focal /= 1000  # Convert mm to m
    delta_t /= 1000  # Convert ms to s
    v, resolution = _calculate_scanner_parameters(meta_pixel_size, n_wpixel, focal, h, delta_t)  # Convert ms to s
    v *= 100  # Convert m/s to cm/s
    resolution *= 100  # Convert m to cm
    return v, resolution

def scanner_calc_tab():
    with gr.Row():
        with gr.Column():
            meta_pixel_size = gr.Number(label="像元尺寸 (um)", value=7.40, precision=2)
            n_wpixel = gr.Number(label="横向空间像素数量", value=640)
            focal = gr.Number(label="焦距 (mm)", value=8.0, precision=1)
            h = gr.Number(label="物距(高度) (m)", value=0.432, precision=3)
            delta_t = gr.Number(label="帧间隔 (ms)", value=100)
            btn = gr.Button("计算", variant="primary")
        with gr.Column():
            v = gr.Number(label="推扫速度 (cm/s)", interactive=False, precision=2)
            resolution = gr.Number(label="分辨率 (cm/pixel)", interactive=False, precision=3)

        btn.click(
            calculate_scanner_parameters,
            inputs=[meta_pixel_size, n_wpixel, focal, h, delta_t],
            outputs=[v, resolution]
        )

__all__ = ["scanner_calc_tab"]