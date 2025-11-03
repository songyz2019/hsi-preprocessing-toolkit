import matplotlib.pyplot as plt
import gradio as gr
import gradio.utils
import logging
from .page.scanner_calc import scanner_calc_tab
from .page.about import about_tab
from .page.hsi_preprocessing import hsi_preprocessing_tab
from .constant import i18n

logging.basicConfig(level=logging.INFO)

def main():
    theme = gr.themes.Default(primary_hue='cyan').set(
        button_primary_background_fill='#39c5bb',
        button_primary_background_fill_hover="#30A8A0",
    )

    with gr.Blocks(title="hsi-preprocessing-toolkit", theme=theme) as demo:
        hsi_preprocessing_tab()
        scanner_calc_tab()
        about_tab()

    demo.launch(share=False, inbrowser=True, i18n=i18n, favicon_path="asset/icon.ico")


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'SimHei'
    main()