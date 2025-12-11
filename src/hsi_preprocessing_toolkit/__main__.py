import matplotlib.pyplot as plt
import gradio as gr
from .page.scanner_calc import ScannerCalcTab
from .page.about import AboutTab
from .page.hsi_preprocessing import HSIProcessingTab
from .common import i18n, DEBUG


def main():
    # Global State
    plt.rcParams['font.family'] = 'SimHei'
    theme = gr.themes.Default(primary_hue='cyan').set(
        button_primary_background_fill='#39c5bb',
        button_primary_background_fill_hover="#30A8A0",
    )

    with gr.Blocks(title=i18n("about.title")) as demo:
        HSIProcessingTab()
        ScannerCalcTab()
        AboutTab()

    demo.launch(debug=DEBUG, share=False, inbrowser=True, i18n=i18n, favicon_path="asset/icon.ico", theme=theme)


if __name__ == "__main__":
    main()