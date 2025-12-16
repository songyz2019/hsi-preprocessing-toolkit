import matplotlib.pyplot as plt
import gradio as gr
from .page.scanner_calc import ScannerCalcTab
from .page.about import AboutTab
from .page.hsi_preprocessing import HSIProcessingTab
from .component.create_logger import create_gr_logger
from .common import i18n, DEBUG, MULTI_USER, LOGGER_MEMORY_HANDLER, CONSTS


def main():
    # Global State
    plt.rcParams['font.family'] = 'SimHei'
    theme = gr.themes.Default(primary_hue='cyan').set(
        button_primary_background_fill='#39c5bb',
        button_primary_background_fill_hover="#30A8A0",
    )

    with gr.Blocks(title=CONSTS['name'], theme=theme) as demo: # title=i18n("about.title")
        HSIProcessingTab()
        ScannerCalcTab()
        AboutTab()
        if not MULTI_USER:
            create_gr_logger(LOGGER_MEMORY_HANDLER)

    demo.launch(debug=DEBUG, share=False, inbrowser=True, i18n=i18n, favicon_path="asset/icon.ico") # , theme=theme


if __name__ == "__main__":
    main()