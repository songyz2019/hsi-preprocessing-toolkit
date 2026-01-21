import matplotlib.pyplot as plt
import gradio as gr
import os

from .page.scanner_calc import ScannerCalcTab
from .page.about import AboutTab
from .page.hsi_preprocessing import HSIProcessingTab
from .component.create_logger import create_gr_logger
from .common import i18n, DEBUG, MULTI_USER, LOGGER_MEMORY_HANDLER, APP_NAME, LOGGER



def main():
    # Global State
    plt.rcParams['font.family'] = 'SimHei'
    theme = gr.themes.Default(primary_hue='cyan').set(
        button_primary_background_fill='#39c5bb',
        button_primary_background_fill_hover="#30A8A0",
    )

    with gr.Blocks(title=APP_NAME, theme=theme) as demo: # title=i18n("about.title")
        HSIProcessingTab()
        ScannerCalcTab()
        AboutTab()
        if not MULTI_USER:
            create_gr_logger(LOGGER_MEMORY_HANDLER)

    demo.launch(
        debug=DEBUG, 
        prevent_thread_lock=True, # 不阻塞主线程，配合webui/webbrowser
        share=False, 
        inbrowser=False, 
        i18n=i18n, 
        favicon_path="asset/icon.ico"
    )
    try:
        from webui import webui
        window = webui.Window()
        window.show_browser(demo.local_url, webui.browser.ChromiumBased) # 这里的行为有些奇怪，启动成功了但依然会返回false
        LOGGER.info('Launched in webui mode successfully')
        window.set_minimum_size(1024, 768)
        webui.wait() # wait似乎有bug
        input("Running ...") # wait bug的quick dirty fix
        # window.destroy()
        # demo.close()
        # os._exit(0)
    except Exception as err:
        window.destroy()
        LOGGER.warning(f'Fallback to browser mode, {err}')
        import webbrowser
        webbrowser.open_new(demo.local_url)
        input("Running ...") # quick dirty fix
        
if __name__ == "__main__":
    main()