import matplotlib.pyplot as plt
import gradio as gr
import os

from .page.scanner_calc import ScannerCalcTab
from .page.about import AboutTab
from .page.hsi_preprocessing import HSIProcessingTab
from .component.create_logger import create_gr_logger
from .common import i18n, ARGS, LOGGER_MEMORY_HANDLER, APP_NAME, LOGGER, ACCESS_TOKEN

def get_port():
    if ARGS.port is None:
        from portpicker import pick_unused_port
        return pick_unused_port(ARGS.host)
    return ARGS.port

def auth_func(req):
    # if ACCESS_TOKEN == req.cookies.get("access_token"):
    #     return "admin"
    if ACCESS_TOKEN == req.query_params.get("access_token"):
        # resp.set_cookie(key="access_token", value=ACCESS_TOKEN, httponly=True, samesite="lax")
        return 'admin'
    if ACCESS_TOKEN in req.headers.get("referer", ""):
        return "admin"
    return None




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
        create_gr_logger(LOGGER_MEMORY_HANDLER)

    port = get_port()
    access_url = f'http://{ARGS.host}:{port}/' + ( f'?access_token={ACCESS_TOKEN}' if not ARGS.no_access_token else '')
    
    mode = ARGS.mode
    if ARGS.mode=='browser' and ARGS.no_access_token:
        mode = 'browser no_access_token'
    
    if ARGS.mode == 'server':
        print(f"{access_url=}")

    demo.launch(
        debug=ARGS.debug, 
        prevent_thread_lock = not (mode in ['server','browser no_access_token']),
        share=False, 
        inbrowser = mode=='browser no_access_token', 
        i18n=i18n, 
        favicon_path="asset/icon.ico",
        auth_dependency=None if ARGS.no_access_token else auth_func,
        server_name=ARGS.host,
        server_port=port,
    )
    # Block here if  in server mode
    if mode == 'window':
        try:
            from webui import webui
            window = webui.Window()
            window.show_browser(access_url, webui.browser.ChromiumBased) # 这里的行为有些奇怪，启动成功了但依然会返回false
            LOGGER.info('Launched in webui mode successfully')
            window.set_minimum_size(1024, 768)
            webui.wait() # wait似乎有bug
            input("Running ...") # wait bug的quick dirty fix
            # window.destroy()
            # demo.close()
            # os._exit(0)
        except Exception as err:
            LOGGER.warning(f'Failed to start in window mode, {err}')
            if window is not None:
                window.destroy()
            mode = 'browser'
    
    if mode == 'browser':
        import webbrowser
        webbrowser.open_new(access_url)
        input(f"Running on {access_url}...") # quick dirty fix
        
if __name__ == "__main__":
    main()