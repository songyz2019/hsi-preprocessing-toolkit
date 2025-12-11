import gradio as gr
from ..common import i18n, CONSTS

def AboutTab():
    with gr.Tab(i18n('about.tab_title')):
        with gr.Column():
            gr.Markdown('# ' + i18n('about.title'))
            gr.Markdown(i18n('about.description'))
            gr.Markdown(f"[{i18n('about.homepage')}]({CONSTS['homepage_url']})")
            gr.Markdown("```text\n" + i18n('about.license') + "\n```")


