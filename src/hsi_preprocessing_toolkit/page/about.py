import gradio as gr
from ..i18n import i18n

def about_tab():
    with gr.Tab(i18n('about.tab_title')):
        with gr.Column(variant="panel"):
            gr.Markdown('# ' + i18n('about.title'))
            gr.Markdown(i18n('about.description'))
            gr.Markdown(i18n('about.homepage'))


