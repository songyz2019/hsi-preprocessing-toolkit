from ..util import records_to_html
import gradio as gr
import logging


def create_gr_logger(handler :logging.handlers.MemoryHandler):
    return gr.HTML(
        value=lambda: records_to_html(handler.buffer),
        label='Logging',
        every= 0.5,
        autoscroll=False,
        container=True,
        css_template="max-height: 20em; overflow: auto;",
        html_template="<label>Logging</label><pre>${value}</pre>",
        # js_on_load="document.querySelector('pre#logging').addEventListener('change', e => e.scrollTo )"
    )