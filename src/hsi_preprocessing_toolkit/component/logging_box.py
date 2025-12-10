import gradio as gr
import logging
from collections import deque

class _DequeLogHandler(logging.Handler):
    def __init__(self, buffer :deque):
        super().__init__()
        self.buffer = buffer

    def emit(self, record):
        msg = self.format(record)
        self.buffer.appendleft(msg) # Some how auto scroll not work :(, we have to use a upside-down logging textbox

def create_gr_logging_box(max_logging_lines=10_000,**kwargs) -> [logging.Handler, gr.Textbox]:
    # Somehow extend gr.Textbox block the UI, we do not wanna dive deeply, just a quick workaround
    """
    Usage: 
    handler, widget = create_gr_logging() # with any kwargs for gr.Textbox you need
    logger.add_handler(handler)

    """
    kwargs |= {
        'label': 'Logging',
        'every': 0.5,
        # 'buttons': ['copy'],
        'lines': 10,
        'max_lines': 10,
        # 'autoscroll': True
    }
    logging_buffer = deque([], maxlen=max_logging_lines)
    textbox = gr.Textbox(
        value=lambda: '\n'.join(logging_buffer),
        **kwargs
    )

    return _DequeLogHandler(logging_buffer), textbox

