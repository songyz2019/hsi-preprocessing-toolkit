# import gradio as gr
# import logging
# from collections import deque

# We assume that there're limited users. Since every user *SHOULD* have a copy of logging info in server.
# But actually, it is only creatd once when opening multiple tabs in browsers, wierd...

# class _DequeLogHandler(logging.Handler):
#     def __init__(self, max_logging_lines: int):
#         super().__init__()
#         self.buffer = deque([], maxlen=max_logging_lines)

#     def emit(self, record):
#         msg = self.format(record)
#         self.buffer.appendleft(msg) # Some how auto scroll not work :(, we have to use a upside-down logging textbox
    
#     def get_logging(self, seprator='\n'):
#         return seprator.join(self.buffer)
    

# def create_gr_logging_box(handler :logging.handlers.BufferingHandler, max_logging_lines=10_000, **kwargs) -> [logging.Handler, gr.Textbox]:
#     # Somehow extend gr.Textbox block the UI, we do not wanna dive deeply, just a quick workaround
#     """
#     Usage: 
#     handler, widget = create_gr_logging() # with any kwargs for gr.Textbox you need
#     logger.add_handler(handler)

#     """
#     kwargs |= {
#         'label': 'Logging',
#         'every': 0.5,
#         'buttons': ['copy'],
#         'lines': 10,
#         'max_lines': 10,
#         'autoscroll': False
#     }
#     textbox = gr.Textbox(
#         value=lambda: handler.get_logging(),
#         **kwargs
#     )

#     return handler, textbox

