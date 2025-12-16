from ..util import records_to_html
import gradio as gr
import logging
# from functools import partial

# def _one_timer_tick(handler, state_n_line):
#     n_line = len(handler.buffer)
#     print(f"{n_line=} {state_n_line=}")
#     if n_line == state_n_line:
#         print("SKIP")
#         return gr.skip(), state_n_line   
#     else:
#         return gr.update(value=records_to_html(handler.buffer, with_pre_tag=True)), n_line

# Gradio 5.50中存在一个BUG，timer.tick中的show_progress="hidden"和gr.skip依然会触发loading动画，通过一个state来workaround

def create_gr_logger(handler :logging.handlers.MemoryHandler):
    state_n_line = gr.State(0)
    timer = gr.Timer(value=2)
    html = gr.HTML(
        value="Loading.....",
        label='Logging',
        show_label=True,
        container=True,
        min_height=50, 
        max_height=100,
    )
    state_n_line.change(
        fn=lambda: records_to_html(handler.buffer, with_pre_tag=True),
        inputs=[],
        outputs=[html],
        show_progress="hidden"
    )
    timer.tick(
        fn=lambda: len(handler.buffer),
        inputs=[],
        outputs=[state_n_line],
        show_progress="hidden"
    )