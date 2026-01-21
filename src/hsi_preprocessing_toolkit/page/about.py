import gradio as gr
from ..common import i18n, APP_VERSION



def AboutTab():
    with gr.Tab(i18n('about.tab_title')):
        # lang_state = gr.State()

        # gr.HTML(
        #     "<script defer>document.querySelector('#language_holder').innerHTML=navigator.language;</script>",
        #     elem_id="language_holder",
        # ).change(
        #     lambda x:x,
        #     outputs=lang_state
        # )
        gr.Markdown(i18n('about.content'))

        # lang_state.change(
        #     fn = load_about_md,
        #     inputs=[lang_state],
        #     outputs=[about_content]
        # )

