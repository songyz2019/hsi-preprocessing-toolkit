import gradio as gr
from ..common import i18n, APP_VERSION

from importlib import resources
from string import Template

def load_about_md(lang :str):
    if lang.startswith('zh'):
        lang = 'zh'
    else:
        lang = 'en'
    text = resources.files("hsi_preprocessing_toolkit.asset.page").joinpath(f"about.{lang}.md").read_text(encoding="utf-8")
    return Template(text).safe_substitute(
        VERSION=APP_VERSION
    )

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
        about_content = gr.Markdown(load_about_md('zh')  + '\n\n' + load_about_md('en'))

        # lang_state.change(
        #     fn = load_about_md,
        #     inputs=[lang_state],
        #     outputs=[about_content]
        # )

