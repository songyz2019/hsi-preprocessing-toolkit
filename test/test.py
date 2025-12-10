import gradio as gr

# Create an I18n instance with translations for multiple languages
i18n = gr.I18n(**{
    'en':{"greeting": "Hello, welcome to my app!", "submit": "Submit"},
    'zh-CN':{"greeting": "¡Hola, bienvenido a mi aplicación!", "submit": "Enviar"},
    'fr':{"greeting": "Bonjour, bienvenue dans mon application!", "submit": "Soumettre"}
}
)

with gr.Blocks() as demo:
    # Use the i18n method to translate the greeting
    gr.Markdown(i18n("greeting"))
    with gr.Row():
        input_text = gr.Textbox(label="Input")
        output_text = gr.Textbox(label="Output")
    
    submit_btn = gr.Button(i18n("submit"))

# Pass the i18n instance to the launch method
demo.launch(i18n=i18n)