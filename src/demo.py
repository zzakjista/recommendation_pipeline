import gradio as gr
from recommender_service import load_generator

generator = load_generator()

def predict(text):
    result = generator.generate(0)
    return f"Predicted label for {result}"

if __name__ == '__main__':
    interface = gr.Interface(fn=predict, inputs="text", outputs="text")
    interface.launch()