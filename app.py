from transformers import pipeline
import gradio as gr

# Load a lightweight Hugging Face model
pipe = pipeline("text2text-generation", model="google/flan-t5-small")

# Define chatbot logic
def chatbot(input_text):
    prompt = f"Act as a postpartum care expert. {input_text}"
    result = pipe(prompt, max_new_tokens=150, temperature=0.5)[0]["generated_text"]
    return result

# Gradio UI
inputs = gr.Textbox(lines=7, label="Chat with AI")
outputs = gr.Textbox(lines=7, label="Reply")

gr.Interface(
    fn=chatbot,
    inputs=inputs,
    outputs=outputs,
    title="AI Postpartum Expert",
    description="Ask anything about Postpartum care",
    theme="compact"
).launch(share=True)
