from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr

# Use Falcon 1B Instruct (small + public)
model_id = "tiiuae/falcon-rw-1b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Chat function
def chat(message):
    prompt = f"{message}"
    response = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    return response.strip()

# Gradio UI
gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Cora: AI Chatbot",
    description="A lightweight AI assistant powered by Falcon 1B Instruct. Ask anything!"
).launch()
