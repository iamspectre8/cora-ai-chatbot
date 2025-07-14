from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr
import torch

# Load the model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Create the text generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

def chat(message):
    prompt = f"[INST] {message} [/INST]"
    output = chatbot(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    return output.split("[/INST]")[-1].strip()

# Gradio interface
iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="Cora: AI Chatbot")

if __name__ == "__main__":
    iface.launch()
