import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "unsloth/mistral-7b-bnb-4bit"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

def generate_answer(question):

    prompt = f"""
You are a helpful medical assistant.

Question: {question}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("Answer:")[-1].strip()


demo = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(label="Ask a medical question"),
    outputs=gr.Textbox(label="AI Medical Response"),
    title="🧠 Medical AI Chatbot (Mistral-7B)",
    description="Fine-tuned medical assistant based on Mistral-7B."
)

demo.launch()
