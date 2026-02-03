# 🩺 Medical AI Chatbot — Fine-Tuning Mistral-7B with LoRA

This project presents a Medical Question-Answering Chatbot created by adapting the **Mistral-7B** language model through efficient **fine-tuning techniques**.
It provides an interactive medical assistant experience via a clean Gradio-based web interface.

---

## 📌 Project Overview

Large Language Models (LLMs) such as Mistral-7B can be adapted to specialized tasks.
In this work, we fine-tuned **Mistral-7B** to behave as a **medical assistant chatbot**
capable of answering basic medical questions clearly and truthfully.

The final model is deployed with an interactive chat interface using **Gradio**.

---

## ✅ Key Features

- Fine-tuning of **Mistral-7B** in the medical domain  
- Parameter-Efficient Fine-Tuning using **LoRA**  
- 4-bit Quantization (QLoRA style) for low GPU memory usage  
- Instruction-tuning format (Alpaca Prompt Template)  
- Chatbot deployment through a Gradio Web UI  

---

## 🧠 Model Used

- **Base Model:** Mistral-7B (Open-source LLM)
- **Fine-tuning Method:** LoRA (PEFT)
- **Optimization:** 4-bit quantization using bitsandbytes

---

## 📂 Dataset

A public medical Question/Answer dataset from Hugging Face was used: https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards


---

## ⚙️ Tools & Libraries

| Tool / Library | Role |
|--------------|------|
| **Mistral-7B** | Pre-trained LLM base model |
| **Unsloth** | Fast fine-tuning and optimized inference |
| **LoRA / PEFT** | Efficient parameter adaptation |
| **bitsandbytes** | 4-bit quantization support |
| **TRL (SFTTrainer)** | Supervised fine-tuning trainer |
| **Transformers** | Tokenization + text generation |
| **Gradio** | Chatbot Web Interface |

---

## 🚀 Training Process

The fine-tuning was performed using supervised learning (SFT):

- Batch size: 2  
- Gradient accumulation: 4  
- Learning rate: 2e-4  
- Steps: 60+  

Only LoRA adapters were trained, making the process feasible on Google Colab GPUs.


