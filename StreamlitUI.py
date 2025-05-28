import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load your model and tokenizer (adjust as needed)
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def predict_answer(question, choices):
    prompt = f"Question: {question}\n"
    for idx, choice in enumerate(choices):
        prompt += f"{chr(ord('A') + idx)}. {choice}\n"
    prompt += "Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=5)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return pred

st.title("CommonsenseQA Demo")

question = st.text_area("Enter your question:")
choices = []
for i in range(5):
    choices.append(st.text_input(f"Choice {chr(ord('A')+i)}:"))

if st.button("Get Model Prediction"):
    if question and all(choices):
        pred = predict_answer(question, choices)
        st.success(f"Predicted Answer: {pred}")
    else:
        st.warning("Please enter a question and all five choices.")
