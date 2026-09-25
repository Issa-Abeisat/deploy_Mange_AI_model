import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Generative AI Demo")
st.write("Model: google/flan-t5-small")

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

try:
    tokenizer, model = load_model_and_tokenizer()

    prompt = st.text_area("Prompt", placeholder="Ask something...")

    if st.button("Generate"):
        if prompt.strip():
            with st.spinner("Generating response..."):
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, max_new_tokens=100)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.write(response)
        else:
            st.warning("Please enter a prompt.")
            
except Exception as e:
    st.error(f"Error loading model: {e}")
