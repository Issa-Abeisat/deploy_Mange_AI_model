import streamlit as st
from transformers import pipeline

st.title("Generative AI Demo")
st.write("Model: google/flan-t5-small")

@st.cache_resource
def load_model():
    # Pass both the explicit task string and the model name
    return pipeline("text2text-generation", model="google/flan-t5-small")

generator = load_model()

prompt = st.text_area("Prompt", placeholder="Ask something...")

if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Generating response..."):
            output = generator(prompt, max_new_tokens=100)
            st.write(output[0]["generated_text"])
    else:
        st.warning("Please enter a prompt.")
