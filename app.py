import torch
import gradio as gr
from transformers import pipeline

# 1. Setup device
device_id = 0 if torch.cuda.is_available() else -1

# 2. Load model (task is inferred automatically for T5)
MODEL_NAME = "google/flan-t5-small"

generator = pipeline(
    model=MODEL_NAME,
    device=device_id,
)

print(f"Loaded: {MODEL_NAME}")

# 3. Define inference function
def generate(prompt: str, max_new_tokens: int = 100) -> str:
    """Run one inference call through the loaded model."""
    if not prompt or not prompt.strip():
        return "Please enter a prompt."
    output = generator(prompt, max_new_tokens=max_new_tokens)
    return output[0]["generated_text"]

# 4. Define and launch Gradio UI
demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt", placeholder="Ask something..."),
    outputs=gr.Textbox(label="Model response"),
    title="Generative AI Demo",
    description=f"Model: {MODEL_NAME}",
)

if __name__ == "__main__":
    demo.launch()