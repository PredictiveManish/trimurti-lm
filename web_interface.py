
"""
Simple web interface using Gradio
"""

import torch
from transformers import GPT2LMHeadModel
import sentencepiece as spm
import gradio as gr
import os

class SimpleModel:
    def __init__(self, model_path="./checkpoints_tiny/final"):
        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer", "spiece.model")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = "./final_corpus/multilingual_spm.model"
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.95):
        # Add language tag if missing
        if not any(prompt.startswith(tag) for tag in ['[EN]', '[HI]', '[PA]']):
            prompt = f"[EN] {prompt}"
        
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_tensor,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=0,
                repetition_penalty=1.1,
            )
        
        generated = self.tokenizer.decode(output[0].tolist())
        if generated.startswith(prompt):
            return generated[len(prompt):].strip()
        return generated

def create_gradio_interface():
    # Initialize model
    model = SimpleModel()
    
    def generate_text(prompt, max_length, temperature, top_p):
        try:
            result = model.generate(prompt, int(max_length), float(temperature), float(top_p))
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="Multilingual LM Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌍 Multilingual Language Model")
        gr.Markdown("Generate text in English, Hindi, or Punjabi")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Enter prompt",
                    placeholder="Start with [EN], [HI], or [PA] for language...",
                    lines=3
                )
                
                with gr.Row():
                    max_length = gr.Slider(20, 500, value=100, label="Max Length")
                    temperature = gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, label="Top-p")
                
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(label="Generated Text", lines=10)
        
        # Examples
        gr.Examples(
            examples=[
                ["[EN] The weather today is"],
                ["[HI] आज का मौसम"],
                ["[PA] ਅੱਜ ਦਾ ਮੌਸਮ"],
                ["[EN] Once upon a time in India"],
                ["[HI] भारत एक महान देश है"],
                ["[PA] ਭਾਰਤ ਇੱਕ ਮਹਾਨ ਦੇਸ਼ ਹੈ"],
            ],
            inputs=prompt,
            label="Try these examples:"
        )
        
        # Button click
        generate_btn.click(
            fn=generate_text,
            inputs=[prompt, max_length, temperature, top_p],
            outputs=output
        )
        
        # Also generate on Enter key
        prompt.submit(
            fn=generate_text,
            inputs=[prompt, max_length, temperature, top_p],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    # Install gradio if not installed
    try:
        import gradio as gr
    except ImportError:
        print("Installing gradio...")
        import subprocess
        subprocess.check_call(["pip", "install", "gradio"])
        import gradio as gr
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to get public link
        debug=False
    )