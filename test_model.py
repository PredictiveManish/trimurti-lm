"""
Step 4: Test your trained multilingual model
"""

import torch
from transformers import GPT2LMHeadModel
import sentencepiece as spm
import os
from pathlib import Path

class MultilingualModel:
    def __init__(self, model_path="./checkpoints_tiny/final"):
        print("="*60)
        print("LOADING MULTILINGUAL MODEL")
        print("="*60)
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found at: {model_path}")
            print("Available checkpoints:")
            checkpoints = list(Path("./checkpoints_tiny").glob("checkpoint-*"))
            checkpoints += list(Path("./checkpoints_tiny").glob("step*"))
            checkpoints += list(Path("./checkpoints_tiny").glob("final"))
            
            for cp in checkpoints:
                if cp.is_dir():
                    print(f"  - {cp}")
            
            if checkpoints:
                model_path = str(checkpoints[-1])  # Use most recent
                print(f"Using: {model_path}")
            else:
                raise FileNotFoundError("No checkpoints found!")
        
        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer", "spiece.model")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = "./final_corpus/multilingual_spm.model"
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on: {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
        print("="*60)
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
        """Generate text from prompt"""
        # Add language tag if missing
        if not any(prompt.startswith(tag) for tag in ['[EN]', '[HI]', '[PA]']):
            # Try to detect language
            if any(char in prompt for char in 'अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह'):
                prompt = f"[HI] {prompt}"
            elif any(char in prompt for char in 'ਅਆਇਈਉਊਏਐਓਔਕਖਗਘਚਛਜਝਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸ਼ਸਹ'):
                prompt = f"[PA] {prompt}"
            else:
                prompt = f"[EN] {prompt}"
        
        # Encode
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_tensor,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_id() if self.tokenizer.pad_id() > 0 else 0,
                eos_token_id=self.tokenizer.eos_id() if self.tokenizer.eos_id() > 0 else 2,
                repetition_penalty=1.1,
            )
        
        # Decode
        generated = self.tokenizer.decode(output[0].tolist())
        
        # Clean up (remove prompt if it's repeated)
        if generated.startswith(prompt):
            result = generated[len(prompt):].strip()
        else:
            result = generated
        
        return result
    
    def batch_generate(self, prompts, **kwargs):
        """Generate for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def calculate_perplexity(self, text):
        """Calculate perplexity of given text"""
        input_ids = self.tokenizer.encode(text)
        if len(input_ids) < 2:
            return float('inf')
        
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, labels=input_tensor)
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    
    def interactive_mode(self):
        """Interactive chat with model"""
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("Enter prompts in any language (add [EN], [HI], [PA] tags)")
        print("Commands: /temp X, /len X, /quit, /help")
        print("="*60)
        
        temperature = 0.7
        max_length = 100
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        break
                    elif user_input == '/help':
                        print("Commands:")
                        print("  /temp X - Set temperature (0.1 to 2.0)")
                        print("  /len X  - Set max length (20 to 500)")
                        print("  /quit   - Exit")
                        print("  /help   - Show this help")
                        continue
                    elif user_input.startswith('/temp'):
                        try:
                            temp = float(user_input.split()[1])
                            if 0.1 <= temp <= 2.0:
                                temperature = temp
                                print(f"Temperature set to {temperature}")
                            else:
                                print("Temperature must be between 0.1 and 2.0")
                        except:
                            print("Usage: /temp 0.7")
                        continue
                    elif user_input.startswith('/len'):
                        try:
                            length = int(user_input.split()[1])
                            if 20 <= length <= 500:
                                max_length = length
                                print(f"Max length set to {max_length}")
                            else:
                                print("Length must be between 20 and 500")
                        except:
                            print("Usage: /len 100")
                        continue
                
                # Generate response
                print("Model: ", end="", flush=True)
                response = self.generate(user_input, max_length=max_length, temperature=temperature)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def run_tests():
    """Run comprehensive tests"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL TESTS")
    print("="*60)
    
    # Load model
    model = MultilingualModel()
    
    # Test prompts by language
    test_suites = {
        "English": [
            "[EN] The weather today is",
            "[EN] I want to learn",
            "[EN] Artificial intelligence",
            "[EN] The capital of India is",
            "[EN] Once upon a time",
        ],
        "Hindi": [
            "[HI] आज का मौसम",
            "[HI] मैं सीखना चाहता हूं",
            "[HI] कृत्रिम बुद्धिमत्ता",
            "[HI] भारत की राजधानी है",
            "[HI] एक बार की बात है",
        ],
        "Punjabi": [
            "[PA] ਅੱਜ ਦਾ ਮੌਸਮ",
            "[PA] ਮੈਂ ਸਿੱਖਣਾ ਚਾਹੁੰਦਾ ਹਾਂ",
            "[PA] ਕ੍ਰਿਤਰਿਮ ਬੁੱਧੀ",
            "[PA] ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਹੈ",
            "[PA] ਇੱਕ ਵਾਰ ਦੀ ਗੱਲ ਹੈ",
        ],
        "Language Switching": [
            "[EN] Hello [HI] नमस्ते",
            "[HI] यह अच्छा है [EN] this is good",
            "[PA] ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ [EN] Hello everyone",
        ],
        "Code Mixing": [
            "Hello दुनिया",  # No tag, should auto-detect
            "मेरा name है",  # Hindi + English
            "Today मौसम is good",  # English + Hindi
        ]
    }
    
    for suite_name, prompts in test_suites.items():
        print(f"\n{'='*40}")
        print(f"{suite_name.upper()} TESTS")
        print('='*40)
        
        for i, prompt in enumerate(prompts):
            print(f"\nTest {i+1}:")
            print(f"Prompt: {prompt}")
            
            # Generate
            response = model.generate(prompt, max_length=50, temperature=0.7)
            print(f"Response: {response}")
            
            # Calculate perplexity
            try:
                perplexity = model.calculate_perplexity(response)
                print(f"Perplexity: {perplexity:.2f}")
            except:
                pass
            
            print("-" * 40)

def benchmark_model():
    """Benchmark model performance"""
    print("\n" + "="*60)
    print("MODEL BENCHMARK")
    print("="*60)
    
    model = MultilingualModel()
    
    import time
    
    # Test generation speed
    test_prompt = "[EN] The quick brown fox"
    
    times = []
    for _ in range(10):
        start = time.time()
        model.generate(test_prompt, max_length=50)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average generation time (50 tokens): {avg_time:.3f}s")
    print(f"Tokens per second: {50/avg_time:.1f}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU Memory reserved: {memory_reserved:.2f} GB")

def create_web_interface():
    """Simple web interface for the model"""
    html_code = """
<!DOCTYPE html>
<html>
<head>
    <title>Multilingual LM Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { display: flex; flex-direction: column; gap: 20px; }
        textarea { width: 100%; height: 100px; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        .output { border: 1px solid #ccc; padding: 15px; min-height: 100px; background: #f9f9f9; }
        .language-tag { display: inline-block; margin: 5px; padding: 5px 10px; background: #e0e0e0; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multilingual Language Model Demo</h1>
        
        <div>
            <strong>Language:</strong>
            <span class="language-tag" onclick="setLanguage('[EN] ')">English</span>
            <span class="language-tag" onclick="setLanguage('[HI] ')">Hindi</span>
            <span class="language-tag" onclick="setLanguage('[PA] ')">Punjabi</span>
        </div>
        
        <textarea id="prompt" placeholder="Enter your prompt here..."></textarea>
        
        <div>
            <label>Temperature: <input type="range" id="temp" min="0.1" max="2.0" step="0.1" value="0.7"></label>
            <label>Max Length: <input type="number" id="maxlen" min="20" max="500" value="100"></label>
        </div>
        
        <button onclick="generate()">Generate</button>
        
        <div class="output" id="output">Response will appear here...</div>
    </div>
    
    <script>
        function setLanguage(tag) {
            document.getElementById('prompt').value = tag;
        }
        
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const temp = document.getElementById('temp').value;
            const maxlen = document.getElementById('maxlen').value;
            
            document.getElementById('output').innerHTML = 'Generating...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, temp, maxlen})
                });
                
                const data = await response.json();
                document.getElementById('output').innerHTML = data.response;
            } catch (error) {
                document.getElementById('output').innerHTML = 'Error: ' + error;
            }
        }
    </script>
</body>
</html>
    """
    
    # Save HTML
    with open("model_demo.html", "w", encoding="utf-8") as f:
        f.write(html_code)
    
    print("Web interface saved as model_demo.html")
    print("To use it, you need a backend server (see create_server.py)")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("MULTILINGUAL MODEL PLAYGROUND")
    print("="*60)
    print("\nOptions:")
    print("1. Interactive chat")
    print("2. Run comprehensive tests")
    print("3. Benchmark model")
    print("4. Create web interface")
    print("5. Quick generation test")
    print("6. Exit")
    
    # Load model once
    model = None
    
    while True:
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                if model is None:
                    model = MultilingualModel()
                model.interactive_mode()
            
            elif choice == '2':
                run_tests()
            
            elif choice == '3':
                benchmark_model()
            
            elif choice == '4':
                create_web_interface()
            
            elif choice == '5':
                if model is None:
                    model = MultilingualModel()
                
                prompt = input("Enter prompt: ").strip()
                if prompt:
                    response = model.generate(prompt)
                    print(f"\nResponse: {response}")
            
            elif choice == '6':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-6.")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()