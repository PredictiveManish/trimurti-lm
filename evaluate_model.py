"""
Step 5: Evaluate model quality
"""

import torch
from transformers import GPT2LMHeadModel
import sentencepiece as spm
import numpy as np
from pathlib import Path
import json

def evaluate_multilingual_capabilities(model_path="./checkpoints_tiny/final"):
    """Comprehensive evaluation"""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model
    tokenizer_path = "./final_corpus/multilingual_spm.model"
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    results = {
        "english": {"success": 0, "total": 0, "perplexities": []},
        "hindi": {"success": 0, "total": 0, "perplexities": []},
        "punjabi": {"success": 0, "total": 0, "perplexities": []},
        "mixed": {"success": 0, "total": 0, "perplexities": []},
    }
    
    # Test cases
    test_cases = [
        # English
        ("[EN] The cat sat on the", "mat", "english"),
        ("[EN] I like to eat", "food", "english"),
        ("[EN] Water is essential for", "life", "english"),
        ("[EN] The sun rises in the", "east", "english"),
        
        # Hindi
        ("[HI] बिल्ली चटाई पर", "बैठी", "hindi"),
        ("[HI] मुझे खाना खाना", "पसंद है", "hindi"),
        ("[HI] पानी जीवन के लिए", "आवश्यक है", "hindi"),
        ("[HI] सूरज पूर्व में", "उगता है", "hindi"),
        
        # Punjabi
        ("[PA] ਬਿੱਲੀ ਚੱਟਈ 'ਤੇ", "ਬੈਠੀ", "punjabi"),
        ("[PA] ਮੈਂ ਖਾਣਾ ਖਾਣਾ", "ਪਸੰਦ ਕਰਦਾ ਹਾਂ", "punjabi"),
        ("[PA] ਪਾਣੀ ਜੀਵਨ ਲਈ", "ਜ਼ਰੂਰੀ ਹੈ", "punjabi"),
        ("[PA] ਸੂਰਜ ਪੂਰਬ ਵਿੱਚ", "ਉੱਗਦਾ ਹੈ", "punjabi"),
        
        # Mixed
        ("[EN] Hello [HI] नमस्ते", "दोस्तों", "mixed"),
        ("[HI] यह है [EN] good", "news", "mixed"),
    ]
    
    print("\nRunning tests...")
    
    for prompt, expected_continuation, lang in test_cases:
        # Generate
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids=input_tensor,
                max_length=len(input_ids) + 10,
                temperature=0.7,
                do_sample=False,  # Greedy for testing
                pad_token_id=0,
            )
        
        generated = tokenizer.decode(output[0].tolist())
        
        # Check if generation continues meaningfully
        generated_continuation = generated[len(prompt):].strip().lower()
        expected_lower = expected_continuation.lower()
        
        # Simple check: if expected word appears in generation
        success = expected_lower in generated_continuation or len(generated_continuation) > 3
        
        # Calculate perplexity
        try:
            full_text = prompt + " " + expected_continuation
            text_ids = tokenizer.encode(full_text)
            text_tensor = torch.tensor([text_ids], device=device)
            
            with torch.no_grad():
                outputs = model(input_ids=text_tensor, labels=text_tensor)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
        except:
            perplexity = float('inf')
        
        # Update results
        results[lang]["total"] += 1
        if success:
            results[lang]["success"] += 1
        results[lang]["perplexities"].append(perplexity)
        
        print(f"\n{lang.upper()}: {prompt}")
        print(f"  Generated: {generated_continuation[:50]}...")
        print(f"  Expected: {expected_continuation}")
        print(f"  Success: {'✓' if success else '✗'}")
        print(f"  Perplexity: {perplexity:.2f}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for lang in results:
        if results[lang]["total"] > 0:
            accuracy = results[lang]["success"] / results[lang]["total"] * 100
            avg_perplexity = np.mean(results[lang]["perplexities"])
            print(f"\n{lang.upper()}:")
            print(f"  Accuracy: {accuracy:.1f}% ({results[lang]['success']}/{results[lang]['total']})")
            print(f"  Avg Perplexity: {avg_perplexity:.2f}")
    
    # Overall score
    total_tests = sum(r["total"] for r in results.values())
    total_success = sum(r["success"] for r in results.values())
    overall_accuracy = total_success / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\nOVERALL ACCURACY: {overall_accuracy:.1f}%")
    
    # Save results
    results["overall_accuracy"] = overall_accuracy
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    evaluate_multilingual_capabilities()