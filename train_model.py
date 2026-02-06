"""
Step 3: STREAMLINED Training - Minimal, Fast
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
import sentencepiece as spm
from tqdm import tqdm
import time

# ===== CONFIG =====
CONFIG = {
    'train_file': './final_corpus/multilingual_corpus_train.txt',
    'val_file': './final_corpus/multilingual_corpus_val.txt',
    'tokenizer_path': './final_corpus/multilingual_spm.model',
    
    # Tiny model for fast training
    'n_positions': 128,
    'n_embd': 256,
    'n_layer': 4,
    'n_head': 4,
    'n_inner': 512,
    
    # Training
    'batch_size': 2,        # Small batch for 4GB
    'grad_accum': 8,        # Effective batch = 16
    'learning_rate': 2e-4,
    'total_steps': 5000,    # Train for 5000 steps only
    'save_every': 1000,
}

class SimpleDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        print("Loading data...")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Tokenize all at once
        self.examples = []
        for line in tqdm(lines[:600000], desc="Tokenizing"):  # Use only 50K lines
            tokens = tokenizer.encode(line)
            if len(tokens) > 10:
                if len(tokens) > block_size:
                    tokens = tokens[:block_size]
                else:
                    tokens = tokens + [0] * (block_size - len(tokens))
                self.examples.append(tokens)
        
        print(f"Created {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

def train_streamlined():
    print("\n" + "="*60)
    print("STREAMLINED TRAINING - FASTEST POSSIBLE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(CONFIG['tokenizer_path'])
    vocab_size = tokenizer.get_piece_size()
    
    # Create tiny model
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=CONFIG['n_positions'],
        n_embd=CONFIG['n_embd'],
        n_layer=CONFIG['n_layer'],
        n_head=CONFIG['n_head'],
        n_inner=CONFIG['n_inner'],
        pad_token_id=0,
    )
    
    model = GPT2LMHeadModel(config)
    model.to(device)
    model.train()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Create dataset (small)
    dataset = SimpleDataset(CONFIG['train_file'], tokenizer, CONFIG['n_positions'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"Training steps: {CONFIG['total_steps']}")
    print(f"Estimated time: {CONFIG['total_steps']*0.3/3600:.1f} hours\n")
    
    # Training loop
    global_step = 0
    accumulation_steps = 0
    start_time = time.time()
    
    while global_step < CONFIG['total_steps']:
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss / CONFIG['grad_accum']
            
            # Backward
            loss.backward()
            accumulation_steps += 1
            
            # Gradient accumulation
            if accumulation_steps == CONFIG['grad_accum']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                accumulation_steps = 0
                
                # Print progress
                if global_step % 100 == 0:
                    elapsed = time.time() - start_time
                    steps_per_second = global_step / elapsed
                    remaining = (CONFIG['total_steps'] - global_step) / steps_per_second
                    
                    print(f"Step {global_step}/{CONFIG['total_steps']} | "
                          f"Loss: {loss.item()*CONFIG['grad_accum']:.3f} | "
                          f"Remaining: {remaining/3600:.1f}h")
                
                # Save checkpoint
                if global_step % CONFIG['save_every'] == 0:
                    save_path = f"./checkpoints_tiny/step{global_step}"
                    model.save_pretrained(save_path)
                    print(f"Saved checkpoint: {save_path}")
                
                # Stop if reached total steps
                if global_step >= CONFIG['total_steps']:
                    break
    
    print(f"\nTraining completed in {(time.time()-start_time)/3600:.2f} hours")
    
    # Save final model
    model.save_pretrained("./checkpoints_tiny/final")
    print("Final model saved to ./checkpoints_tiny/final")

if __name__ == "__main__":
    train_streamlined()