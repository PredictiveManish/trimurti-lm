"""
Step 2: Model configuration
"""

from dataclasses import dataclass
from transformers import GPT2Config

@dataclass
class ModelConfig:
    # Model architecture
    vocab_size: int = 8000  # Updated from tokenizer
    n_positions: int = 256  # Context length
    n_embd: int = 512       # Hidden size
    n_layer: int = 8        # Number of layers
    n_head: int = 8         # Attention heads
    n_inner: int = 1024     # FFN dimension
    
    # Training - REALISTIC VALUES
    batch_size: int = 8     # Per GPU batch size
    gradient_accumulation: int = 4  # Effective batch = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 20000  # ~8-9 epochs, NOT 50000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Data
    train_file: str = "./final_corpus/multilingual_corpus_train.txt"
    val_file: str = "./final_corpus/multilingual_corpus_val.txt"
    tokenizer_path: str = "./final_corpus/multilingual_spm.model"
    
    # Checkpoints
    output_dir: str = "./checkpoints"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Mixed precision
    fp16: bool = True
    
    def __post_init__(self):
        print(f"\nModel Configuration (REALISTIC):")
        print(f"  Parameters: ~{self.total_params:.1f}M")
        print(f"  Hidden size: {self.n_embd}")
        print(f"  Layers: {self.n_layer}")
        print(f"  Context length: {self.n_positions}")
        print(f"  Effective batch: {self.effective_batch_size}")
        print(f"  Total steps: {self.total_steps} (~8-9 epochs)")
        print(f"  Learning rate: {self.learning_rate}")
    
    @property
    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation
    
    @property
    def total_params(self):
        # Rough estimate
        embedding = self.vocab_size * self.n_embd
        attention = 4 * self.n_embd * self.n_embd
        ffn = 2 * self.n_embd * self.n_inner
        ln = 2 * self.n_embd
        per_layer = attention + ffn + ln
        total = embedding + (self.n_layer * per_layer)
        return total / 1e6  # Millions