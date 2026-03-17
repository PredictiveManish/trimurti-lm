# Trimurti-LM

A compact multilingual language model supporting **English**, **Hindi**, and **Punjabi**. Built on GPT2 architecture with efficient training on parallel translation corpora.

## Overview

Trimurti-LM is a lightweight transformer-based language model trained on multilingual text data. It uses language-specific tags (`[EN]`, `[HI]`, `[PA]`) to generate text in three languages.

| Property | Value |
|----------|-------|
| Architecture | GPT2 (Decoder-only Transformer) |
| Parameters | ~4.7M |
| Languages | English, Hindi, Punjabi |
| Tokenizer | SentencePiece (8K vocab) |
| Max Context | 128 tokens |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Trimurti-LM Architecture             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input Text ──► SentencePiece Tokenizer ──► Embeddings  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │           GPT2 Transformer Blocks (×4)          │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │  Multi-Head Self-Attention (4 heads)    │    │    │
│  │  │  - Query, Key, Value projections        │    │    │
│  │  │  - Attention dropout (0.1)              │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  │                        │                        │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │  Feed-Forward Network (GELU)            │    │    │
│  │  │  - Hidden: 512                          │    │    │
│  │  │  - Dropout: 0.1                         │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  │                        │                        │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │  LayerNorm → Residual Connections       │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  Linear Head ──► Softmax ──► Next Token Prediction      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 8000 | Tokenizer vocabulary size |
| `n_positions` | 128 | Maximum context length |
| `n_embd` | 256 | Embedding/hidden dimension |
| `n_layer` | 4 | Number of transformer layers |
| `n_head` | 4 | Number of attention heads |
| `n_inner` | 512 | Feed-forward hidden dimension |
| `activation_function` | gelu_new | GELU activation |
| `attn_pdrop` | 0.1 | Attention dropout |
| `embd_pdrop` | 0.1 | Embedding dropout |
| `resid_pdrop` | 0.1 | Residual dropout |

### Parameter Count

```
Embedding Layer:     vocab_size × n_embd = 8000 × 256 = 2,048,000
4× Transformer Layers:
  - Attention:        3 × (n_embd × n_embd) = 3 × 65,536 = 196,608
  - Output:          n_embd × n_embd = 65,536
  - FFN:             n_embd × n_inner + n_inner × n_embd = 262,144 + 262,144 = 524,288
  - LayerNorm:       2 × n_embd × 4 = 2,048
  Total per layer:    ~850K
  4 layers:          ~3.4M
Output Head:         n_embd × vocab_size = 2,048,000
─────────────────────────────────────────────────
Total Parameters:    ~4.7M
```

---

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 8 | Per-GPU batch size |
| `gradient_accumulation` | 4 | Gradient accumulation steps |
| `effective_batch_size` | 32 | Total batch size |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `warmup_steps` | 1000 | LR warmup steps |
| `total_steps` | 20000 | Total training steps |
| `weight_decay` | 0.1 | Weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `save_steps` | 1000 | Checkpoint frequency |
| `eval_steps` | 500 | Evaluation frequency |
| `fp16` | True | Mixed precision training |

### Training Data Distribution

| Language | Tag | Ratio |
|----------|-----|-------|
| English | `[EN]` | 40% |
| Hindi | `[HI]` | 40% |
| Punjabi | `[PA]` | 20% |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trimurti-lm.git
cd trimurti-lm

# Install dependencies
pip install torch transformers sentencepiece tqdm gradio pandas numpy ftfy langdetect
```

---

## Project Structure

```
trimurti-lm/
├── data/                      # Raw parallel data
│   ├── main.py               # Data extraction
│   ├── en-hi.csv             # English-Hindi pairs
│   └── en-pa.csv             # English-Punjabi pairs
│
├── final_corpus/             # Preprocessed corpus
│   ├── multilingual_corpus.txt
│   ├── multilingual_corpus_train.txt
│   ├── multilingual_corpus_val.txt
│   ├── multilingual_spm.model    # SentencePiece model
│   └── multilingual_spm.vocab
│
├── checkpoints_tiny/         # Trained model checkpoints
│   ├── step1000/
│   ├── step2000/
│   └── final/                # Final checkpoint
│       ├── config.json
│       ├── generation_config.json
│       └── model.safetensors
│
├── train_model.py            # Training script
├── test_model.py             # Inference & testing
├── evaluate_model.py         # Evaluation
├── preprocess.py             # Data preprocessing
├── model_config.py           # Configuration dataclass
├── web_interface.py           # Gradio UI
└── model_demo.html           # HTML demo
```

---

## Usage

### 1. Data Preprocessing

```bash
python preprocess.py
```

Creates the multilingual corpus and trains the SentencePiece tokenizer.

### 2. Training

```bash
python train_model.py
```

Training parameters can be modified in `model_config.py` or directly in the script.

### 3. Evaluation

```bash
python evaluate_model.py
```

Evaluates perplexity and accuracy across all supported languages.

### 4. Inference

```bash
python test_model.py
```

Provides interactive mode with commands:
- `/temp X` - Set temperature (0.1-2.0)
- `/len X` - Set max length (20-500)
- `/quit` - Exit

### 5. Web Interface

```bash
python web_interface.py
```

Launches a Gradio web interface at `http://localhost:7860`.

---

## Evaluation Results

| Language | Success Rate | Avg Perplexity |
|----------|-------------|----------------|
| English | 100% (4/4) | 42.29 |
| Hindi | 100% (4/4) | 50.56 |
| Punjabi | 100% (4/4) | 63.42 |
| Mixed | 100% (2/2) | 94.77 |
| **Overall** | **100%** | **62.76** |

---

## Generation Examples

```python
from test_model import MultilingualModel

model = MultilingualModel("checkpoints_tiny/final")

# English generation
print(model.generate("[EN] The weather is"))

# Hindi generation
print(model.generate("[HI] आज मौसम"))

# Punjabi generation
print(model.generate("[PA] ਅੱਜ ਹਵਾ"))
```

---

## License

MIT License

---

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [SentencePiece](https://github.com/google/sentencepiece)
