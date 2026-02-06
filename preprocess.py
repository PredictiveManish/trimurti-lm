"""
Step 1: Create final shuffled corpus and train tokenizer
"""

import random
from pathlib import Path
import sentencepiece as spm
from collections import defaultdict
import numpy as np

def create_final_corpus(en_file, hi_file, pa_file, output_file, lang_ratios=None):
    """
    Create final multilingual corpus with language tags
    
    Args:
        en_file: English sentences file
        hi_file: Hindi sentences file
        pa_file: Punjabi sentences file
        output_file: Output corpus file
        lang_ratios: Dict with language ratios, {'en': 0.4, 'hi': 0.4, 'pa': 0.2}
    """
    
    print("Creating final corpus...")
    
    # Default ratios
    if lang_ratios is None:
        lang_ratios = {'en': 0.4, 'hi': 0.4, 'pa': 0.2}
    
    # Read sentences
    with open(en_file, 'r', encoding='utf-8') as f:
        en_sentences = [line.strip() for line in f if line.strip()]
    
    with open(hi_file, 'r', encoding='utf-8') as f:
        hi_sentences = [line.strip() for line in f if line.strip()]
    
    with open(pa_file, 'r', encoding='utf-8') as f:
        pa_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(en_sentences):,} English sentences")
    print(f"Loaded {len(hi_sentences):,} Hindi sentences")
    print(f"Loaded {len(pa_sentences):,} Punjabi sentences")
    
    # Determine sample sizes
    total_target = min(len(en_sentences), len(hi_sentences), len(pa_sentences)) * 2
    target_counts = {
        'en': int(total_target * lang_ratios['en']),
        'hi': int(total_target * lang_ratios['hi']),
        'pa': int(total_target * lang_ratios['pa'])
    }
    
    print(f"\nTarget counts:")
    print(f"  English: {target_counts['en']:,}")
    print(f"  Hindi: {target_counts['hi']:,}")
    print(f"  Punjabi: {target_counts['pa']:,}")
    
    # Sample sentences
    sampled_en = random.sample(en_sentences, min(target_counts['en'], len(en_sentences)))
    sampled_hi = random.sample(hi_sentences, min(target_counts['hi'], len(hi_sentences)))
    sampled_pa = random.sample(pa_sentences, min(target_counts['pa'], len(pa_sentences)))
    
    # Create corpus with language tags
    corpus = []
    for sent in sampled_en:
        corpus.append(f"[EN] {sent}")
    for sent in sampled_hi:
        corpus.append(f"[HI] {sent}")
    for sent in sampled_pa:
        corpus.append(f"[PA] {sent}")
    
    # Shuffle
    random.shuffle(corpus)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(f"{line}\n")
    
    # Create train/validation split (95/5)
    val_size = int(len(corpus) * 0.05)
    train_corpus = corpus[val_size:]
    val_corpus = corpus[:val_size]
    
    train_file = output_file.replace('.txt', '_train.txt')
    val_file = output_file.replace('.txt', '_val.txt')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_corpus:
            f.write(f"{line}\n")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for line in val_corpus:
            f.write(f"{line}\n")
    
    # Statistics
    print(f"\nCorpus created:")
    print(f"  Total sentences: {len(corpus):,}")
    print(f"  Training sentences: {len(train_corpus):,}")
    print(f"  Validation sentences: {len(val_corpus):,}")
    
    # Language distribution
    lang_counts = defaultdict(int)
    for line in corpus:
        if line.startswith('[EN]'):
            lang_counts['en'] += 1
        elif line.startswith('[HI]'):
            lang_counts['hi'] += 1
        elif line.startswith('[PA]'):
            lang_counts['pa'] += 1
    
    print(f"\nLanguage distribution:")
    for lang, count in lang_counts.items():
        percentage = (count / len(corpus)) * 100
        print(f"  {lang.upper()}: {count:,} ({percentage:.1f}%)")
    
    return train_file, val_file

def train_tokenizer(corpus_file, vocab_size=8000, model_prefix='multilingual'):
    """
    Train SentencePiece tokenizer
    """
    print(f"\nTraining SentencePiece tokenizer with vocab size {vocab_size}...")
    
    # First, create a version without language tags for tokenizer training
    temp_corpus = 'temp_tokenizer_corpus.txt'
    with open(corpus_file, 'r', encoding='utf-8') as f_in, \
         open(temp_corpus, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Remove language tags for tokenizer training
            if line.startswith('[EN]'):
                f_out.write(line[5:])  # Remove "[EN] "
            elif line.startswith('[HI]'):
                f_out.write(line[5:])  # Remove "[HI] "
            elif line.startswith('[PA]'):
                f_out.write(line[5:])  # Remove "[PA] "
            else:
                f_out.write(line)
    
    # SentencePiece training parameters
    spm.SentencePieceTrainer.train(
        input=temp_corpus,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,  # Important for multilingual
        model_type='unigram',       # Better for multilingual
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        byte_fallback=True,         # Important for Indic scripts
        split_by_unicode_script=True,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        # Don't use normalization for Indic scripts
        normalization_rule_name='identity',
        seed_sentencepiece_size=1000000,
        num_threads=4
    )
    
    # Load and test tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    
    print(f"Tokenizer trained successfully!")
    print(f"Vocabulary size: {sp.get_piece_size()}")
    
    # Test tokenizer
    test_sentences = [
        "Hello world",  # English
        "नमस्ते दुनिया",  # Hindi
        "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਦੁਨਿਆ"  # Punjabi
    ]
    
    print("\nTokenizer test:")
    for sent in test_sentences:
        tokens = sp.encode_as_pieces(sent)
        ids = sp.encode_as_ids(sent)
        print(f"  '{sent}' -> {tokens} (ids: {ids})")
    
    # Clean up
    Path(temp_corpus).unlink()
    
    return sp

def analyze_tokenizer(sp, corpus_file):
    """Analyze tokenizer coverage"""
    print("\nAnalyzing tokenizer coverage...")
    
    languages = {'en': 0, 'hi': 0, 'pa': 0}
    total_tokens = 0
    lang_tokens = defaultdict(int)
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Sample 1000 sentences per language
    samples_per_lang = 1000
    
    for line in lines:
        if line.startswith('[EN]'):
            lang = 'en'
            text = line[5:].strip()
        elif line.startswith('[HI]'):
            lang = 'hi'
            text = line[5:].strip()
        elif line.startswith('[PA]'):
            lang = 'pa'
            text = line[5:].strip()
        else:
            continue
        
        languages[lang] += 1
        if languages[lang] <= samples_per_lang:
            tokens = sp.encode_as_ids(text)
            total_tokens += len(tokens)
            lang_tokens[lang] += len(tokens)
    
    print(f"Token counts per language (sampled {samples_per_lang} sentences each):")
    for lang in ['en', 'hi', 'pa']:
        avg_tokens = lang_tokens[lang] / samples_per_lang
        print(f"  {lang.upper()}: {avg_tokens:.1f} tokens per sentence")

def main():
    # Configuration
    EN_FILE = r"C:\Users\manis\Desktop\2026-projects\foundational-model\data\extracted_sentences\en.txt"
    HI_FILE = r"C:\Users\manis\Desktop\2026-projects\foundational-model\data\extracted_sentences\hi.txt"       
    PA_FILE = r"C:\Users\manis\Desktop\2026-projects\foundational-model\data\extracted_sentences\pa.txt" 
    
    OUTPUT_DIR = "./final_corpus"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    FINAL_CORPUS = f"{OUTPUT_DIR}/multilingual_corpus.txt"
    TOKENIZER_PREFIX = f"{OUTPUT_DIR}/multilingual_spm"
    
    # Create final corpus
    train_file, val_file = create_final_corpus(
        EN_FILE, HI_FILE, PA_FILE, FINAL_CORPUS,
        lang_ratios={'en': 0.4, 'hi': 0.4, 'pa': 0.2}
    )
    
    # Train tokenizer
    sp = train_tokenizer(train_file, vocab_size=8000, model_prefix=TOKENIZER_PREFIX)
    
    # Analyze tokenizer
    analyze_tokenizer(sp, train_file)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Files created in {OUTPUT_DIR}:")
    print(f"  1. {FINAL_CORPUS} - Full corpus")
    print(f"  2. {train_file} - Training split")
    print(f"  3. {val_file} - Validation split")
    print(f"  4. {TOKENIZER_PREFIX}.model - SentencePiece model")
    print(f"  5. {TOKENIZER_PREFIX}.vocab - Vocabulary")
    print(f"\nNext step: Train the model with train_model.py")

if __name__ == "__main__":
    # Install sentencepiece if not available
    try:
        import sentencepiece as spm
    except ImportError:
        import subprocess
        import sys
        print("Installing sentencepiece...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
        import sentencepiece as spm
    
    main()