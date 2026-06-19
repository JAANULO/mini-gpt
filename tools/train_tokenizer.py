import sentencepiece as spm
import os
import sys
from pathlib import Path

def main():
    corpus_file = 'data/raw_corpus.txt'
    export_dir = Path('exports')
    export_dir.mkdir(exist_ok=True)
    
    model_prefix = str(export_dir / 'tokenizer')
    
    if not os.path.exists(corpus_file):
        print(f"❌ Brak pliku: {corpus_file}. Uruchom najpierw data/prepare_corpus.py")
        sys.exit(1)
        
    print(f"⚙️ Trenowanie tokenizera BPE na korpusie {corpus_file}...")
    
    # 100 tokenów to absolutne minimum na tak mały zbiór jak dane.json
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=100,
        model_type='bpe',
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        character_coverage=1.0, # chcemy objąć 100% znaków w małym zbiorze
    )
    
    print(f"✅ Tokenizer BPE wytrenowany! Zapisano do: {model_prefix}.model oraz .vocab")

if __name__ == '__main__':
    main()
