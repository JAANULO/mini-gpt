import sys
import io
import json
import os
from pathlib import Path

import torch
from mini_gpt.transformer import MiniGPT
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.utils import (
    DEVICE, hash_data, load_cache, load_export, top_k_top_p_sampling
)
from mini_gpt.inference import generate_response

# ============================================================
# SETTINGS
# ============================================================

from mini_gpt.config import cfg

DATA_FILE = cfg.data_file
EXPORT_DIR = Path("exports")
EXPORT_FILE = EXPORT_DIR / "model_export.pt"
CACHE_FILE = EXPORT_DIR / "model_cache.pkl"

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"❌ File not found: '{filepath}'!")
        exit(1)
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("zdania", [])



if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 55)
    print("  MINI-GPT – Chat")
    print("=" * 55 + "\n")

    sentences = load_data(DATA_FILE)
    current_hash = hash_data(DATA_FILE)
    
    tokenizer = Tokenizer()
    loaded = tokenizer.load("exports/tokenizer.model")
    if not loaded:
        print("❌ No trained BPE tokenizer found. Run train.py first!")
        sys.exit(1)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        max_length=cfg.max_length,
    ).to(DEVICE)

    tokenizer_from_cache, cache_ok = load_cache(model, current_hash, CACHE_FILE)
    if cache_ok:
        tokenizer = tokenizer_from_cache
        print("✅ Loaded model from cache!\n")
    else:
        tokenizer_export, export_ok = load_export(model, EXPORT_FILE)
        if export_ok:
            tokenizer = tokenizer_export
            print("✅ Loaded model from export!\n")
        else:
            print("❌ No trained model found. Run train.py first!")
            sys.exit(1)

    model.set_training(False)

    print("\n" + "═" * 55)
    print("  💬 CHAT MODE – model remembers context!")
    print("═" * 55)
    print("  Examples:")
    print("    what is a cat")
    print("    where is warsaw")
    print("    who are you")
    print()
    print("  Commands: /temp 0.1 | /history | /clear | /help | exit")
    print("═" * 55 + "\n")

    temperature = cfg.default_temp
    history = []

    while True:
        try:
            user_input = input("  You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "koniec", "quit"]:
                print("\n  See you! 👋\n")
                break

            if user_input.startswith("/temp"):
                parts = user_input.split()
                if len(parts) == 2:
                    try:
                        temperature = float(parts[1])
                        print(f"  ✅ Temperature: {temperature}\n")
                    except ValueError:
                        print("  ⚠️  Usage: /temp 0.1\n")
                continue

            if user_input == "/history":
                if not history:
                    print("  (no history)\n")
                else:
                    print("\n  📜 Chat history:")
                    for i, (q, a) in enumerate(history, 1):
                        print(f"  {i}. You:   {q}")
                        print(f"     Model: {a}")
                    print()
                continue

            if user_input == "/clear":
                history.clear()
                print("  🗑️  History cleared.\n")
                continue

            if user_input == "/help":
                print("""
  Commands:
    /temp 0.1    → temperature (0.01=always same, 1.0=random)
    /history     → show chat history
    /clear       → clear model memory
    exit         → end chat
                """)
                continue

            response = generate_response(model, tokenizer, user_input, history, temperature)
            print(f"  🤖 Model: {response}\n")

            if response != "...":
                question_clean = user_input.lower().strip().rstrip("?")
                history.append((question_clean, response))

        except KeyboardInterrupt:
            print("\n\n  Program terminated.\n")
            break