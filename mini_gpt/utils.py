import os
import hashlib
import logging
from pathlib import Path
from typing import Any, Tuple, Optional, List

import torch
import torch.nn as nn
import numpy as np

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("mini_gpt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hash_data(file_path: str) -> str:
    """Calculates MD5 hash of the data file to detect changes."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def save_cache(model: nn.Module, tokenizer: Any, file_hash: str, cache_path: Path, export_path: Path) -> None:
    """Saves model PyTorch state and tokenizer to cache."""
    data = {
        "hash": file_hash,
        "tokenizer": tokenizer,
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": getattr(model, 'embed_dim', 128),
            "max_length": getattr(model, 'max_length', 256),
        }
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    logger.info(f"Model saved to cache: {cache_path}")
    export_model(model, tokenizer, file_hash, export_path)

def load_cache(model: nn.Module, file_hash: str, cache_path: Path) -> Tuple[Optional[Any], bool]:
    """Loads model from cache if data hash matches."""
    if not cache_path.exists():
        return None, False

    try:
        data = torch.load(cache_path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        logger.warning(f"Failed to load cache '{cache_path}', error: {e}. Training from scratch.")
        if cache_path.exists():
            cache_path.unlink()
        return None, False

    if data.get("hash") != file_hash:
        logger.info("Data has changed. Training from scratch.")
        return None, False

    state = data["state_dict"]
    new_state = {}
    for k, v in state.items():
        k = k.replace("bloki.", "blocks.")
        k = k.replace("glowa.", "head.")
        new_state[k] = v

    try:
        model.load_state_dict(new_state)
    except Exception as e:
        logger.warning(f"State dict mismatch (probably old model structure): {e}. Training from scratch.")
        return None, False

    return data["tokenizer"], True

def export_model(model: nn.Module, tokenizer: Any, file_hash: str, export_path: Path) -> None:
    """Exports compressed model for GitHub."""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "hash": file_hash,
        "tokenizer": tokenizer,
        "state_dict": {k: v.half() for k, v in model.state_dict().items()},
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": getattr(model, 'embed_dim', 128),
            "max_length": getattr(model, 'max_length', 256),
        }
    }
    torch.save(data, export_path, _use_new_zipfile_serialization=True)
    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    logger.info(f"Exported to '{export_path}': {size_mb:.1f} MB (ready for GitHub)")

def load_export(model: nn.Module, export_path: Path) -> Tuple[Optional[Any], bool]:
    """Loads compressed model."""
    if not export_path.exists():
        logger.warning(f"Export file not found: '{export_path}'")
        return None, False

    try:
        data = torch.load(export_path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        logger.warning(f"Failed to load export '{export_path}', error: {e}. Training from scratch.")
        if export_path.exists():
            export_path.unlink()
        return None, False

    state = {k: v.float() for k, v in data["state_dict"].items()}
    new_state = {}
    for k, v in state.items():
        k = k.replace("bloki.", "blocks.")
        k = k.replace("glowa.", "head.")
        new_state[k] = v
        
    try:
        model.load_state_dict(new_state)
    except Exception as e:
        logger.warning(f"Export state dict mismatch: {e}. Training from scratch.")
        return None, False
        
    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    logger.info(f"Loaded export '{export_path}' ({size_mb:.1f} MB)")
    return data["tokenizer"], True

def top_k_top_p_sampling(logits_in, top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0) -> int:
    """Advanced sampling from logit distribution using pure PyTorch."""
    if isinstance(logits_in, np.ndarray):
        logits = torch.tensor(logits_in, dtype=torch.float32, device=DEVICE)
    else:
        logits = logits_in.clone()

    logits = logits / max(temperature, 0.01)

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        val, _ = torch.topk(logits, k)
        threshold = val[-1]
        logits[logits < threshold] = -float('Inf')

    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Usuń tokeny, których skumulowane prawdopodobieństwo przekracza top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Przesuń by zawsze zachować chociaż pierwszy odrzucony token nad progiem
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)

    if temperature < 0.05:
        return int(torch.argmax(probs).item())
    
    return int(torch.multinomial(probs, num_samples=1).item())

def generate_text(model: nn.Module, tokenizer: Any, start_text: str, max_tokens: int = 60, temperature: float = 1.0) -> str:
    """Generates text from the given starting prompt."""
    ids = tokenizer.encode(start_text)

    with torch.no_grad():
        for _ in range(max_tokens):
            max_len = getattr(model, 'max_length', 256)
            input_ids = ids[-max_len:]
            
            # Convert to tensor and unsqueeze
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            logits, _ = model.forward(input_tensor)
            
            # Use raw tensor directly on GPU
            last_logits = logits[0, -1]
            next_id = top_k_top_p_sampling(
                last_logits, top_k=50, top_p=0.9, temperature=temperature
            )

            ids.append(next_id)
            
            current_text = tokenizer.decode(ids)
            if "koniec" in current_text[-10:]:
                break

    return tokenizer.decode(ids)
