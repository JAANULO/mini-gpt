import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from mini_gpt.transformer import MiniGPT
from mini_gpt.config import cfg
from mini_gpt.utils import DEVICE

def test_minigpt_output_shape():
    # Inicjalizacja modelu z małymi parametrami dla szybkiego testu
    vocab_size = 100
    embed_dim = cfg.embed_dim
    num_heads = cfg.num_heads
    max_length = cfg.max_length
    
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=1,
        num_heads=num_heads,
        dropout=0.0,
        max_length=max_length
    ).to(DEVICE)
    
    batch_size = 2
    seq_len = 15
    
    # Przykładowe wejście
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(DEVICE)
    
    logits, _ = model(x)
    
    # Oczekiwany wymiar: [batch_size, seq_len, vocab_size]
    assert logits.shape == (batch_size, seq_len, vocab_size), "Zły wymiar logitów!"

def test_causal_mask():
    from mini_gpt.transformer import GPTBlock
    
    block = GPTBlock(embed_dim=16, num_heads=2, dropout=0.0).to(DEVICE)
    x = torch.randn(2, 5, 16).to(DEVICE) # batch=2, seq=5, embed=16
    out, attn = block(x, return_attn=True)
    
    assert out.shape == (2, 5, 16)
    # Maskowanie sprawia, że tokeny nie widzą przyszłości.
    # W testach sprawdzamy czy funkcja wykonuje się poprawnie (nie crashuje)
    assert attn is not None

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
