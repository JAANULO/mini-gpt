import torch
import torch.nn as nn
import numpy as np

from mini_gpt.utils import DEVICE, logger

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

class GPTBlock(nn.Module):
    """Pojedynczy blok Transformera (Causal Self-Attention + Feed Forward).
    
    Args:
        embed_dim (int): Wymiar osadzeń (embeddings).
        num_heads (int): Liczba głów w mechanizmie Multi-Head Attention.
        dropout (float): Wartość dropoutu zapobiegająca przeuczeniu.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Przejście danych w przód (Forward Pass) przez blok Transformera.
        
        Aplikuje maskę przyczynową (Causal Mask), dzięki czemu dany token 
        może zwracać uwagę tylko na siebie i tokeny poprzedzające.
        
        Args:
            x (torch.Tensor): Tensor wejściowy o wymiarach (Batch, Sequence, Embed).
            return_attn (bool): Czy zwrócić wagi uwagi mechanizmu Attention.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Tensor wyjściowy oraz opcjonalnie wagi uwagi.
        """
        T = x.shape[1]
        mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        x2 = self.ln1(x)
        x2, attn_weights = self.attn(x2, x2, x2, attn_mask=mask, is_causal=True,
                                     need_weights=return_attn, average_attn_weights=False)
        x = x + x2
        x = x + self.ff(self.ln2(x))

        if return_attn:
            return x, attn_weights
        return x, None


class MiniGPT(nn.Module):
    """Architektura Mini-GPT oparta na dekoderze z modelu Transformer.
    
    Args:
        vocab_size (int): Rozmiar słownika (liczba unikalnych tokenów BPE).
        embed_dim (int): Wymiar przestrzeni ukrytej modelu.
        num_layers (int): Liczba warstw (bloków Transformera).
        num_heads (int): Liczba głów dla mechanizmu uwagi.
        dropout (float): Dropout dla stabilności treningu.
        max_length (int): Maksymalna obsługiwana długość sekwencji.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_layers: int = 4,
                 num_heads: int = 4, dropout: float = 0.1, max_length: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.tok_emb.weight

        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Model parameters: {total:,}")
        logger.info(f"Device: {DEVICE}")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, return_attn: bool = False):
        if isinstance(ids, (list, np.ndarray)):
            ids = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        else:
            ids = ids.to(DEVICE)

        is_batch = ids.dim() == 2

        if is_batch:
            B, T = ids.shape
            T = min(T, self.max_length)
            ids = ids[:, :T]
        else:
            T = min(ids.shape[0], self.max_length)
            ids = ids[:T]
            ids = ids.unsqueeze(0)

        pos = torch.arange(T, device=DEVICE)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))

        all_attn = [] if return_attn else None
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn:
                all_attn.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        if not is_batch:
            logits = logits.squeeze(0)

        return logits, all_attn

    def set_training(self, is_training: bool = True):
        self.train() if is_training else self.eval()


class Adam:
    def __init__(self, lr: float = 0.001, parameters=None):
        self._opt = torch.optim.AdamW(
            parameters, lr=lr, weight_decay=0.01,
            betas=(0.9, 0.95)
        )

    def step(self, _=None):
        self._opt.step()

    def zero_grad(self):
        self._opt.zero_grad(set_to_none=True)
