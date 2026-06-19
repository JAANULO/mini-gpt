"""
Wizualizacja wag attention dla mini-GPT.
Użycie: python tools/visualize_attention.py "co to jest warszawa"
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_gpt.transformer import MiniGPT
from mini_gpt.utils import DEVICE
from mini_gpt.tokenizer import Tokenizer

PLIK_CACHE       = Path("exports") / "model_cache.pkl"
KATALOG_OUTPUTS  = Path("outputs")
PLIK_WYNIKI      = KATALOG_OUTPUTS / "attention_heatmap.html"

def wczytaj_model():
    if not PLIK_CACHE.exists():
        print(f"❌ Brak {PLIK_CACHE} — uruchom najpierw main.py")
        sys.exit(1)
    dane = torch.load(PLIK_CACHE, map_location=DEVICE, weights_only=False)
    cfg  = dane["config"]
    model = MiniGPT(
        rozmiar_slownika = cfg["rozmiar_slownika"],
        wymiar           = cfg["wymiar"],
        maks_dlugosc     = cfg["maks_dlugosc"],
    ).to(DEVICE)
    model.load_state_dict(dane["state_dict"])
    model.eval()
    return model, dane["tokenizer"]

def generuj_heatmapy(tekst, model, tokenizer):
    ids    = tokenizer.koduj(tekst)
    tokeny = [tokenizer.id_na_slowo.get(i, "?") for i in ids]

    wejscie = torch.tensor(ids, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, wszystkie_attn = model.forward(wejscie, return_attn=True)

    return tokeny, wszystkie_attn

def zapisz_html(tokeny, wszystkie_attn, sciezka):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("❌ Brak plotly — zainstaluj: pip install plotly")
        sys.exit(1)

    n_warstw  = len(wszystkie_attn)
    n_glowic  = wszystkie_attn[0].shape[1]
    etykiety  = [f"'{t}'" for t in tokeny]

    fig = make_subplots(
        rows=n_warstw, cols=n_glowic,
        subplot_titles=[
            f"W{w+1} G{g+1}" for w in range(n_warstw) for g in range(n_glowic)
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )

    for w_idx, attn_warstwy in enumerate(wszystkie_attn):
        # attn_warstwy: (B, n_glowic, T, T) lub (n_glowic, T, T)
        macierz = attn_warstwy[0] if attn_warstwy.dim() == 4 else attn_warstwy
        macierz = macierz.cpu().float().numpy()

        for g_idx in range(n_glowic):
            fig.add_trace(
                go.Heatmap(
                    z=macierz[g_idx],
                    x=etykiety,
                    y=etykiety,
                    colorscale="Blues",
                    showscale=False,
                    zmin=0, zmax=1,
                ),
                row=w_idx + 1,
                col=g_idx + 1,
            )

    fig.update_layout(
        title=f"Attention heatmap — '{' '.join(tokeny)}'",
        template="plotly_dark",
        height=300 * n_warstw,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(exist_ok=True)
    fig.write_html(str(sciezka))
    print(f"✅ Heatmapa zapisana: {sciezka}")

if __name__ == "__main__":
    tekst = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "co to jest warszawa"
    print(f"🔍 Analizuję: '{tekst}'")
    model, tokenizer = wczytaj_model()
    tokeny, wszystkie_attn = generuj_heatmapy(tekst, model, tokenizer)
    print(f"  Tokeny ({len(tokeny)}): {tokeny}")
    zapisz_html(tokeny, wszystkie_attn, PLIK_WYNIKI)