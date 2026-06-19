"""
Wizualizacja embeddingów tokenów w 2D (PCA).
Użycie: python tools/visualize_embeddings.py
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

PLIK_CACHE      = Path("exports") / "model_cache.pkl"
KATALOG_OUTPUTS = Path("outputs")
PLIK_WYNIKI     = KATALOG_OUTPUTS / "embeddings_2d.html"


def wczytaj_model():
    if not PLIK_CACHE.exists():
        print("❌ Brak exports/model_cache.pkl — uruchom najpierw main.py")
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


def generuj_pca(model, tokenizer):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("❌ Brak scikit-learn — zainstaluj: pip install scikit-learn")
        sys.exit(1)

    embeddings = model.tok_emb.weight.detach().cpu().float().numpy()

    n_komponenty = min(2, embeddings.shape[0], embeddings.shape[1])
    pca    = PCA(n_components=n_komponenty)
    coords = pca.fit_transform(embeddings)

    wariancja = pca.explained_variance_ratio_ * 100

    tokeny = [tokenizer.id_na_slowo.get(i, f"<{i}>")
              for i in range(tokenizer.rozmiar)]

    return coords, tokeny, wariancja


def zapisz_html(coords, tokeny, wariancja, sciezka):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("❌ Brak plotly — zainstaluj: pip install plotly")
        sys.exit(1)

    # Kolory — tokeny specjalne szare, reszta według kategorii
    SPECJALNE   = {"<PAD>", "<UNK>"}
    LITERY      = set("aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż")
    CYFRY       = set("0123456789")

    def kolor(t):
        if t in SPECJALNE:   return "#888888"
        if t in CYFRY:       return "#f59e0b"
        if t in LITERY:      return "#6366f1"
        return "#10b981"  # znaki specjalne (spacja, przecinek itp.)

    kolory = [kolor(t) for t in tokeny]

    fig = go.Figure(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=tokeny,
        textposition="top center",
        textfont=dict(size=11),
        marker=dict(size=8, color=kolory, opacity=0.85),
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=(
            f"Token Embeddings — PCA 2D<br>"
            f"<sup>PC1: {wariancja[0]:.1f}% wariancji  |  "
            f"PC2: {wariancja[1]:.1f}% wariancji</sup>"
        ),
        template="plotly_dark",
        height=700,
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis_title=f"PC1 ({wariancja[0]:.1f}%)",
        yaxis_title=f"PC2 ({wariancja[1]:.1f}%)",
    )

    # Legenda ręczna
    for nazwa, kol in [("litery", "#6366f1"), ("cyfry", "#f59e0b"),
                       ("znaki specjalne", "#10b981"), ("PAD/UNK", "#888888")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=kol),
            name=nazwa, showlegend=True,
        ))

    sciezka = Path(sciezka)
    sciezka.parent.mkdir(exist_ok=True)
    fig.write_html(str(sciezka))
    print(f"✅ Wykres zapisany: {sciezka}")


if __name__ == "__main__":
    print("🔍 Wczytuję model...")
    model, tokenizer = wczytaj_model()
    print(f"  Słownik: {tokenizer.rozmiar} tokenów")

    print("📐 Obliczam PCA...")
    coords, tokeny, wariancja = generuj_pca(model, tokenizer)
    print(f"  PC1: {wariancja[0]:.1f}%  |  PC2: {wariancja[1]:.1f}% wariancji")

    zapisz_html(coords, tokeny, wariancja, PLIK_WYNIKI)