"""
Wizualizacja metryk treningowych modelu (Strata, Perplexity)
oraz analiza wariancji/średniej kroczącej (Wykład 2).
Użycie: python tools/plot_metrics.py
"""

import sys
import os
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ścieżki
KATALOG_OUTPUTS = Path("outputs")
PLIK_LOSS_CSV   = KATALOG_OUTPUTS / "loss_curve.csv"
PLIK_WYNIKI     = KATALOG_OUTPUTS / "metrics_analysis.html"

def wygeneruj_wykresy():
    if not PLIK_LOSS_CSV.exists():
        print(f"❌ Brak pliku {PLIK_LOSS_CSV}. Uruchom najpierw trening w main.py")
        sys.exit(1)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("❌ Brak biblioteki plotly — zainstaluj: pip install plotly")
        sys.exit(1)

    # Wczytanie danych (Pandas - Wykład 5)
    df = pd.read_csv(PLIK_LOSS_CSV)
    
    # Wykład 2: Średnia i wariancja na oknie przesuwnym
    # Obliczamy wariancję kroczącą dla straty z oknem = 10 epok
    okno = min(10, len(df))
    df['strata_srednia'] = df['strata'].rolling(window=okno, min_periods=1).mean()
    df['strata_wariancja'] = df['strata'].rolling(window=okno, min_periods=1).var().fillna(0)

    # Tworzenie interaktywnego wykresu (Plotly - Wykład 5)
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Strata (Cross-Entropy Loss) z wygładzaniem (Średnia krocząca)",
            "Wariancja straty na oknie przesuwnym",
            "Perplexity"
        ),
        vertical_spacing=0.1,
    )

    # 1. Strata i Średnia krocząca
    fig.add_trace(
        go.Scatter(x=df['epoka'], y=df['strata'], mode="lines", name="Strata",
                   line=dict(color="rgba(99, 102, 241, 0.4)", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['epoka'], y=df['strata_srednia'], mode="lines", name="Średnia krocząca",
                   line=dict(color="#10b981", width=2)),
        row=1, col=1,
    )

    # 2. Wariancja
    fig.add_trace(
        go.Scatter(x=df['epoka'], y=df['strata_wariancja'], mode="lines", name="Wariancja straty",
                   line=dict(color="#ef4444", width=1.5)),
        row=2, col=1,
    )

    # 3. Perplexity
    fig.add_trace(
        go.Scatter(x=df['epoka'], y=df['perplexity'], mode="lines", name="Perplexity",
                   line=dict(color="#f59e0b", width=1.5)),
        row=3, col=1,
    )

    fig.update_layout(
        title="Zaawansowana analiza metryk modelu mini-GPT",
        template="plotly_dark",
        height=900,
        showlegend=True,
        margin=dict(l=50, r=30, t=60, b=40),
    )
    
    KATALOG_OUTPUTS.mkdir(exist_ok=True)
    fig.write_html(str(PLIK_WYNIKI))
    print(f"✅ Wykres zapisany: {PLIK_WYNIKI}")

if __name__ == "__main__":
    wygeneruj_wykresy()
