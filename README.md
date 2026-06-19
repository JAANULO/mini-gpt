# mini-gpt — Transformer od zera

Eksperymentalna implementacja architektury GPT napisana od zera w PyTorch, bez użycia gotowych frameworków NLP. Projekt służy do nauki i zrozumienia jak działają duże modele językowe — od matematyki po kod.

---

## Architektura

Pełna implementacja dekodera Transformera (~831 000 parametrów):

- Multi-Head Causal Attention z maską kauzalną
- Feed-Forward Network z aktywacją GELU
- Pre-Norm LayerNorm (normalizacja przed każdą operacją)
- Learned Positional Embeddings
- Weight Tying (współdzielone wagi embedding i głowy modelu)
- Optymalizator AdamW z Cosine LR Scheduling

Szczegółowy opis matematyczny w [MATEMATYKA.md](./MATEMATYKA.md).

---

## Instalacja

```bash
pip install -r requirements.txt
python train.py
```

Wymagania: Python 3.10+, PyTorch, NumPy, tqdm, plotly, scikit-learn, onnx.

---

## Struktura projektu

```
mini_gpt/
  transformer.py     # architektura modelu (GPTBlok, MiniGPT)
  tokenizer.py       # SentencePiece BPE tokenizer loader
  utils.py           # Narzędzia pomocnicze i ładowanie wag
data/
  dane.json          # zbiór danych w formacie JSON
  prepare_corpus.py  # wyciąganie tekstu z JSON do formatu text
  raw_corpus.txt     # wygenerowany plik pod trening tokenizera
tools/
  train_tokenizer.py       # trenowanie tokenizera BPE
  visualize_attention.py   # heatmapa wag attention
  visualize_embeddings.py  # wizualizacja embeddingów tokenów (PCA 2D)
  plot_metrics.py          # wykresy perplexity i straty
  export_onnx.py           # eksport modelu do formatu ONNX
exports/
  model_export.pt    # skompresowany model (float16)
  model_cache.pkl    # cache treningu
outputs/
  loss_curve.csv     # historia straty i perplexity
  loss_curve.html    # interaktywny wykres treningu (Plotly)
  attention_heatmap.html   # wizualizacja attention
  embeddings_2d.html       # embeddingi tokenów w przestrzeni 2D
  model.onnx               # graf architektury (otwórz na netron.app)
checkpoints/
  checkpoint_epoch_N.pt    # checkpointy co 100 epok
app.py               # aplikacja Flask i API UI
chat.py              # CLI z historią pamięci konwersacji
train.py             # główny pipeline treningowy (z automatycznym treningiem BPE)
MATEMATYKA.md        # dokumentacja matematyczna architektury
WYKLADY_POWIAZANIA.md # korelacja projektu z zajęciami Big Data
```

---

## Narzędzia diagnostyczne

Po treningu (`python train.py`) dostępne są trzy narzędzia wizualizacyjne:

```bash
# Krzywa uczenia — loss i perplexity przez epoki
# (generuje się automatycznie po treningu)
# outputs/loss_curve.html

# Heatmapa wag attention dla podanego tekstu
python tools/visualize_attention.py "co to jest warszawa"

# Embeddingi tokenów zredukowane do 2D przez PCA
python tools/visualize_embeddings.py

# Graf architektury modelu (wymaga: pip install onnx)
python tools/export_onnx.py
# Następnie wgraj outputs/model.onnx na https://netron.app
```

---

## Komendy czatu

| Komenda | Opis |
|---|---|
| `/temp 0.1` | temperatura generowania (0.01 = deterministyczne, 1.0 = losowe) |
| `/historia` | podgląd aktualnego kontekstu pamięci |
| `/zapomnij` | wyczyszczenie historii rozmowy |
| `/pomoc` | lista dostępnych komend |
| `koniec` | zamknięcie programu |

---

## Plan rozwoju

Szczegółowa roadmapa w [plan.md](./plan.md).
