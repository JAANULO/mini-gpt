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
python main.py
```

Wymagania: Python 3.10+, PyTorch, NumPy, tqdm, plotly, scikit-learn, onnx.

---

## Struktura projektu

```
mini_gpt/
  transformer.py     # architektura modelu (GPTBlok, MiniGPT)
  tokenizer.py       # character-level tokenizer
data/
  dane.json          # zbiór danych treningowych
tools/
  visualize_attention.py   # heatmapa wag attention
  visualize_embeddings.py  # wizualizacja embeddingów tokenów (PCA 2D)
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
tests/
  test_gpu.py        # test dostępności GPU i benchmark CUDA
main.py              # trening + interaktywny czat
MATEMATYKA.md        # dokumentacja matematyczna architektury
```

---

## Narzędzia diagnostyczne

Po treningu (`python main.py`) dostępne są trzy narzędzia wizualizacyjne:

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
