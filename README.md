# 🤖 mini-gpt

> Eksperymentalna implementacja architektury Transformera (GPT) napisana od zera w PyTorch.

Projekt ten jest środowiskiem badawczym ("research playground") przeznaczonym do nauki i eksperymentów z generatywnymi modelami językowymi. Został odseparowany od systemów produkcyjnych, aby zachować czystość i czytelność kodu.

---

## 🚀 Kluczowe cechy

- **Autorska architektura**: Pełna implementacja bloków Transformera, mechanizmu Multi-Head Attention oraz LayerNorm bez korzystania z gotowych modeli.
- **Lekkość**: Model zoptymalizowany pod kątem nauki na domowym procesorze lub karcie graficznej.
- **Tryb rozmowy**: Interaktywny czat z modelem, który potrafi zapamiętywać kontekst (okno pamięci).
- **Diagnostyka**: Wbudowane narzędzia do wizualizacji straty (loss) podczas treningu.

---

## 🛠️ Instalacja i uruchomienie

### Wymagania
- Python 3.12+
- PyTorch
- NumPy
- tqdm (opcjonalnie)

### Szybki start
```bash
# 1. Zainstaluj zależności
pip install -r requirements.txt

# 2. Uruchom trening i czat
python main.py
```

### Komendy w czacie
- `/temp 0.1` — Zmiana temperatury generowania (determinizm vs kreatywność).
- `/historia` — Podgląd aktualnego kontekstu pamięci.
- `/zapomnij` — Wyczyszczenie historii rozmowy.
- `koniec` — Zamknięcie programu.

---

## 🧠 Dokumentacja matematyczna
Szczegółowy opis wzorów matematycznych i architektury znajdziesz w pliku [MATEMATYKA.md](./MATEMATYKA.md).

---

## 📂 Struktura projektu
- `main.py` — Główny punkt wejścia (trening + czat).
- `shared/` — Rdzeń architektury (Transformer i Tokenizer).
- `dane.json` — Zbiór danych treningowych.
- `model_export.pt` — Wyeksportowane wagi modelu.
