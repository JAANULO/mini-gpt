# Plan Rozwoju i Audyt Architektury

Aktualny stan (zaktualizowano: 2026-06-01)

- Zawartość repozytorium została przejrzana: `main.py`, `shared/transformer.py`, `shared/tokenizer.py`, `model_export.pt`, `dane.json`.
- Zaimplementowane: gradient clipping podczas treningu, eksport skompresowany (`model_export.pt`), interaktywny tryb czatu z pamięcią, zapis końcowy cache (`model_cache.pkl`).
- Częściowo: obecne są printy/monitorowanie przez tqdm — brak interaktywnego wykresu loss (Plotly/TensorBoard). Brak periodycznych checkpointów co N epok.
- Niezaimplementowane: wizualizacja attention (heatmap), embeddings 2D, tokenizer BPE, beam/top-k/top-p, LR scheduler.

## Legenda statusów
- `DONE` — funkcja zaimplementowana i dostępna
- `PARTIAL` — funkcja wdrożona częściowo lub z ograniczeniami
- `TODO` — funkcja planowana

0. Aktualizacja opisu
- `TODO` Aktualizacja readme,(profejnolany język, bez emoi itp), przygotowanie opisu na githubie, taki itp 


1. Loss curve
- `PARTIAL` Obecnie logowanie strat odbywa się przez print / `tqdm`. Rekomendacja: zapisywać straty do pliku CSV lub dodać Plotly/TensorBoard dla interaktywnych wykresów. (łatwe)

2. Wizualizacja architektury sieci
- `PARTIAL` Netron + eksport do ONNX istnieją jako pomysł w planie; `eksportuj_model()` zapisuje skompresowany `.pt`. Netron działa z `.pt`/ONNX, ale nie ma automatycznego skryptu konwersji w repo. (łatwe)

3. Poprawa nazewnictwa i porządki w repo
- `DONE` Wykonane kroki: dodano `.gitignore`, utworzono katalog `exports/`, przeniesiono `model_export.pt` i `model_cache.pkl` do `exports/`, utworzono `tests/` i przeniesiono `test_gpu.py`, zaktualizowano `README.md`.

	- Uwaga: katalog `v1/` został usunięty (zawierał jedynie artefakty i pliki cache). (łatwe)

4. Attention heatmap
- `TODO` Wizualizacja wag mechanizmu uwagi (Multi-Head Attention). W kodzie `MultiheadAttention` jest użyte, ale wagi attention nie są zapisywane ani eksponowane; trzeba dodać ekstrakcję wag i skrypt rysujący heatmapę (BertViz / seaborn). (średnie)

5. Embeddings tokenów (2D)
- `TODO` Redukcja wymiarów wektorów tokenów (PCA/t-SNE) i wizualizacja (sklearn + Plotly/seaborn). (średnie)

6. Refaktoryzacja bazy danych / źródeł danych
- `TODO` Zdefiniować źródła danych, format i pipeline czyszczenia/preprocessing. (trudne)

## Krótkie rekomendacje (następne kroki)

- Dodać LR scheduler (torch.optim.lr_scheduler) — szybka poprawka w `main.py`.
- Wprowadzić periodyczne checkpointy (np. co 100 epok) w pętli treningowej.
- Dodać prosty eksport strat do `loss.csv` aby móc szybko rysować wykresy.
- Dodać opcjonalne zwracanie wag attention z `GPTBlok` dla przyszłej wizualizacji.

Jeśli chcesz, mogę automatycznie zaaplikować te drobne zmiany (LR scheduler + checkpointy + zapis loss), lub przygotować patch dodający zwracanie wag attention.