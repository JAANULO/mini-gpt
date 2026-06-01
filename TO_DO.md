# Plan Rozwoju i Audyt Architektury 

Ten dokument zawiera wizję rozwoju projektu, uporządkowaną według poziomu trudności implementacji oraz aktualnego statusu prac.

## Legenda statusów
- `DONE` — funkcja zaimplementowana i dostępna
- `PARTIAL` — funkcja wdrożona częściowo lub z ograniczeniami
- `TODO` — funkcja planowana


1. Attention heatmap ← najlepsza opcja
- `DONE` Wizualizacja tego co "widzi" mechanizm uwagi (Multi-Head Attention). Każdy token patrzy na inne tokeny — można to pokazać jako macierz ciepła. Biblioteka: BertViz lub ręcznie z matplotlib/seaborn.

2. Loss curve
- `PARTIAL` Już masz, ale można ładniej — Plotly zamiast print.

3. Embeddings tokenów (2D)
- `TODO` Tu wracamy do tematu wektorów — tokeny mają embeddingi (wektory). PCA/t-SNE na nich pokazuje np. że podobne słowa są blisko siebie. To sensowna wizualizacja specyficzna dla GPT.

4. Wizualizacja architektury sieci
- `PARTIAL` Gotowe narzędzie: Netron — wczytuje .pt i rysuje graf sieci. Zero kodu.

5. Refaktoryzacja bazy danych
- `TODO` Zastonowienie się z kąd wziąć dane, (wikipiedia, facebook dane, rozmowe z czatów ia)

6. Poprawa nazewnictwa
- `TODO` zmiana nazwy v1 na coś mądrego, zmienienie architektury projektu na bardziej profesjonalnie
git ignore