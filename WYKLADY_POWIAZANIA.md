# Powiązania projektu z materiałem wykładowym

Projekt `mini-gpt` jest świetnym poligonem doświadczalnym dla zagadnień z przedmiotu **Metody i narzędzia Big Data**. Poniżej zestawiono, jak konkretne elementy projektu realizują wiedzę z poszczególnych prezentacji.

## Wykład 2: Przygotowanie algorytmu
*Wykład omawia m.in. metody liczenia średniej i wariancji na bieżąco (np. na oknie przesuwnym) oraz filtry.*
- **W projekcie:** W nowym skrypcie `tools/plot_metrics.py` zastosowano wygładzanie krzywej straty (loss) poprzez **średnią kroczącą** i obliczanie lokalnej **wariancji** za pomocą metody `rolling(window=10)` z biblioteki Pandas. Pozwala to lepiej interpretować niestabilności w procesie treningu sieci.

## Wykład 3: Rozkłady macierzy
*Wykład opisuje algebrę liniową, w tym rozkład macierzy na wartości własne (Eigendecomposition) oraz osobliwe (SVD), co stanowi podstawę metody PCA.*
- **W projekcie:** Mechanizm uwagi (*Attention*) bazuje całkowicie na mnożeniu dużych macierzy. Dodatkowo skrypt `tools/visualize_embeddings.py` redukuje wymiarowość ogromnych wektorów słów ze 128 wymiarów do przestrzeni 2D używając algorytmu **PCA (Principal Component Analysis)**, który pod spodem wykorzystuje rozkłady SVD poznane na wykładzie.

## Wykład 4: Selekcja cech i Regularyzacja
*Prezentacja wyjaśnia różne normy (L1, L2) jako metody "karania" modelu za zbytnią złożoność.*
- **W projekcie:** Podczas uczenia modelu używamy optymalizatora o nazwie AdamW. Litera "W" oznacza *weight decay*. Jest to wprost matematyczna implementacja **regularyzacji L2**, zapobiegająca przetrenowaniu modelu poprzez dodawanie "kary za wielkość" do wariancji wag (obniżanie złożoności modelu wg wykładu). Osadzone mechanizmy uwagi stanowią z kolei rodzaj ekstrakcji ważnych cech z tekstu bez ich manualnego projektowania.

## Wykład 5: Źródła danych i narzędzia dla inżyniera
*Wykład kładzie nacisk na analizę i wizualizację danych z użyciem takich narzędzi jak Python, Pandas, Matplotlib czy interaktywne biblioteki (Plotly/Bokeh).*
- **W projekcie:** Trzonem wizualnym są skrypty w katalogu `tools/`. Zamiast prostych, statycznych zrzutów ekranu, generujemy **interaktywne pliki HTML za pomocą Plotly** (`plot_metrics.py`, `visualize_attention.py`, `visualize_embeddings.py`), dokładnie tak, jak rekomendowano to na zajęciach jako nowoczesny standard dla analityków. Obsługa danych CSV odbywa się przy użyciu polecanej biblioteki **Pandas**.

## Wykład 8: Na czym polega uczenie się
*Nowoczesne formy kształcenia (Problem-Based Learning, Active Learning).*
- **W projekcie:** Samodzielna konstrukcja tak złożonej architektury (Transformer) bez użycia wyższopoziomowych, gotowych zabawek to esencja **Problem-Based Learning**. Pozwala to "dotknąć" matematyki ukrytej za szumem marketingowym AI.
