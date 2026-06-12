# Plan rozbudowy mini-gpt pod kątem Big Data

Plan ma na celu połączenie budowy modelu LLM z narzędziami i teoriami omawianymi na wykładach z przedmiotu "Metody i narzędzia Big Data", zgodnie z ustaleniami z wywiadu. Na ten moment omijamy implementację rozproszonego przetwarzania danych (Spark).

## User Review Required
> [!IMPORTANT]
> Plan został zaktualizowany zgodnie z Twoimi wyborami. Jeśli akceptujesz poniższy plan, daj znać, a rozpocznę tworzenie wizualizacji i dokumentacji.

---

## Proposed Changes

### 1. Zaawansowana wizualizacja danych (Wykład 5: Narzędzia)
Wykorzystanie bibliotek do interaktywnej wizualizacji (Plotly / Bokeh) zamiast statycznych obrazków.
#### [MODIFY] `tools/visualize_embeddings.py`
- Generowanie interaktywnego wykresu PCA dla embeddingów przy użyciu biblioteki Plotly (lub Bokeh), zapisywanego do formatu HTML.
#### [MODIFY] `tools/visualize_attention.py`
- Zastosowanie Plotly do stworzenia interaktywnej heatmapy obrazującej wagi *Attention* pomiędzy słowami.
#### [NEW] `tools/plot_metrics.py`
- Interaktywny wykres krzywej uczenia (Loss) oraz wariancji/średniej kroczącej w Plotly.

### 2. Teoria i opisy matematyczne (Wykłady 2, 3 i 4)
Utworzenie dedykowanego pliku mapującego fragmenty projektu na wykłady, zamiast wplatania wszystkiego w matematykę.
#### [NEW] `WYKLADY_POWIAZANIA.md`
- **Wykład 4:** Wyjaśnienie, dlaczego *weight decay* w optymalizatorze (np. AdamW) to implementacja regularyzacji L2 obniżającej złożoność modelu.
- **Wykład 3:** Wskazanie, że PCA w skryptach wizualizacyjnych opiera się na rozkładach macierzy omawianych na wykładzie.
- **Wykład 2:** Powiązanie metody liczenia średniej i wariancji w logach treningu ze średnią kroczącą/online z wykładu.
- **Wykład 5:** Podkreślenie wykorzystania nowoczesnych narzędzi wizualizacyjnych (Plotly/Bokeh).

---

## Verification Plan
### Automated Tests
- Uruchomienie zaktualizowanych skryptów z folderu `tools/` i sprawdzenie, czy generują poprawne, interaktywne pliki `.html`.

### Manual Verification
- Użytkownik otwiera pliki HTML w przeglądarce i ocenia interaktywność i czytelność wykresów.
- Użytkownik czyta `WYKLADY_POWIAZANIA.md` i weryfikuje jasność powiązań z materiałem z zajęć.
