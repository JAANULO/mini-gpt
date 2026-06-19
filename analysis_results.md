# Analiza powiązań projektu z materiałem z wykładów

Na podstawie dostarczonego projektu (`mini-gpt`) oraz materiałów z przedmiotu **Metody i narzędzia Big Data**, poniżej przedstawiam podsumowanie, które elementy zostały zrealizowane, których brakuje, oraz propozycje na przyszłość.

## 1. Co z wykładów znajduje się już w projekcie?

Projekt wykorzystuje wiele zagadnień, głównie od strony inżynierii danych i matematyki w ML:
- **Redukcja wymiarowości (Wykład 3):** Świetne wykorzystanie algorytmu **PCA** (opartego na SVD) w `visualize_embeddings.py` do mapowania 128-wymiarowych wektorów słów na przestrzeń 2D. 
- **Obliczenia macierzowe (Wykład 3):** Sam rdzeń sieci (Multi-Head Attention) to zaawansowana algebra liniowa (mnożenie wielkich macierzy $Q, K, V$).
- **Statystyka bieżąca (Wykład 2):** Liczenie średniej kroczącej i lokalnej wariancji (`rolling` window) na wykresach oceny postępów (`plot_metrics.py`).
- **Regularyzacja (Wykład 4):** Użyty optymalizator AdamW wprost realizuje karę za złożoność modelu (weight decay, czyli odpowiednik regularyzacji L2).
- **Narzędzia analityczne (Wykład 5):** Wykorzystanie polecanego na zajęciach stosu technologicznego (Pandas, interaktywne wykresy w Plotly).
- **Metodyka (Wykład 8):** Samodzielna implementacja Transformera świetnie wpisuje się w Problem-Based Learning.

## 2. Czego z wykładów brakuje w projekcie?

W projekcie brakuje podejścia stricte "skalowalnego" dla olbrzymich zbiorów danych (Big Data):
- **Brak frameworków do zarządzania potokami (MLOps):** Na wykładzie organizacyjnym wskazano narzędzia **Kedro, MLflow oraz DVC**. Projekt zarządza modelami, parametrami i danymi w prosty, skryptowy sposób.
- **Brak podejścia LangChain/LangGraph:** Wspomniane na wykładach jako standard orkiestracji przepływów z użyciem modeli językowych (LLM). Model z projektu działa samodzielnie ("na surowo").
- **Brak środowisk Big Data:** Nie wykorzystano ekosystemów rozproszonych takich jak **Hadoop** czy **Apache Spark**. Tokenizacja i parsowanie danych są wykonywane sekwencyjnie.
- **Brak Explainable AI (XAI):** Choć dostępna jest wizualizacja uwag (Attention Heatmap), projekt nie wykorzystuje sformalizowanych bibliotek do wyjaśniania, dlaczego model podejmuje daną decyzję w skali makro.
- **Model 3V (Volume, Velocity, Variety):** Projekt trenowany jest na ograniczonym zbiorze w formacie JSON i nie styka się jeszcze z problemami masowego, szybkiego dopływu różnorodnych danych.

---

## 3. Co i jak można zaimplementować, by lepiej powtórzyć materiał?

Aby w 100% wykorzystać ten projekt jako narzędzie do nauki do egzaminu lub rozbudowania portfolio, rekomenduję wdrożenie następujących modyfikacji:

> [!TIP]
> **Zastąp zapisywanie parametrów w CSV i ręczne wersjonowanie modeli bibliotekami MLflow oraz DVC.**

### MLOps: Wdrożenie MLflow oraz DVC
- **Działanie:** Narzędzia do śledzenia eksperymentów ML. 
- **Implementacja:** Wewnątrz `train.py` zintegruj **MLflow**. Zamiast dopisywać wyniki `loss` do pliku `outputs/loss_curve.csv`, wysyłaj je do MLflow (`mlflow.log_metric("loss", loss)`). Będziesz mógł oglądać przebieg treningu w profesjonalnym UI przez przeglądarkę, dokładnie tak jak na zajęciach.
- **DVC (Data Version Control):** Zamiast trzymać `dane.json` po prostu w folderze, "Śledź" je przez DVC. To pokaże ustrukturyzowane zarządzanie artefaktami.

### LangChain: Integracja własnego modelu
- **Działanie:** Framework do łączenia narzędzi AI.
- **Implementacja:** Napisz klasę w Pythonie, która dziedziczy po abstrakcyjnym `LLM` z biblioteki LangChain i podpina pod spód model `mini-gpt`. Pozwoli to używać standardowych promptów i łańcuchów (Chains) LangChain na twoim własnym modelu, co świetnie przypomni architekturę nowoczesnych przepływów.

### Big Data: Użycie Apache Spark do tokenizacji (PySpark)
- **Działanie:** Przetwarzanie współbieżne dla masywnych logów.
- **Implementacja:** Zmień plik `data/prepare_corpus.py`. Zamiast wczytywać JSON standardowym Pythonem w jednym wątku, stwórz mini-skrypt w **PySpark**. Wczytaj JSON jako *Spark DataFrame*, oczyść go i zrzuć z powrotem. Nawet przy małych danych będzie to wspaniały "Proof of Concept" (dowód koncepcji) na to, że system można skalować przez MapReduce i środowisko Hadoop/Spark omówione na Wykładzie 6.

### XAI (Explainable AI): Narzędzie do perturbacji
- **Działanie:** Wyjaśnianie predykcji czarnej skrzynki.
- **Implementacja:** Skoro wykład wspominał o XAI, stwórz skrypt, który sprawdza co się stanie z pewnością predykcji, jeśli z danego zdania wyrzucisz po jednym słowie (np. usuwasz słowo wejściowe, przepuszczasz przez sieć, sprawdzasz jak zmienił się wynik). To świetna baza do zrozumienia, na czym opiera się XAI, nawet bez gotowych frameworków jak SHAP.
