# 🧠 Architektura i Matematyka Mini-GPT

Dokument ten opisuje fundamenty matematyczne modelu **Mini-GPT**. Architektura oparta jest na dekoderze Transformera (GPT), zaimplementowanym od zera przy użyciu biblioteki PyTorch.

---

## 1. Przetwarzanie Wejścia

Zanim tekst trafi do sieci neuronowej, musi zostać zamieniony na liczby w przestrzeni wektorowej.

### A. Token & Positional Embedding
Model nie rozumie kolejności znaków automatycznie, dlatego stosujemy dwa rodzaje embeddingów, które są sumowane:
1. **Token Embedding ($E_{tok}$)**: Mapuje każdy unikalny znak na wektor o wymiarze $d_{model}$.
2. **Learned Positional Embedding ($E_{pos}$)**: Wyuczone wektory reprezentujące pozycję (0, 1, 2...) w sekwencji.

$$
x = \text{Dropout}(E_{tok}(\text{tokens}) + E_{pos}(\text{positions}))
$$

---

## 2. Blok Transformera (GPT Block)

Każdy blok składa się z mechanizmu uwagi oraz sieci prostej, połączonych rezdualnie (Residual Connections).

### A. Causal Multi-Head Attention
Mechanizm ten pozwala modelowi skupić się na różnych częściach historii rozmowy. Kluczowym elementem jest **maska kauzalna**, która zeruje prawdopodobieństwa dla tokenów "z przyszłości".

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Gdzie $M$ to macierz maskująca:
$$
M_{ij} = \begin{cases} 0 & \text{dla } i \ge j \\ -\infty & \text{dla } i < j \end{cases}
$$

### B. Feed-Forward Network (FFN)
Po analizie kontekstu przez Attention, każdy wektor jest przetwarzany niezależnie przez dwuwarstwową sieć neuronową z aktywacją **GELU**.

$$
\text{FFN}(x) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(x)))
$$

### C. Layer Normalization (Pre-Norm)
Stosujemy normalizację przed każdą operacją (Attention/FFN), co znacząco poprawia stabilność treningu głębokich sieci.

$$
\text{GPT\_Block}(x) = x + \text{Attention}(\text{LN}_1(x)) + \text{FFN}(\text{LN}_2(x))
$$

---

## 3. Optymalizacja i Generowanie

### A. Weight Tying
W celu redukcji liczby parametrów (o ok. 30%), wagi warstwy **Token Embedding** są identyczne z wagami końcowej warstwy liniowej (**Głowa modelu**). Dzięki temu model uczy się spójnej reprezentacji słów na wejściu i wyjściu.

### B. Cross-Entropy Loss
Model trenowany jest tak, aby maksymalizować prawdopodobieństwo poprawnego następnego znaku.

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t | x_{<t})
$$

### C. Sampling z Temperaturą ($T$)
Podczas rozmowy model generuje rozkład prawdopodobieństwa, który modyfikujemy parametrem temperatury:

$$
P(x_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}
$$

*   **Niska temperatura (0.1)**: Model wybiera najbardziej prawdopodobne znaki (deterministyczny).
*   **Wysoka temperatura (1.0)**: Model staje się bardziej kreatywny i różnorodny.

---

## 4. Parametry Implementacji
*   **Aktywacja**: GELU (Gaussian Error Linear Unit)
*   **Optymalizator**: AdamW (z Weight Decay i Betas 0.9/0.95)
*   **Inicjalizacja**: Normalna ($\mu=0, \sigma=0.02$) zgodnie z oryginalnym GPT-2.
