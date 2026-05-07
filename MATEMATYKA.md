# 🧠 Architektura i Matematyka Projektu

Dokument ten opisuje, jak system działa "pod maską". Przedstawia algorytmy i wzory matematyczne użyte w projekcie, które zostały zaimplementowane od zera, bez użycia gotowych bibliotek z gotowymi modelami (takich jak `scikit-learn` czy `transformers`).

Projekt skupia się na **Modelu Generatywnym (Mini-GPT)**.

---

## 1. Model Generatywny (Mini-GPT)

Mini-GPT to głęboka sieć neuronowa zbudowana w oparciu o architekturę Transformera, uczona przewidywania następnego tokenu (znaku) na podstawie historii.

### A. Mechanizm Samouwagi (Multi-Head Attention)
Pozwala sieci ocenić, które znaki w zdaniu są najważniejsze dla zrozumienia obecnego kontekstu.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

*   **$Q$ (Query), $K$ (Key), $V$ (Value)**: Macierze zapytań, kluczy i wartości.
*   **$\sqrt{d_k}$**: Czynnik skalujący. Zapobiega eksplozji gradientów (zbyt dużym liczbom psującym sieć).
*   **Maska Causalna**: Górny trójkąt macierzy $QK^T$ jest zerowany (wartość $-\infty$), co uniemożliwia modelowi "patrzenie w przyszłość" podczas treningu.

### B. Normalizacja i Aktywacja (LayerNorm & GELU)
Wewnątrz bloków Transformera użyto najnowocześniejszych technik stabilizujących sieć:

**1. Layer Normalization (Normalizacja Warstwy):**
Wyrównuje rozkład aktywacji wewnątrz sieci:

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

**2. GELU (Gaussian Error Linear Unit):**
Zamiast prostej aktywacji ReLU (odcinającej wartości ujemne), użyto nieliniowej funkcji GELU (stosowanej m.in. w modelach BERT i GPT-3):

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$
*(gdzie $\Phi(x)$ to dystrybuanta standardowego rozkładu normalnego).*

### C. Funkcja Straty (Cross-Entropy Loss)
Model trenowany jest z wykorzystaniem funkcji straty krzyżowej entropii (Cross-Entropy Loss). Karze ona model tym mocniej, im bardziej był "pewny" błędnej odpowiedzi.

$$
\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)
$$

Gdzie $y_i$ to wektor oczekiwany (Prawda), a $\hat{y}_i$ to prawdopodobieństwo wygenerowane przez model.

### D. Generowanie tekstu (Softmax i Temperatura)
Ostatnia warstwa sieci (w której zastosowano *Weight Tying* w celu redukcji parametrów) zwraca surowe liczby (Logits). Przekształcamy je na ostateczne prawdopodobieństwa używając funkcji Softmax z parametrem Temperatury ($T$).

$$
P(x_i) = \frac{e^{x_i / T}}{\sum_{j} e^{x_j / T}}
$$

*   **$T < 1.0$ (np. 0.1)**: Model staje się sztywny i deterministyczny.
*   **$T = 1.0$**: Naturalne prawdopodobieństwa modelu.
