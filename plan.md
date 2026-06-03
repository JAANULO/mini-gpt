# mini-gpt — Plan Rozwoju

> Projekt hobbystyczny / nauka / portfolio  
> Sprzęt: i7-11700F · 32GB RAM · RTX 3060 Ti (8GB VRAM) · SSD 954GB

---

## Faza 1 — Stabilizacja i diagnostyka
> Cel: solidna podstawa przed rozbudową. Status: Częściowo DONE

### 1.1 Gradient Clipping
- Status: DONE
- Co to: Obcina zbyt duże gradienty podczas treningu. Zapobiega "eksplodowaniu" wag modelu.
- Gdzie: main.py, pętla treningowa
- Kod:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Trudność: łatwe · Czas: Wykonane

---

### 1.2 Learning Rate Scheduler
- Status: TODO
- Co to: Automatycznie zmniejsza tempo uczenia w trakcie treningu. Model uczy się szybciej na początku, stabilniej na końcu.
- Gdzie: main.py, po inicjalizacji optymalizatora
- Kod:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optymalizator._opt, T_max=EPOKI)
# w pętli po optymalizator.krok():
scheduler.step()
```
- Trudność: łatwe · Czas: < 1 dzień

---

### 1.3 Checkpoint Saving
- Status: TODO
- Co to: Zapis stanu modelu co N epok. Nie tracisz postępu przy awarii lub przerwaniu treningu.
- Gdzie: main.py, pętla treningowa
- Kod:
```python
if epoka % 100 == 0:
    torch.save({
        'epoch': epoka,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optymalizator._opt.state_dict(),
        'loss': strata,
    }, f"checkpoints/checkpoint_epoch_{epoka}.pt")
```
- Trudność: łatwe · Czas: < 1 dzień

---

### 1.4 Perplexity Metric
- Status: PARTIAL
- Co to: Standardowa miara jakości modelu językowego. Im niższa wartość, tym lepiej model przewiduje następny token.
- Gdzie: main.py, po obliczeniu loss
- Kod:
```python
perplexity = torch.exp(torch.tensor(strata))
print(f"Epoka {epoka} | Loss: {strata:.4f} | Perplexity: {perplexity:.2f}")
```
- Trudność: łatwe · Czas: < 1 dzień

---

## Faza 2 — Wizualizacja i diagnostyka modelu
> Cel: zrozumieć co model "widzi" i jak działa. Dobry materiał do portfolio.

### 2.1 Eksport do ONNX + Netron
- Status: TODO
- Co to: ONNX to otwarty format wymiany modeli. Netron to przeglądarkowe narzędzie które rysuje graf architektury sieci — zero kodu.
- Instalacja: `pip install onnx`
- Kod:
```python
import torch
from mini_gpt.transformer import MiniGPT

model = MiniGPT(...)
model.load_state_dict(torch.load("exports/model_export.pt")["state_dict"])
model.eval()

dummy_input = torch.zeros(1, 16, dtype=torch.long)
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=14)
```
- Użycie: wrzuć `model.onnx` na [netron.app](https://netron.app)
- Trudność: łatwe · Czas: 1 dzień

---

### 2.2 Loss Curve (Plotly/CSV)
- Status: PARTIAL
- Co to: Zapis strat treningowych do CSV lub Plotly HTML. Interaktywne wykresy zamiast print-ów.
- Instalacja: `pip install plotly`
- Kod:
```python
import plotly.graph_objects as go

train_losses = []
for epoka in range(1, EPOKI + 1):
    strata = trenuj(model, optymalizator, zdania_ids)
    train_losses.append(strata)

fig = go.Figure()
fig.add_trace(go.Scatter(y=train_losses, name='Loss'))
fig.write_html("loss_curve.html")
```
- Trudność: łatwe · Czas: < 1 dzień

---

### 2.3 Attention Heatmap
- Status: TODO
- Co to: Wizualizacja mechanizmu uwagi (Multi-Head Attention). Pokazuje na które tokeny "patrzy" model podczas generacji tekstu. Świetny materiał do wyjaśnienia jak działa Transformer.
- Instalacja: `pip install bertviz matplotlib seaborn`
- Kod (matplotlib):
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, tokens, head=0):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attention_weights[head].detach().cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        ax=ax,
        cmap='Blues'
    )
    ax.set_title(f"Attention Head {head}")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png")
```
- Uwaga: musisz zwrócić `attention_weights` z forward() modelu
- Trudność: średnie · Czas: 2–3 dni

---

### 2.4 Embeddings 2D (PCA / t-SNE)
- Status: TODO
- Co to: Wektory tokenów mają np. 128 wymiarów. PCA/t-SNE sprowadza je do 2D. Podobne słowa pojawiają się blisko siebie na wykresie.
- Instalacja: `pip install scikit-learn plotly`
- Kod:
```python
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np

embeddings = model.tok_emb.weight.detach().cpu().numpy()
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

fig = px.scatter(
    x=coords[:, 0], y=coords[:, 1],
    text=list(range(len(embeddings))),
    title="Token Embeddings (PCA 2D)"
)
fig.write_html("embeddings_2d.html")
```
- Trudność: średnie · Czas: 2–3 dni

---

## Faza 3 — Lepsza generacja tekstu
> Cel: jakościowo lepszy output modelu. Widoczna różnica bez zmiany architektury.

### 3.1 Top-k i Top-p Sampling
- Status: TODO
- Co to: Bardziej zaawansowane próbkowanie niż temperatura. Top-k bierze tylko k najbardziej prawdopodobnych tokenów. Top-p (nucleus sampling) bierze tokeny których łączne prawdopodobieństwo przekracza p. Eliminuje "losowe" słabe tokeny.
- Kod:
```python
def top_k_top_p_sampling(logits, top_k=50, top_p=0.9, temperature=1.0):
    logits = logits / temperature

    if top_k > 0:
        top_k_vals = torch.topk(logits, top_k)[0]
        logits[logits < top_k_vals[..., -1, None]] = -float('Inf')

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_idx_to_remove = cumulative_probs > top_p
        sorted_logits[sorted_idx_to_remove] = -float('Inf')
        logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```
- Trudność: średnie · Czas: 2–3 dni

---

### 3.2 Beam Search
- Status: TODO
- Co to: Algorytm generacji który rozważa N najlepszych ścieżek (beam) jednocześnie zamiast zachłannie brać zawsze jeden token. Daje spójniejszy i bardziej sensowny tekst.
- Uwaga: Wolniejszy niż sampling — używaj do "showcase", nie do treningu
- Trudność: średnie · Czas: 3–5 dni

---

### 3.3 Tokenizer BPE
- Status: TODO
- Co to: Byte Pair Encoding — zastępuje tokenizację znak-po-znaku. Dzieli tekst na podsłowa (np. "running" → "run" + "ning"). Znacznie lepszy dla języka naturalnego, mniejsze vocab, lepsza generalizacja.
- Instalacja: `pip install sentencepiece`
- Kod (trening tokenizera):
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='dane.txt',
    model_prefix='tokenizer',
    vocab_size=4096,
    model_type='bpe',
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)
```
- Użycie:
```python
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
tokens = sp.encode("Hello world", out_type=int)
```
- Uwaga: wymaga przeszkolenia modelu od nowa po zmianie tokenizera
- Trudność: średnie · Czas: 3–5 dni

---

## Faza 4 — Przekształcenie w SLM
> Cel: działający Small Language Model na wybranej domenie.
> SLM = mały model językowy trenowany na konkretnej dziedzinie, zdolny do sensownych odpowiedzi.

### 4.1 Dane domenowe
- Status: TODO
- Co to: Jakość danych decyduje o jakości SLM bardziej niż architektura. Potrzebujesz 1–10 MB tekstów z wybranej dziedziny.
- Przykładowe domeny pasujące do Twoich projektów:
  - Prawo/regulacje (naturalnie pasuje do pwr-regulatory-assistant)
  - Dokumentacja techniczna / kod
  - Wikipedia dump (polski lub angielski)
- Źródła:
  - [HuggingFace Datasets](https://huggingface.co/datasets)
  - Wikipedia: `datasets.load_dataset("wikipedia", "20220301.pl")`
  - Własne pliki .txt / .json
- Trudność: średnie · Czas: zależy od domeny

---

### 4.2 Skalowanie architektury
- Status: TODO
- Co to: Zwiększenie liczby warstw, głowic attention i wymiaru modelu. Większy model = lepsza jakość, ale więcej VRAM i czasu treningu.
- Rekomendowana konfiguracja dla RTX 3060 Ti (8GB):

| Parametr | Obecny | Docelowy |
|---|---|---|
| Warstwy (n_layers) | 4 | 6 |
| Głowice attention (n_heads) | 4 | 8 |
| Wymiar modelu (d_model) | 128 | 512 |
| Batch size | 32 | eksperymentuj |

- Uwaga: przy d_model=512 i 6 warstwach masz ~25–50M parametrów — mieści się w 8GB VRAM
- Trudność: trudne · Czas: tydzień+

---

### 4.3 Instruction Tuning
- Status: TODO
- Co to: Doszkolenie na parach `pytanie → odpowiedź`. Zamienia model "dokańczający tekst" w model "konwersacyjny". Wymaga przygotowania datasetu QA.
- Format danych:
```json
[
  {"input": "Co to jest Transformer?", "output": "Transformer to architektura sieci neuronowej..."},
  {"input": "Wyjaśnij mechanizm uwagi.", "output": "Mechanizm uwagi (attention) pozwala..."}
]
```
- Gotowe datasety QA: [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- Trudność: trudne · Czas: 1–2 tygodnie

---

### 4.4 Fine-tuning gotowego SLM z LoRA (opcjonalnie)
- Status: TODO
- Co to: LoRA (Low-Rank Adaptation) trenuje tylko małą część wag istniejącego modelu. Reszta wag jest zamrożona. Drastycznie mniej VRAM i czasu niż pełny fine-tuning.
- Modele pasujące do RTX 3060 Ti (8GB):
  - microsoft/phi-2 (2.7B parametrów)
  - Qwen/Qwen2.5-1.5B
  - HuggingFaceTB/SmolLM2-1.7B
- Instalacja: `pip install transformers peft accelerate`
- Kod (LoRA setup):
```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```
- Trudność: trudne · Czas: tydzień+

---

## Podsumowanie

| Faza | Zadania | Trudność | Szacowany czas |
|---|---|---|---|
| 1 — Stabilizacja | Gradient clipping, scheduler, checkpoints, perplexity | łatwe | kilka dni |
| 2 — Wizualizacja | ONNX/Netron, attention heatmap, embeddings 2D, loss curve | łatwe–średnie | 1–2 tygodnie |
| 3 — Generacja | Top-k/p sampling, beam search, tokenizer BPE | średnie | 1–2 tygodnie |
| 4 — SLM | Dane domenowe, skalowanie, instruction tuning, LoRA | średnie–trudne | kilka tygodni |

---

## Uwagi sprzętowe

- RTX 3060 Ti (8GB VRAM) pozwala na wszystkie fazy bez ograniczeń sprzętowych
- Przy skalowaniu modelu (Faza 4.2) monitoruj VRAM: `nvidia-smi` lub `torch.cuda.memory_summary()`
- Jeśli przekroczysz VRAM — zmniejsz `batch_size` lub użyj `gradient_accumulation_steps`
- Trening dużego modelu od zera może trwać wiele godzin — checkpointy (Faza 1.3) są niezbędne
