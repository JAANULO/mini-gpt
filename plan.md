# mini-gpt — Roadmapa Projektu


## Legenda statusów
- `✅ DONE` — zaimplementowane i działa
- `🔶 PARTIAL` — częściowo wdrożone
- `⬜ TODO` — do zrobienia


## 🟢 ŁATWE

### #1  Gradient Clipping
- `✅ DONE`
- **Plik:** `main.py` — pętla treningowa, po `loss.backward()`
- Obcina zbyt duże gradienty podczas treningu. Zapobiega "eksplodowaniu" wag modelu.
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### #2  Porządki w repo i nazewnictwo
- `✅ DONE`
- **Pliki:** struktura katalogów (`.gitignore`, `exports/`, `tests/`)
- Dodano `.gitignore`, katalog `exports/`, przeniesiono `model_export.pt` i `model_cache.pkl` do `exports/`, utworzono `tests/` i przeniesiono `test_gpu.py`.
- Usunięto katalog `v1/` (zawierał tylko artefakty i cache).

---

### #3  Aktualizacja README i opisu na GitHubie
- `⬜ TODO`
- **Plik:** `README.md`
- Profesjonalny opis projektu: bez emoji, poprawny język techniczny, sekcje: co to jest, jak uruchomić, architektura, przykłady użycia.

---

### #4  Learning Rate Scheduler
- `✅ DONE`
- **Plik:** `main.py` — po inicjalizacji optymalizatora
- Automatycznie zmniejsza tempo uczenia w trakcie treningu. Model uczy się szybciej na początku, stabilniej na końcu.
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optymalizator._opt, T_max=EPOKI)
# w pętli po optymalizator.krok():
scheduler.step()
```

---

### #5  Checkpoint Saving
- `✅ DONE`
- **Plik:** `main.py` — pętla treningowa
- **Nowy katalog:** `checkpoints/`
- Zapis stanu modelu co N epok. Nie tracisz postępu przy awarii lub przerwaniu treningu.
```python
if epoka % 100 == 0:
    torch.save({
        'epoch': epoka,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optymalizator._opt.state_dict(),
        'loss': strata,
    }, f"checkpoints/checkpoint_epoch_{epoka}.pt")
```

---

### #6  Perplexity Metric
- `🔶 PARTIAL`
- **Plik:** `main.py` — po obliczeniu loss
- Standardowa miara jakości modelu językowego. Im niższa wartość, tym lepiej model przewiduje następny token. Obecne logowanie przez `print`/`tqdm` — brak integracji z wykresami.
```python
perplexity = torch.exp(torch.tensor(strata))
print(f"Epoka {epoka} | Loss: {strata:.4f} | Perplexity: {perplexity:.2f}")
```

---

### #7  Loss Curve (Plotly / CSV)
- `✅ DONE`
- **Plik:** `main.py` — pętla treningowa
- **Nowy plik wyjściowy:** `loss_curve.html` lub `loss.csv`
- Obecne logowanie strat przez `print`/`tqdm`. Do dodania: zapis do CSV i interaktywny wykres Plotly.
- **Instalacja:** `pip install plotly` → dodaj do `requirements.txt`
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

---

### #8  Eksport do ONNX + Netron
- `🔶 PARTIAL`
- **Plik źródłowy modelu:** `exports/model_export.pt`
- **Klasa modelu:** `transformer.py` → `MiniGPT`
- **Nowy skrypt:** `tools/export_onnx.py` (do utworzenia)
- **Nowy plik wyjściowy:** `exports/model.onnx`
- `eksportuj_model()` zapisuje `.pt`, ale brak skryptu konwersji do ONNX w repo.
- **Instalacja:** `pip install onnx` → dodaj do `requirements.txt`
```python
from shared.transformer import MiniGPT

model = MiniGPT(...)
model.load_state_dict(torch.load("exports/model_export.pt")["state_dict"])
model.eval()

dummy_input = torch.zeros(1, 16, dtype=torch.long)
torch.onnx.export(model, dummy_input, "exports/model.onnx", opset_version=14)
```
- **Użycie:** wrzuć `model.onnx` na [netron.app](https://netron.app)

---

## 🟡 ŚREDNIE

### #9  Attention Heatmap
- `✅ DONE`
- **Plik do modyfikacji:** `transformer.py` → klasa `GPTBlok` i `MultiheadAttention`
- **Nowy skrypt:** `tools/visualize_attention.py` (do utworzenia)
- **Nowy plik wyjściowy:** `attention_heatmap.png`
- `MultiheadAttention` jest w kodzie, ale wagi attention nie są zwracane ani eksponowane — trzeba dodać ekstrakcję z `forward()`.
- **Instalacja:** `pip install matplotlib seaborn` → dodaj do `requirements.txt`
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
- **Uwaga:** najpierw dodaj `return attn_weights` z `forward()` w `GPTBlok`

---

### #10  Embeddings tokenów 2D (PCA / t-SNE)
- `✅ DONE`
- **Plik źródłowy:** `transformer.py` → warstwa `tok_emb` (embedding tokenów)
- **Nowy skrypt:** `tools/visualize_embeddings.py` (do utworzenia)
- **Nowy plik wyjściowy:** `embeddings_2d.html`
- Wektory tokenów mają 128 wymiarów. PCA/t-SNE sprowadza je do 2D — podobne słowa pojawiają się blisko siebie.
- **Instalacja:** `pip install scikit-learn plotly` → dodaj do `requirements.txt`
```python
import plotly.express as px
from sklearn.decomposition import PCA

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

---

### #11  Top-k i Top-p Sampling
- `✅ DONE`
- **Plik do modyfikacji:** `main.py` — funkcja generacji tekstu (tryb czatu)
- Bardziej zaawansowane próbkowanie niż temperatura. Top-k bierze tylko k najbardziej prawdopodobnych tokenów. Top-p bierze tokeny których łączne prawdopodobieństwo przekracza p.
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

---

### #12  Tokenizer BPE
- `⬜ TODO` 
- **Plik do zastąpienia:** `tokenizer.py` — obecna tokenizacja znak-po-znaku
- **Nowy plik wyjściowy:** `tokenizer.model` + `tokenizer.vocab` (SentencePiece)
- Byte Pair Encoding dzieli tekst na podsłowa. Mniejsze vocab, lepsza generalizacja niż char-level.
- **Instalacja:** `pip install sentencepiece` → dodaj do `requirements.txt`
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='dane.txt',       # eksportuj dane.json do .txt najpierw
    model_prefix='tokenizer',
    vocab_size=4096,
    model_type='bpe',
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)
```
- **⚠️ Uwaga:** wymaga przeszkolenia modelu od nowa — zmiana tokenizera = nowy vocab

---

### #13  Beam Search
- `⬜ TODO` 
- **Plik do modyfikacji:** `main.py` — funkcja generacji tekstu (tryb czatu)
- Algorytm generacji który rozważa N najlepszych ścieżek jednocześnie zamiast zachłannie brać jeden token. Daje spójniejszy tekst.
- **⚠️ Uwaga:** wolniejszy niż sampling — używaj do showcase, nie w treningu.

---

## 🔴 TRUDNE

### #14  Dane domenowe
- `⬜ TODO` 
- **Plik do zastąpienia/rozbudowania:** `dane.json`
- **Nowy katalog:** `data/` z pipeline'em preprocessing
- Jakość danych decyduje o jakości SLM bardziej niż architektura. Potrzebujesz 1–10 MB tekstów z wybranej dziedziny.
- **Pasujące domeny:** prawo/regulacje (→ `pwr-regulatory-assistant`), dokumentacja techniczna, Wikipedia PL
- **Źródła:**
  - [HuggingFace Datasets](https://huggingface.co/datasets)
  - `datasets.load_dataset("wikipedia", "20220301.pl")`
  - Własne pliki `.txt` / `.json`

---

### #15  Refaktoryzacja pipeline danych
- `⬜ TODO` 
- **Pliki do modyfikacji:** `dane.json`, `main.py` (ładowanie danych)
- **Nowy katalog:** `data/` — surowe dane, przetworzone dane, skrypty preprocessing
- Zdefiniować format danych, pipeline czyszczenia i preprocessing. Powiązane z #12 (BPE) i #14 (dane domenowe).

---

### #16  Skalowanie architektury
- `⬜ TODO` 
- **Plik do modyfikacji:** `transformer.py` — parametry `MiniGPT`, `GPTBlok`
- **Plik do modyfikacji:** `main.py` — konfiguracja hiperparametrów
- Zwiększenie liczby warstw, głowic attention i wymiaru modelu.

| Parametr | Obecny | Docelowy |
|---|---|---|
| Warstwy (n_layers) | 4 | 6 |
| Głowice attention (n_heads) | 4 | 8 |
| Wymiar modelu (d_model) | 128 | 512 |
| Batch size | 32 | eksperymentuj |

- Przy d_model=512 i 6 warstwach → ~25–50M parametrów, mieści się w 8GB VRAM.
- Monitoruj VRAM: `nvidia-smi` lub `torch.cuda.memory_summary()`

---

### #17  Instruction Tuning
- `⬜ TODO` 
- **Nowy plik danych:** `data/instruct_dataset.json`
- **Plik do modyfikacji:** `main.py` — pętla treningowa (nowy tryb fine-tuning)
- Doszkolenie na parach `pytanie → odpowiedź`. Zamienia model dokańczający tekst w model konwersacyjny. Wymaga danych QA.
```json
[
  {"input": "Co to jest Transformer?", "output": "Transformer to architektura..."},
  {"input": "Wyjaśnij mechanizm uwagi.", "output": "Mechanizm uwagi (attention) pozwala..."}
]
```
- **Gotowe datasety:** [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5)

---

### #18  Fine-tuning gotowego SLM z LoRA (opcjonalnie)
- `⬜ TODO` 
- **Nowy skrypt:** `tools/lora_finetune.py` (do utworzenia)
- **Niezależny od obecnej architektury** — używasz gotowego modelu z HuggingFace zamiast `transformer.py`
- LoRA (Low-Rank Adaptation) trenuje tylko ~1% wag istniejącego modelu. Mieści się w 8GB VRAM.
- **Modele:** `microsoft/phi-2` (2.7B), `Qwen/Qwen2.5-1.5B`, `HuggingFaceTB/SmolLM2-1.7B`
- **Instalacja:** `pip install transformers peft accelerate` → dodaj do `requirements.txt`
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

