import os
import json
import random
from pathlib import Path

import torch
from mini_gpt.transformer import MiniGPT, Adam
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.utils import (
    DEVICE, hash_data, save_cache, load_cache, load_export, generate_text, logger
)

# ============================================================
# SETTINGS
# ============================================================

from mini_gpt.config import cfg

DATA_FILE = cfg.data_file

EXPORT_DIR = Path("exports")
EXPORT_FILE = EXPORT_DIR / "model_export.pt"
OUTPUTS_DIR = Path("outputs")
LOSS_CSV_FILE = OUTPUTS_DIR / "loss_curve.csv"
LOSS_HTML_FILE = OUTPUTS_DIR / "loss_curve.html"
CACHE_FILE = EXPORT_DIR / "model_cache.pkl"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ============================================================
# DIAGNOSTICS
# ============================================================

def save_loss_csv(loss_history):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(LOSS_CSV_FILE, "w", encoding="utf-8") as f:
        f.write("epoch,loss,perplexity\n")
        for epoch, loss in loss_history:
            perp = float(torch.exp(torch.tensor(loss)).item())
            f.write(f"{epoch},{loss:.6f},{perp:.4f}\n")

def generate_chart(loss_history):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("Plotly is not installed. Skipping chart generation.")
        return

    epochs = [e for e, _ in loss_history]
    losses = [s for _, s in loss_history]
    perplexities = [float(torch.exp(torch.tensor(s)).item()) for _, s in loss_history]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cross-Entropy Loss", "Perplexity"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=losses, mode="lines", name="Loss", line=dict(color="#6366f1", width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=perplexities, mode="lines", name="Perplexity", line=dict(color="#f59e0b", width=1.5)),
        row=2, col=1,
    )
    fig.update_layout(
        title="Mini-GPT — Learning Curve",
        template="plotly_dark",
        height=600,
        showlegend=False,
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.write_html(LOSS_HTML_FILE)
    logger.info(f"Chart saved to: {LOSS_HTML_FILE}")

# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath):
    if not os.path.exists(filepath):
        logger.error(f"File not found: '{filepath}'!")
        exit(1)
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("zdania", [])

# ============================================================
# TRAINING
# ============================================================

def build_batch(sentences_ids, batch_size, max_length):
    sample = random.sample(sentences_ids, min(batch_size, len(sentences_ids)))
    sample = [ids[:max_length] for ids in sample if len(ids) >= 2]
    max_len_in_batch = max(len(ids) for ids in sample)

    input_list = []
    target_list = []
    for ids in sample:
        inp = ids[:-1]
        tgt = ids[1:]
        pad_len = max_len_in_batch - 1 - len(inp)
        inp = inp + [0] * pad_len
        tgt = tgt + [0] * pad_len
        input_list.append(inp)
        target_list.append(tgt)

    inputs = torch.tensor(input_list, dtype=torch.long, device=DEVICE)
    targets = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
    return inputs, targets

def train_epoch(model, optimizer, sentences_ids):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    n_batches = max(1, min(20, len(sentences_ids) // BATCH_SIZE))
    total_loss = 0.0

    for _ in range(n_batches):
        inputs, targets = build_batch(sentences_ids, BATCH_SIZE, MAX_LENGTH)
        optimizer.zero_grad()
        logits, _ = model.forward(inputs)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return total_loss / n_batches

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    import io

    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 55)
    print("  MINI-GPT – Transformer from scratch!")
    print("=" * 55 + "\n")

    print("📚 Loading data...")
    sentences = load_data(DATA_FILE)
    current_hash = hash_data(DATA_FILE)
    print(f"  Loaded {len(sentences)} sentences.\n")

    print("🧠 Creating/Loading BPE Tokenizer...")
    
    # 1. Automatyczne przygotowanie korpusu
    from data.prepare_corpus import main as prepare_corpus_main
    from tools.train_tokenizer import main as train_tokenizer_main
    
    if not os.path.exists("data/raw_corpus.txt"):
        print("   Przygotowywanie korpusu...")
        prepare_corpus_main()
        
    # 2. Wytrenowanie tokenizera jeśli nie istnieje
    if not os.path.exists("exports/tokenizer.model"):
        print("   Trenowanie nowego modelu SentencePiece...")
        train_tokenizer_main()
        
    tokenizer_temp = Tokenizer()
    loaded = tokenizer_temp.load("exports/tokenizer.model")
    if not loaded:
        print("❌ Błąd ładowania BPE Tokenizera.")
        sys.exit(1)

    model = MiniGPT(
        vocab_size=tokenizer_temp.vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        max_length=cfg.max_length,
    ).to(DEVICE)
    print()

    tokenizer_from_cache, cache_ok = load_cache(model, current_hash, CACHE_FILE)

    if cache_ok:
        tokenizer = tokenizer_from_cache
        model.set_training(False)
        print("✅ Loaded trained model from cache!")
        print("   (Data unchanged – skipping training)\n")
    else:
        tokenizer_export, export_ok = load_export(model, EXPORT_FILE)
        if export_ok:
            tokenizer = tokenizer_export
            model.set_training(False)
            print(f"✅ Loaded {EXPORT_FILE} – skipping training\n")
        else:
            tokenizer = tokenizer_temp
            sentences_ids = [tokenizer.encode(s) for s in sentences]
            optimizer = Adam(model.parameters(), lr=cfg.lr)

            logger.info(f"Target Epochs: {cfg.epochs}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer._opt, T_max=cfg.epochs
            )

            try:
                import mlflow
                mlflow.set_experiment("mini-gpt-training")
                mlflow.start_run()
                mlflow.log_params({
                    "epochs": cfg.epochs,
                    "learning_rate": cfg.lr,
                    "embed_dim": cfg.embed_dim,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "max_length": cfg.max_length,
                })
            except ImportError:
                pass

            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                has_tqdm = False

            print(f"⚙️  Training for {cfg.epochs} epochs...\n")

            loss_history = []
            model.set_training(True)

            if has_tqdm:
                pbar = tqdm(
                    range(1, cfg.epochs + 1),
                    desc="  Training",
                    unit="epoch",
                    bar_format="{l_bar}{bar:40}{r_bar}",
                    dynamic_ncols=True,
                )
                for epoch in pbar:
                    loss = train_epoch(model, optimizer, sentences_ids)
                    scheduler.step()
                    if epoch % 100 == 0:
                        os.makedirs("checkpoints", exist_ok=True)
                        ckpt_path = f"checkpoints/checkpoint_epoch_{epoch}.pt"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer._opt.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': loss,
                        }, ckpt_path)
                        try:
                            import mlflow
                            if mlflow.active_run():
                                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                        except ImportError:
                            pass
                    perplexity = float(torch.exp(torch.tensor(loss)).item())
                    try:
                        import mlflow
                        if mlflow.active_run():
                            mlflow.log_metrics({"loss": loss, "perplexity": perplexity}, step=epoch)
                    except ImportError:
                        pass
                    pbar.set_postfix(loss=f"{loss:.4f}", perplexity=f"{perplexity:.2f}")
                    loss_history.append((epoch, loss))
            else:
                for epoch in range(1, cfg.epochs + 1):
                    loss = train_epoch(model, optimizer, sentences_ids)
                    scheduler.step()
                    if epoch % 100 == 0:
                        os.makedirs("checkpoints", exist_ok=True)
                        ckpt_path = f"checkpoints/checkpoint_epoch_{epoch}.pt"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer._opt.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': loss,
                        }, ckpt_path)
                        print(f"    💾 Checkpoint: epoch_{epoch}.pt")
                        try:
                            import mlflow
                            if mlflow.active_run():
                                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                        except ImportError:
                            pass
                    loss_history.append((epoch, loss))
                    perplexity = float(torch.exp(torch.tensor(loss)).item())
                    try:
                        import mlflow
                        if mlflow.active_run():
                            mlflow.log_metrics({"loss": loss, "perplexity": perplexity}, step=epoch)
                    except ImportError:
                        pass
                    if epoch % 100 == 0 or epoch == cfg.epochs:
                        perplexity = float(torch.exp(torch.tensor(loss)).item())
                        proc = epoch / cfg.epochs * 100
                        print(f"  Epoch {epoch}/{cfg.epochs} ({proc:.0f}%)  loss: {loss:.4f}  perplexity: {perplexity:.2f}")

            print("\n  ✅ Training completed!")
            model.set_training(False)
            save_cache(model, tokenizer, current_hash, CACHE_FILE, EXPORT_FILE)
            print(f"  💾 Model saved to '{CACHE_FILE}'")
            save_loss_csv(loss_history)
            print(f"  📊 CSV saved: {LOSS_CSV_FILE}")
            generate_chart(loss_history)
            print()
            try:
                import mlflow
                if mlflow.active_run():
                    mlflow.log_artifact(CACHE_FILE)
                    mlflow.log_artifact(LOSS_CSV_FILE)
                    mlflow.log_artifact(LOSS_HTML_FILE)
                    mlflow.end_run()
            except ImportError:
                pass

    print("🧪 Generation test:")
    for word in ["warszawa", "polska", "kot", "wisła"]:
        result = generate_text(model, tokenizer, word, temperature=0.5)
        print(f"  '{word}' → {result}")
