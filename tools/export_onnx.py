"""
Eksport modelu mini-GPT do formatu ONNX.
Użycie: python tools/export_onnx.py

Następnie wgraj outputs/model.onnx na https://netron.app
aby zobaczyć interaktywny graf architektury sieci.
"""

import sys
import os
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_gpt.transformer import MiniGPT, URZADZENIE

PLIK_CACHE      = Path("exports") / "model_cache.pkl"
KATALOG_OUTPUTS = Path("outputs")
PLIK_ONNX       = KATALOG_OUTPUTS / "model.onnx"

DLUGOSC_WEJSCIA = 16  # liczba tokenów w przykładowym wejściu


def wczytaj_model():
    if not PLIK_CACHE.exists():
        print("❌ Brak exports/model_cache.pkl — uruchom najpierw main.py")
        sys.exit(1)

    dane = torch.load(PLIK_CACHE, map_location=URZADZENIE, weights_only=False)
    cfg  = dane["config"]
    model = MiniGPT(
        rozmiar_slownika = cfg["rozmiar_slownika"],
        wymiar           = cfg["wymiar"],
        maks_dlugosc     = cfg["maks_dlugosc"],
    ).to(URZADZENIE)
    model.load_state_dict(dane["state_dict"])
    model.eval()
    print(f"✅ Model wczytany — {sum(p.numel() for p in model.parameters()):,} parametrów")
    return model, dane["tokenizer"]


def eksportuj_onnx(model, tokenizer):
    try:
        import onnx
    except ImportError:
        print("❌ Brak onnx — zainstaluj: pip install onnx")
        sys.exit(1)

    KATALOG_OUTPUTS.mkdir(exist_ok=True)

    # Przykładowe wejście — sekwencja tokenów
    dlugosc  = min(DLUGOSC_WEJSCIA, tokenizer.rozmiar - 1)
    dummy    = torch.zeros(1, dlugosc, dtype=torch.long, device=URZADZENIE)

    print(f"📐 Eksportuję z wejściem shape: {list(dummy.shape)}")

    # Wrapper — ONNX nie obsługuje zwracania krotek z None
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, ids):
            logits, _ = self.model(ids)
            return logits

    wrapper = ModelWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy,
        str(PLIK_ONNX),
        opset_version    = 14,
        input_names      = ["tokeny"],
        output_names     = ["logity"],
        dynamic_axes     = {
            "tokeny": {0: "batch", 1: "sekwencja"},
            "logity": {0: "batch", 1: "sekwencja"},
        },
        do_constant_folding = True,
    )

    # Weryfikacja poprawności pliku
    model_onnx = onnx.load(str(PLIK_ONNX))
    onnx.checker.check_model(model_onnx)

    rozmiar = PLIK_ONNX.stat().st_size / 1024 / 1024
    print(f"✅ Eksport zakończony: {PLIK_ONNX} ({rozmiar:.1f} MB)")
    print()
    print("─" * 50)
    print("  Jak otworzyć graf architektury:")
    print("  1. Wejdź na https://netron.app")
    print("  2. Kliknij 'Open Model'")
    print(f"  3. Wybierz plik: {PLIK_ONNX.resolve()}")
    print("  4. Klikaj w węzły grafu aby zobaczyć szczegóły operacji")
    print("─" * 50)


if __name__ == "__main__":
    model, tokenizer = wczytaj_model()
    eksportuj_onnx(model, tokenizer)