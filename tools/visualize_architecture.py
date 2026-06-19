"""
Uproszczona wizualizacja architektury mini-GPT.
Użycie: python tools/visualize_architecture.py

Generuje outputs/architecture.html — diagram wysokiego poziomu
z wymiarami tensorów przepływających przez sieć.
"""

import sys
import os
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_gpt.utils import DEVICE

PLIK_CACHE      = Path("exports") / "model_cache.pkl"
KATALOG_OUTPUTS = Path("outputs")
PLIK_WYNIKI     = KATALOG_OUTPUTS / "architecture.html"


def wczytaj_config():
    if not PLIK_CACHE.exists():
        print("❌ Brak exports/model_cache.pkl — uruchom najpierw main.py")
        sys.exit(1)
    dane = torch.load(PLIK_CACHE, map_location="cpu", weights_only=False)
    return dane["config"], dane["tokenizer"]


def generuj_html(cfg, tokenizer):
    V = cfg["rozmiar_slownika"]
    D = cfg["wymiar"]
    T = cfg["maks_dlugosc"]
    n_warstw  = cfg.get("n_warstw",  4)
    n_glowic  = cfg.get("n_glowic",  4)
    d_glowica = D // n_glowic
    d_ff      = D * 4

    # Węzły diagramu — (id, etykieta, opis, kolor)
    wezly = [
        ("input",    "Input",             f"sekwencja tokenów\nshape: [B, T]\nT ≤ {T}",                         "#6366f1"),
        ("tok_emb",  "Token Embedding",   f"Embedding(vocab={V}, dim={D})\nshape: [B, T, {D}]",                "#8b5cf6"),
        ("pos_emb",  "Positional Emb.",   f"Embedding(max_len={T}, dim={D})\nshape: [B, T, {D}]",              "#8b5cf6"),
        ("add_emb",  "Add + Dropout",     f"tok_emb + pos_emb\nshape: [B, T, {D}]",                            "#a78bfa"),
        ("gpt_blok", f"GPT Block × {n_warstw}", (
            f"── LayerNorm({D})\n"
            f"── MultiHeadAttention\n"
            f"     heads={n_glowic}, d_head={d_glowica}\n"
            f"     Q/K/V shape: [B, T, {D}]\n"
            f"     attn shape: [B, {n_glowic}, T, T]\n"
            f"── Residual Add\n"
            f"── LayerNorm({D})\n"
            f"── FFN: {D}→{d_ff}→{D}\n"
            f"── GELU + Dropout\n"
            f"── Residual Add\n"
            f"output shape: [B, T, {D}]"
        ),                                                                                                        "#2563eb"),
        ("ln_f",     "LayerNorm",         f"LayerNorm({D})\nshape: [B, T, {D}]",                               "#0891b2"),
        ("glowa",    "Linear Head",       f"Linear({D} → {V}, bias=False)\nWeight Tying z Token Emb.\nshape: [B, T, {V}]", "#0f766e"),
        ("output",   "Output Logits",     f"rozkład prawdop. dla każdego tokenu\nshape: [B, T, {V}]\nnastępnie: Top-k/Top-p sampling", "#059669"),
    ]

    # Połączenia (od, do)
    polaczenia = [
        ("input",   "tok_emb"),
        ("input",   "pos_emb"),
        ("tok_emb", "add_emb"),
        ("pos_emb", "add_emb"),
        ("add_emb", "gpt_blok"),
        ("gpt_blok","ln_f"),
        ("ln_f",    "glowa"),
        ("glowa",   "output"),
    ]

    # Pozycje węzłów (x, y) w pikselach — canvas 900×1100
    pozycje = {
        "input":    (450, 40),
        "tok_emb":  (280, 160),
        "pos_emb":  (620, 160),
        "add_emb":  (450, 300),
        "gpt_blok": (450, 500),
        "ln_f":     (450, 720),
        "glowa":    (450, 840),
        "output":   (450, 980),
    }

    # Szerokości węzłów
    szer = {
        "input": 200, "tok_emb": 220, "pos_emb": 220,
        "add_emb": 220, "gpt_blok": 380, "ln_f": 220,
        "glowa": 280, "output": 300,
    }

    def box_h(id_):
        tekst = next(o for i, _, o, _ in wezly if i == id_)
        return max(60, tekst.count("\n") * 18 + 36)

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>mini-GPT — Architektura</title>
<style>
  body {{ margin: 0; background: #0f172a; font-family: 'Segoe UI', monospace; color: #e2e8f0; }}
  h1 {{ text-align: center; padding: 24px 0 0; font-size: 20px; font-weight: 500; color: #a5b4fc; }}
  p.sub {{ text-align: center; font-size: 13px; color: #64748b; margin: 4px 0 16px; }}
  .canvas-wrap {{ display: flex; justify-content: center; overflow-x: auto; padding: 0 24px 40px; }}
  svg text {{ font-family: 'Segoe UI', monospace; }}
  .node {{ cursor: pointer; }}
  .node:hover rect {{ filter: brightness(1.15); }}
  .tooltip {{
    position: fixed; background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 10px 14px; font-size: 12px; line-height: 1.7;
    color: #e2e8f0; pointer-events: none; display: none; max-width: 320px;
    white-space: pre-wrap; z-index: 100; box-shadow: 0 4px 24px #0008;
  }}
  .legenda {{ display:flex; gap:20px; justify-content:center; flex-wrap:wrap;
              font-size:12px; color:#94a3b8; padding-bottom:24px; }}
  .legenda span {{ display:flex; align-items:center; gap:6px; }}
  .legenda i {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
</style>
</head>
<body>
<h1>mini-GPT — Architektura modelu</h1>
<p class="sub">Kliknij węzeł aby zobaczyć szczegóły · vocab={V} · d_model={D} · {n_warstw} warstwy · {n_glowic} głowice · ~{sum(1 for _ in [0])*830592:,} parametrów</p>
<div class="legenda">
  <span><i style="background:#6366f1"></i> Wejście/wyjście</span>
  <span><i style="background:#8b5cf6"></i> Embeddingi</span>
  <span><i style="background:#2563eb"></i> GPT Block</span>
  <span><i style="background:#0891b2"></i> Normalizacja</span>
  <span><i style="background:#059669"></i> Projekcja wyjściowa</span>
</div>
<div class="canvas-wrap">
<svg id="svg" width="900" height="1060" viewBox="0 0 900 1060">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#475569" stroke-width="1.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
"""

    # Rysuj połączenia
    for (src, dst) in polaczenia:
        sx, sy = pozycje[src]
        dx, dy = pozycje[dst]
        sh = box_h(src)
        # wyjście z dołu boxa źródłowego, wejście na górę docelowego
        x1, y1 = sx, sy + sh // 2
        x2, y2 = dx, dy - box_h(dst) // 2

        if abs(x1 - x2) < 5:
            # prosta linia pionowa
            html += (f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'stroke="#475569" stroke-width="1.5" '
                     f'marker-end="url(#arr)"/>\n')
        else:
            # łamana przez środek
            my = (y1 + y2) // 2
            html += (f'  <path d="M{x1},{y1} L{x1},{my} L{x2},{my} L{x2},{y2}" '
                     f'fill="none" stroke="#475569" stroke-width="1.5" '
                     f'marker-end="url(#arr)"/>\n')

    # Rysuj węzły
    for (id_, etyk, opis, kolor) in wezly:
        cx, cy = pozycje[id_]
        w  = szer[id_]
        h  = box_h(id_)
        x  = cx - w // 2
        y  = cy - h // 2

        # Escape dla JS
        opis_js = opis.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

        html += f"""  <g class="node" onclick="pokazTooltip(event, '{opis_js}')">
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10"
          fill="{kolor}" fill-opacity="0.25"
          stroke="{kolor}" stroke-width="1.5"/>
    <text x="{cx}" y="{cy - 8}" text-anchor="middle"
          font-size="13" font-weight="600" fill="{kolor}">{etyk}</text>
"""
        # Pierwsza linia opisu pod etykietą
        pierwsza = opis.split("\n")[0]
        html += (f'    <text x="{cx}" y="{cy + 10}" text-anchor="middle" '
                 f'font-size="10" fill="#94a3b8">{pierwsza}</text>\n')
        html += "  </g>\n"

    html += """</svg>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
function pokazTooltip(e, tekst) {
  const t = document.getElementById('tooltip');
  t.textContent = tekst.replace(/\\n/g, '\\n');
  t.style.display = 'block';
  t.style.left = (e.clientX + 16) + 'px';
  t.style.top  = (e.clientY - 10) + 'px';
}
document.addEventListener('click', function(e) {
  if (!e.target.closest('.node')) {
    document.getElementById('tooltip').style.display = 'none';
  }
});
document.addEventListener('mousemove', function(e) {
  const t = document.getElementById('tooltip');
  if (t.style.display === 'block') {
    t.style.left = (e.clientX + 16) + 'px';
    t.style.top  = (e.clientY - 10) + 'px';
  }
});
</script>
</body>
</html>"""

    return html


if __name__ == "__main__":
    print("🔍 Wczytuję konfigurację modelu...")
    cfg, tokenizer = wczytaj_config()

    print("🎨 Generuję diagram...")
    KATALOG_OUTPUTS.mkdir(exist_ok=True)
    html = generuj_html(cfg, tokenizer)

    with open(PLIK_WYNIKI, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Diagram zapisany: {PLIK_WYNIKI}")
    print()
    print("  Otwórz w przeglądarce — każdy węzeł jest klikalny")
    print("  i pokazuje szczegóły: shape tensorów, parametry warstwy.")