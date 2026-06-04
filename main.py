# ============================================================
#  MAIN – trening i rozmowa z mini-GPT
#
#  Uruchom: python main.py
#  Wymagania: pip install numpy
#
#  Pliki:
#    main.py         ← ten plik
#    transformer.py  ← architektura modelu
#    tokenizer.py    ← słownik tokenów
#    dane.json       ← zdania treningowe
# ============================================================

import numpy as np
import json
import os
import hashlib
import random
from pathlib import Path
from mini_gpt.transformer import MiniGPT, Adam, URZADZENIE
from mini_gpt.tokenizer   import Tokenizer

# ============================================================
# USTAWIENIA
# ============================================================

PLIK_DANYCH     = "data/dane.json"

KATALOG_EKSPORT  = Path("exports")
PLIK_EKSPORTU    = KATALOG_EKSPORT / "model_export.pt"
KATALOG_OUTPUTS  = Path("outputs")
PLIK_LOSS_CSV    = KATALOG_OUTPUTS / "loss_curve.csv"
PLIK_LOSS_HTML   = KATALOG_OUTPUTS / "loss_curve.html"
PLIK_CACHE       = Path("exports") / "model_cache.pkl"

WYMIAR          = 128     # większy wymiar = więcej pojemności
N_WARSTW        = 4        # więcej warstw = głębszy model
N_GLOWIC        = 4        # głowice Multi-Head Attention
DROPOUT         = 0.05     # mniej dropout = lepsze zapamiętanie
EPOKI           = 3000     # więcej epok = lepsze zapamiętanie
LR              = 0.001    # wyższy LR = szybsza nauka)
MAKS_DLUGOSC    = 256      # dłuższy kontekst
BATCH_SIZE      = 32       # ← NOWE

# Przyspieszenie GPU – cuDNN automatycznie dobiera najszybszy algorytm
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True

# ============================================================
# CACHE – zapis i wczytywanie modelu
# ============================================================

def hash_danych(sciezka):
    """Oblicza hash pliku dane.json – wykrywa zmiany danych."""
    with open(sciezka, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def zapisz_cache(model, tokenizer, hash_pliku):
    """
    Zapisuje model PyTorch i tokenizer.
    Używamy torch.save – to właściwy sposób zapisu modeli PyTorch.
    """
    import torch
    dane = {
        "hash":        hash_pliku,
        "tokenizer":   tokenizer,
        "state_dict":  model.state_dict(),   # wszystkie wagi modelu
        "config": {                           # architektura modelu
            "rozmiar_slownika": tokenizer.rozmiar,
            "wymiar":           model.wymiar,
            "maks_dlugosc":     model.maks_dlugosc,
        }
    }
    PLIK_CACHE.parent.mkdir(exist_ok=True)
    torch.save(dane, PLIK_CACHE)
    eksportuj_model(model, tokenizer, hash_pliku)


def znajdz_plik_eksportu():
    kandydaci = [
        PLIK_EKSPORTU,
        Path("model_export.pt"),
        Path("v1") / "model_export.pt",
    ]
    for sciezka in kandydaci:
        if sciezka.exists():
            return sciezka
    return PLIK_EKSPORTU

def wczytaj_cache(model, hash_pliku):
    """
    Wczytuje model z cache jeśli dane.json się nie zmieniło.
    Zwraca (tokenizer, True) lub (None, False).
    """
    import torch
    from mini_gpt.transformer import URZADZENIE

    if not PLIK_CACHE.exists():
        return None, False

    try:
        dane = torch.load(PLIK_CACHE, map_location=URZADZENIE, weights_only=False)
        model.load_state_dict(dane["state_dict"])
    except ModuleNotFoundError:
        print(f"  ⚠️  Stary format cache — usuwam '{PLIK_CACHE}' i trenuję od nowa.")
        PLIK_CACHE.unlink()
        return None, False
    except Exception:
        return None, False

    if dane["hash"] != hash_pliku:
        print("  ⚠️  Dane zmieniły się – trenuję od nowa.\n")
        return None, False

    # Wczytaj wagi do modelu
    model.load_state_dict(dane["state_dict"])

    return dane["tokenizer"], True

def eksportuj_model(model, tokenizer, hash_pliku, sciezka=PLIK_EKSPORTU):
    """
    Zapisuje skompresowany model do wysyłania na GitHuba.
    Rozmiar: ~5-20MB zamiast ~150MB
    """
    import torch
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    dane = {
        "hash":       hash_pliku,
        "tokenizer":  tokenizer,
        "state_dict": {k: v.half()
                       for k, v in model.state_dict().items()},
        "config": {
            "rozmiar_slownika": tokenizer.rozmiar,
            "wymiar":           model.wymiar,
            "maks_dlugosc":     model.maks_dlugosc,
            "n_warstw":         N_WARSTW,
            "n_glowic":         N_GLOWIC,
            "dropout":          DROPOUT,
        }
    }
    torch.save(dane, sciezka, _use_new_zipfile_serialization=True)
    rozmiar = os.path.getsize(sciezka) / 1024 / 1024
    print(f"  📦 Eksport do '{sciezka}': {rozmiar:.1f} MB (gotowy na GitHub)")

def wczytaj_eksport(model, sciezka=None):
    """
    Wczytuje skompresowany model na słabszym sprzęcie.
    """
    import torch
    from mini_gpt.transformer import URZADZENIE

    if sciezka is None:
        sciezka = znajdz_plik_eksportu()

    sciezka = Path(sciezka)

    if not sciezka.exists():
        print(f"  ❌ Nie znaleziono '{sciezka}'")
        return None, False

    try:
        dane = torch.load(sciezka, map_location=URZADZENIE, weights_only=False)
    except ModuleNotFoundError:
        print(f"  ⚠️  Stary format eksportu — usuwam '{sciezka}' i trenuję od nowa.")
        sciezka.unlink()
        return None, False

    state = {k: v.float() for k, v in dane["state_dict"].items()}
    model.load_state_dict(state)

    rozmiar = os.path.getsize(sciezka) / 1024 / 1024
    print(f"  ✅ Wczytano eksport '{sciezka}' ({rozmiar:.1f} MB)")
    return dane["tokenizer"], True

# ============================================================
# DIAGNOSTYKA – zapis straty i wykres
# ============================================================

def zapisz_loss_csv(historia_strat):
    """Zapisuje historię strat do CSV (epoka, strata, perplexity)."""
    KATALOG_OUTPUTS.mkdir(exist_ok=True)
    with open(PLIK_LOSS_CSV, "w", encoding="utf-8") as f:
        f.write("epoka,strata,perplexity\n")
        for epoka, strata in historia_strat:
            perp = float(torch.exp(torch.tensor(strata)).item())
            f.write(f"{epoka},{strata:.6f},{perp:.4f}\n")

def generuj_wykres(historia_strat):
    """Generuje interaktywny wykres HTML z Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  ⚠️  Brak plotly – zainstaluj: pip install plotly")
        return

    epoki   = [e for e, _ in historia_strat]
    straty  = [s for _, s in historia_strat]
    perplxy = [float(torch.exp(torch.tensor(s)).item()) for _, s in historia_strat]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Strata (Cross-Entropy Loss)", "Perplexity"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(x=epoki, y=straty, mode="lines", name="Loss",
                   line=dict(color="#6366f1", width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epoki, y=perplxy, mode="lines", name="Perplexity",
                   line=dict(color="#f59e0b", width=1.5)),
        row=2, col=1,
    )
    fig.update_layout(
        title="Mini-GPT — krzywa uczenia",
        template="plotly_dark",
        height=600,
        showlegend=False,
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.write_html(PLIK_LOSS_HTML)
    print(f"  📈 Wykres zapisany: {PLIK_LOSS_HTML}")

# ============================================================
# KROK 1: WCZYTAJ DANE
# ============================================================

def wczytaj_dane(sciezka):
    if not os.path.exists(sciezka):
        print(f"❌ Nie znaleziono '{sciezka}'!")
        exit(1)
    with open(sciezka, encoding="utf-8") as f:
        dane = json.load(f)
    return dane.get("zdania", [])

# ============================================================
# KROK 2: FUNKCJA STRATY
# ============================================================

def cross_entropy_loss(logits, cel_ids):
    """Zostawiamy dla kompatybilności – używana tylko w starej wersji"""
    import numpy as np
    from mini_gpt.transformer import softmax
    T     = len(cel_ids)
    probs = softmax(logits)
    probs = np.clip(probs, 1e-9, 1.0)
    strata = -np.log(probs[np.arange(T), cel_ids]).mean()
    grad   = probs.copy()
    grad[np.arange(T), cel_ids] -= 1
    grad  /= T
    return strata, grad

# ============================================================
# KROK 3: TRENING (PyTorch)
# ============================================================

def zbuduj_batch(zdania_ids, batch_size, maks_dlugosc):
    probka = random.sample(zdania_ids, min(batch_size, len(zdania_ids)))
    probka = [ids[:maks_dlugosc] for ids in probka if len(ids) >= 2]
    dlugosc = max(len(ids) for ids in probka)

    wejscie_lista = []
    cel_lista     = []
    for ids in probka:
        w = ids[:-1]
        c = ids[1:]
        pad_len = dlugosc - 1 - len(w)
        w = w + [0] * pad_len
        c = c + [0] * pad_len
        wejscie_lista.append(w)
        cel_lista.append(c)

    wejscie = torch.tensor(wejscie_lista, dtype=torch.long, device=URZADZENIE)
    cel     = torch.tensor(cel_lista,     dtype=torch.long, device=URZADZENIE)
    return wejscie, cel

def trenuj(model, optymalizator, zdania_ids):
    kryterium = torch.nn.CrossEntropyLoss(ignore_index=0)
    n_batchy  = max(1, min(20, len(zdania_ids) // BATCH_SIZE))
    calkowita_strata = 0.0

    for _ in range(n_batchy):
        wejscie, cel = zbuduj_batch(zdania_ids, BATCH_SIZE, MAKS_DLUGOSC)
        optymalizator.zeruj_gradienty()
        logits, _ = model.forward(wejscie)       # (B, T, V)
        B, T, V = logits.shape
        strata = kryterium(logits.reshape(B * T, V), cel.reshape(B * T))
        calkowita_strata += strata.item()
        strata.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optymalizator.krok()

    return calkowita_strata / n_batchy

# ============================================================
# SAMPLING – Top-k i Top-p (nucleus sampling)
# ============================================================

def top_k_top_p_sampling(logits_np, top_k=0, top_p=1.0, temperatura=1.0):
    """
    Zaawansowane próbkowanie z rozkładu logitów.

    top_k  — zostaw tylko k najbardziej prawdopodobnych tokenów (0 = wyłączone)
    top_p  — zostaw tokeny których łączne prawdopodobieństwo <= p (1.0 = wyłączone)
              nazywane też nucleus sampling
    temperatura — skaluje rozkład przed próbkowaniem

    Zwraca indeks wybranego tokenu (int).
    """
    logits = logits_np.astype(np.float64).copy()
    logits = logits / max(temperatura, 0.01)

    # Top-k: zeruj wszystko poza k największymi
    # (top_k nie może przekroczyć rozmiaru słownika)
    if top_k > 0:
        k = min(top_k, len(logits))
        prog = np.sort(logits)[-k]
        logits[logits < prog] = -1e10

    # Softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # Top-p (nucleus): zostaw minimalny zestaw tokenów sumujący się do p
    if top_p < 1.0:
        posortowane_idx   = np.argsort(probs)[::-1]
        skumulowane_probs = np.cumsum(probs[posortowane_idx])

        # Znajdź próg — pierwszy indeks gdzie suma przekracza top_p
        prog_idx = np.searchsorted(skumulowane_probs, top_p) + 1
        odrzucone = posortowane_idx[prog_idx:]
        probs[odrzucone] = 0.0
        probs /= probs.sum()

    if temperatura < 0.05:
        return int(np.argmax(probs))
    return int(np.random.choice(len(probs), p=probs))

# ============================================================
# KROK 4: GENEROWANIE TEKSTU
# ============================================================

def generuj(model, tokenizer, tekst_start, max_znakow=60, temperatura=1.0):
    ids = tokenizer.koduj(tekst_start)

    with torch.no_grad():
        for _ in range(max_znakow):
            wejscie = ids[-MAKS_DLUGOSC:]
            logits,_  = model.forward(wejscie)

            ostatnie = logits[-1].cpu().numpy()
            nastepny = top_k_top_p_sampling(
                ostatnie, top_k=50, top_p=0.9, temperatura=temperatura
            )

            ids.append(nastepny)

            tekst_do_tej_pory = tokenizer.dekoduj(ids)
            if "koniec" in tekst_do_tej_pory[-10:]:
                break

    return tokenizer.dekoduj(ids)

# ============================================================
# GŁÓWNY PROGRAM
# ============================================================

if __name__ == "__main__":
    import sys
    import io

    # Wymuszenie UTF-8 dla terminala Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 55)
    print("  MINI-GPT – Transformer od zera!")
    print("=" * 55 + "\n")

    # 1. Dane + hash
    print("📚 Wczytuję dane...")
    zdania = wczytaj_dane(PLIK_DANYCH)
    aktualny_hash = hash_danych(PLIK_DANYCH)
    print(f"  Załadowano {len(zdania)} zdań.\n")

    # 2. Model (zawsze tworzymy strukturę)
    print("🧠 Tworzę model...")

    # Najpierw budujemy tokenizer żeby znać rozmiar słownika
    tokenizer_temp = Tokenizer()
    tokenizer_temp.buduj_slownik(zdania)

    from mini_gpt.transformer import URZADZENIE
    model = MiniGPT(
        rozmiar_slownika = tokenizer_temp.rozmiar,
        wymiar           = WYMIAR,
        n_warstw         = N_WARSTW,
        n_glowic         = N_GLOWIC,
        dropout          = DROPOUT,
        maks_dlugosc     = MAKS_DLUGOSC,
    ).to(URZADZENIE)   # ← przenosi wszystkie wagi modelu na GPU
    print()

    # 3. Sprawdź cache
    tokenizer_z_cache, cache_ok = wczytaj_cache(model, aktualny_hash)

    if cache_ok:
        tokenizer = tokenizer_z_cache
        model.ustaw_trening(False)
        print("✅ Wczytano wytrenowany model z cache!")
        print("   (dane.json nie zmieniło się – trening pominięty)\n")

    else:
        # spróbuj wczytać eksport z GitHuba
        tokenizer_export, eksport_ok = wczytaj_eksport(model)
        if eksport_ok:
            tokenizer = tokenizer_export
            model.ustaw_trening(False)
            print(f"✅ Wczytano {PLIK_EKSPORTU} – pomijam trening\n")
        else:
            # ── Brak cache i eksportu – trenujemy ──────────
            tokenizer  = tokenizer_temp
            zdania_ids = [tokenizer.koduj(z) for z in zdania]
            optymalizator = Adam(lr=LR, parametry=model.parameters())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optymalizator._opt, T_max=EPOKI
            )

            try:
                from tqdm import tqdm
                ma_tqdm = True
            except ImportError:
                ma_tqdm = False
                print("  💡 Wskazówka: zainstaluj tqdm dla ładniejszego paska:")
                print("     pip install tqdm\n")

            print(f"⚙️  Trenuję przez {EPOKI} epok...\n")

            historia_strat = []
            model.ustaw_trening(True)

            if ma_tqdm:
                pasek = tqdm(
                    range(1, EPOKI + 1),
                    desc="  Trening",
                    unit="epoka",
                    bar_format="  {l_bar}{bar:40}{r_bar}",
                    dynamic_ncols=True,
                )
                for epoka in pasek:
                    strata = trenuj(model, optymalizator, zdania_ids)
                    scheduler.step()
                    
                    if epoka % 100 == 0:
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save({
                            'epoch': epoka,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optymalizator._opt.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': strata,
                        }, f"checkpoints/checkpoint_epoch_{epoka}.pt")
                    
                    perplexity = float(torch.exp(torch.tensor(strata)).item())
                    pasek.set_postfix(strata=f"{strata:.4f}", perplexity=f"{perplexity:.2f}")
                    historia_strat.append((epoka, strata))
                    
            else:
                for epoka in range(1, EPOKI + 1):
                    strata = trenuj(model, optymalizator, zdania_ids)
                    scheduler.step()
                    
                    if epoka % 100 == 0:
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save({
                            'epoch': epoka,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optymalizator._opt.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': strata,
                        }, f"checkpoints/checkpoint_epoch_{epoka}.pt")
                        print(f"    💾 Checkpoint: epoch_{epoka}.pt")
                
                    historia_strat.append((epoka, strata))
                    if epoka % 100 == 0 or epoka == EPOKI:
                        perplexity = float(torch.exp(torch.tensor(strata)).item())
                        proc = epoka / EPOKI * 100
                        print(f"  Epoka {epoka}/{EPOKI} ({proc:.0f}%)  strata: {strata:.4f}  perplexity: {perplexity:.2f}")

            print("\n  ✅ Trening zakończony!")
            model.ustaw_trening(False)
            zapisz_cache(model, tokenizer, aktualny_hash)
            print(f"  💾 Model zapisany do '{PLIK_CACHE}'")
            zapisz_loss_csv(historia_strat)
            print(f"  📊 CSV zapisany: {PLIK_LOSS_CSV}")
            generuj_wykres(historia_strat)
            print()

    # 4. Test
    print("🧪 Test generowania:")
    for slowo in ["warszawa", "polska", "kot", "wisła"]:
        wynik = generuj(model, tokenizer, slowo, temperatura=0.5)
        print(f"  '{slowo}' → {wynik}")

    # 6. Tryb rozmowy z pamięcią
    print("\n" + "═" * 55)
    print("  💬 TRYB ROZMOWY – model pamięta kontekst!")
    print("═" * 55)
    print("  Przykłady pytań:")
    print("    co to jest kot")
    print("    gdzie jest warszawa")
    print("    kim jesteś")
    print()
    print("  Komendy: /temp 0.1 | /historia | /zapomnij | /pomoc | koniec")
    print("═" * 55 + "\n")

    temperatura   = 0.1   # niska = pewne odpowiedzi
    OKNO_PAMIECI  = 3     # ile ostatnich wymian model pamięta

    # Historia rozmowy – lista par (pytanie, odpowiedź)
    historia = []

    id_koniec = tokenizer.slowo_na_id.get("koniec", -1)

    def buduj_kontekst(historia, nowe_pytanie):
        """
        Buduje wejście dla modelu z historią rozmowy.

        Przykład z historią 2 wymian:
          użytkownik masz kota asystent tak mam kota
          użytkownik ile kotów asystent [tu model generuje]

        Im więcej historii → model lepiej rozumie kontekst.
        Ale za dużo → przekracza MAKS_DLUGOSC, więc bierzemy
        tylko ostatnie OKNO_PAMIECI wymian.
        """
        czesci = []

        # Dodaj ostatnie N wymian jako kontekst
        for (stare_pyt, stara_odp) in historia[-OKNO_PAMIECI:]:
            czesci.append(f"użytkownik {stare_pyt} asystent {stara_odp}")

        # Dodaj nowe pytanie
        czesci.append(f"użytkownik {nowe_pytanie} asystent")

        return " ".join(czesci)

    # Mapa polskich znaków – naprawia literówki bez ogonków
    BEZ_OGONKOW = {
        "a": "ą", "c": "ć", "e": "ę", "l": "ł", "n": "ń",
        "o": "ó", "s": "ś", "z": "ź", "x": "ż",
    }

    def napraw_ogonki(slowo):
        """
        Próbuje dopasować słowo bez ogonków do słownika modelu.
        Przykład: "jestes" → "jesteś", "czesc" → "cześć"
        """
        if tokenizer.koduj(slowo)[0] != tokenizer.UNK:
            return slowo  # słowo znane – nie zmieniaj

        # Spróbuj podmienić ostatnią literę na wersję z ogonkiem
        for i, litera in enumerate(slowo):
            if litera in BEZ_OGONKOW:
                kandydat = slowo[:i] + BEZ_OGONKOW[litera] + slowo[i+1:]
                if tokenizer.koduj(kandydat)[0] != tokenizer.UNK:
                    return kandydat
        return slowo  # nie znaleziono – zwróć oryginał


    def generuj_odpowiedz(pytanie, historia, temperatura):
        pytanie = pytanie.lower().strip().rstrip("?")
        kontekst = f"użytkownik {pytanie} asystent"
        ids = tokenizer.koduj(kontekst)

        with torch.no_grad():
            for _ in range(300):
                logits, _ = model.forward(ids[-MAKS_DLUGOSC:])
                ostatnie = logits[-1].cpu().numpy()
                nastepny = top_k_top_p_sampling(
                    ostatnie, top_k=50, top_p=0.9, temperatura=temperatura
                )
                ids.append(nastepny)

                tekst = tokenizer.dekoduj(ids)
                if "koniec" in tekst[-10:]:
                    break

        tekst = tokenizer.dekoduj(ids)

        # wyciągnij ostatnią odpowiedź po "asystent"
        if "asystent" in tekst:
            idx = tekst.rfind("asystent") + len("asystent")
            odpowiedz = tekst[idx:]
        else:
            odpowiedz = tekst

        # utnij na "koniec" lub "użytkownik"
        for stop in ["koniec", "użytkownik"]:
            if stop in odpowiedz:
                odpowiedz = odpowiedz[:odpowiedz.index(stop)]

        odpowiedz = odpowiedz.strip()
        return odpowiedz if odpowiedz else "..."

    while True:
        try:
            wejscie = input("  Ty: ").strip()

            if not wejscie:
                continue

            # ── komendy ───────────────────────────────────────

            if wejscie.lower() == "koniec":
                print("\n  Do zobaczenia! 👋\n")
                break

            if wejscie.startswith("/temp"):
                czesci = wejscie.split()
                if len(czesci) == 2:
                    try:
                        temperatura = float(czesci[1])
                        print(f"  ✅ Temperatura: {temperatura}\n")
                    except ValueError:
                        print("  ⚠️  Użycie: /temp 0.1\n")
                continue

            if wejscie == "/historia":
                if not historia:
                    print("  (brak historii)\n")
                else:
                    print("\n  📜 Historia rozmowy:")
                    for i, (p, o) in enumerate(historia, 1):
                        print(f"  {i}. Ty:    {p}")
                        print(f"     Model: {o}")
                    print()
                continue

            if wejscie == "/zapomnij":
                historia.clear()
                print("  🗑️  Historia wyczyszczona.\n")
                continue

            if wejscie == "/pomoc":
                print("""
  Komendy:
    /temp 0.1    → temperatura (0.01=zawsze to samo, 1.0=losowy)
    /historia    → pokaż historię rozmowy
    /zapomnij    → wyczyść pamięć modelu
    koniec       → zakończ rozmowę

  Jak działa pamięć:
    Model pamięta ostatnie 3 wymiany.
    Użyj /zapomnij żeby zacząć nowy temat.
                """)
                continue

            # ── generuj odpowiedź ──────────────────────────────

            odpowiedz = generuj_odpowiedz(wejscie, historia, temperatura)
            print(f"  🤖 Model: {odpowiedz}\n")

            # Zapisz do historii (tylko jeśli odpowiedź sensowna)
            if odpowiedz != "...":
                pytanie_clean = wejscie.lower().strip().rstrip("?")
                if len(pytanie_clean.split()) == 1:
                    pytanie_clean = f"co to jest {pytanie_clean}"
                historia.append((pytanie_clean, odpowiedz))

        except KeyboardInterrupt:
            print("\n\n  Program zakończony.\n")
            break