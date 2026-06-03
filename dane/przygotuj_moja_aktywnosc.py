"""Przetwarza eksport Gemini do czystego zbioru treningowego.

Skrypt nie modyfikuje oryginalnego pliku "RAW.json".
Domyślnie tworzy:
- "gemini_aktywnosc_trening.jsonl" - rekordy w formacie JSONL
- "gemini_aktywnosc_messages.jsonl" - rekordy gotowe do treningu w formacie messages
- "gemini_aktywnosc_przeglad.md" - wersja do ręcznego przeglądu
- "gemini_aktywnosc_odrzucone.jsonl" - rekordy odrzucone przez heurystyki
- "gemini_aktywnosc_raport.json" - statystyki przetwarzania

Uruchomienie z katalogu repozytorium:
    python dane/przygotuj_moja_aktywnosc.py

Można też podać własne ścieżki:
    python dane/przygotuj_moja_aktywnosc.py --input dane/RAW.json --output dane/wyjscie.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PREFIX_ZAPYTANIA = "Wysłano zapytanie:"
PROMPT_NOISE_PREFIXES = (
    "Opcje treści:",
    "Wybierz dane do pobrania",
    "Wybierz konkretne dane o aktywności",
    "Wybrano ",
    "Usługi",
    "Android",
    "Aplikacje z Gemini",
    "Asystent",
    "Developers",
    "Takeout",
)


class _HTMLStripper:
    def __init__(self) -> None:
        from html.parser import HTMLParser

        class Stripper(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.parts: list[str] = []

            def handle_data(self, data: str) -> None:
                self.parts.append(data)

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag in {"p", "div", "br", "li", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
                    self.parts.append("\n")

        self._parser = Stripper()

    def feed(self, value: str) -> None:
        self._parser.feed(value)

    def text(self) -> str:
        return "".join(self._parser.parts)


@dataclass
class RekordTreningowy:
    source: str
    timestamp: str | None
    prompt: str
    response: str
    header: str | None = None
    products: list[str] | None = None
    attachments: list[str] | None = None
    record_hash: str | None = None


@dataclass
class RekordOdrzucony:
    source: str
    timestamp: str | None
    prompt: str
    response_preview: str | None
    reason: str


@dataclass
class RekordMessages:
    messages: list[dict[str, str]]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "rekord"


def normalize_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def strip_html(value: str) -> str:
    parser = _HTMLStripper()
    parser.feed(html.unescape(value))
    text = parser.text()
    text = re.sub(r"\n[ \t]+", "\n", text)
    return normalize_whitespace(text)


def extract_prompt(title: str) -> str:
    prompt = title.strip()
    if prompt.startswith(PREFIX_ZAPYTANIA):
        prompt = prompt[len(PREFIX_ZAPYTANIA) :].strip()

    cleaned_lines: list[str] = []
    for raw_line in prompt.splitlines():
        line = normalize_whitespace(raw_line)
        if not line:
            continue
        if any(line.startswith(prefix) for prefix in PROMPT_NOISE_PREFIXES):
            break
        cleaned_lines.append(line)

    if cleaned_lines:
        return normalize_whitespace("\n".join(cleaned_lines))
    return normalize_whitespace(prompt)


def looks_like_noise_prompt(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    noisy_fragments = (
        "opcje treści",
        "wybierz dane do pobrania",
        "wybrano 0 z",
        "aktywność w aplikacjach z gemini",
        "moja aktywność",
        "takeout",
    )
    return any(fragment in prompt_lower for fragment in noisy_fragments)


def looks_like_noise_response(response: str) -> bool:
    response_lower = response.lower()
    noisy_fragments = (
        "kliknij następny krok",
        "utwórz eksport",
        "odznacz wszystko",
        "wybierz z tej listy opcję",
        "pliku z historią czatów",
        "właściwa procedura krok po kroku",
    )
    return any(fragment in response_lower for fragment in noisy_fragments)


def stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def to_messages_record(rekord: RekordTreningowy) -> RekordMessages:
    return RekordMessages(
        messages=[
            {"role": "user", "content": rekord.prompt},
            {"role": "assistant", "content": rekord.response},
        ]
    )


def format_markdown_record(rekord: RekordTreningowy, index: int) -> str:
    title = rekord.prompt.splitlines()[0].strip() if rekord.prompt else f"Rekord {index}"
    title = title[:120]
    lines = [
        f"## {index}. {title}",
        f"- source: {rekord.source}",
    ]
    if rekord.timestamp:
        lines.append(f"- timestamp: {rekord.timestamp}")
    if rekord.header:
        lines.append(f"- header: {rekord.header}")
    if rekord.products:
        lines.append(f"- products: {', '.join(rekord.products)}")
    if rekord.attachments:
        lines.append(f"- attachments: {', '.join(rekord.attachments)}")
    lines.extend([
        "",
        "### Prompt",
        "",
        rekord.prompt,
        "",
        "### Odpowiedź",
        "",
        rekord.response,
        "",
        "---",
        "",
    ])
    return "\n".join(lines)


def iter_records(items: Iterable[dict[str, Any]]) -> tuple[list[RekordTreningowy], list[RekordOdrzucony], dict[str, int]]:
    wynik: list[RekordTreningowy] = []
    odrzucone: list[RekordOdrzucony] = []
    statystyki = {
        "wszystkie": 0,
        "z_tekstem": 0,
        "bez_podwajania": 0,
        "bez_prompta": 0,
        "bez_odpowiedzi": 0,
        "odrzucone_szum": 0,
    }
    seen: set[str] = set()

    for item in items:
        statystyki["wszystkie"] += 1

        title = str(item.get("title", ""))
        prompt = extract_prompt(title)
        if not prompt:
            statystyki["bez_prompta"] += 1
            continue
        if looks_like_noise_prompt(prompt):
            statystyki["odrzucone_szum"] += 1
            odrzucone.append(
                RekordOdrzucony(
                    source=str(item.get("header") or (item.get("products") or ["Gemini"])[0]),
                    timestamp=str(item.get("time")) if item.get("time") else None,
                    prompt=prompt,
                    response_preview=None,
                    reason="prompt_looks_like_ui_noise",
                )
            )
            continue

        safe_html_items = item.get("safeHtmlItem") or []
        html_chunks: list[str] = []
        for chunk in safe_html_items:
            html_value = chunk.get("html") if isinstance(chunk, dict) else None
            if isinstance(html_value, str) and html_value.strip():
                html_chunks.append(html_value)

        if not html_chunks:
            statystyki["bez_odpowiedzi"] += 1
            continue

        response = normalize_whitespace("\n\n".join(strip_html(chunk) for chunk in html_chunks))
        if not response:
            statystyki["bez_odpowiedzi"] += 1
            continue
        if looks_like_noise_response(response):
            statystyki["odrzucone_szum"] += 1
            odrzucone.append(
                RekordOdrzucony(
                    source=str(item.get("header") or (item.get("products") or ["Gemini"])[0]),
                    timestamp=str(item.get("time")) if item.get("time") else None,
                    prompt=prompt,
                    response_preview=response[:300],
                    reason="response_looks_like_ui_noise",
                )
            )
            continue

        if len(prompt) < 8 or len(response) < 20:
            statystyki["odrzucone_szum"] += 1
            odrzucone.append(
                RekordOdrzucony(
                    source=str(item.get("header") or (item.get("products") or ["Gemini"])[0]),
                    timestamp=str(item.get("time")) if item.get("time") else None,
                    prompt=prompt,
                    response_preview=response[:300],
                    reason="too_short",
                )
            )
            continue

        statystyki["z_tekstem"] += 1

        rekord_podpis = stable_hash({"prompt": prompt, "response": response})
        if rekord_podpis in seen:
            continue
        seen.add(rekord_podpis)
        statystyki["bez_podwajania"] += 1

        wynik.append(
            RekordTreningowy(
                source=str(item.get("header") or (item.get("products") or ["Gemini"])[0]),
                timestamp=str(item.get("time")) if item.get("time") else None,
                prompt=prompt,
                response=response,
                header=str(item.get("header")) if item.get("header") else None,
                products=[str(x) for x in item.get("products", []) if x is not None] or None,
                attachments=[str(x) for x in item.get("attachedFiles", []) if x is not None] or None,
                record_hash=rekord_podpis,
            )
        )

    return wynik, odrzucone, statystyki


def main() -> int:
    parser = argparse.ArgumentParser(description="Przetwórz RAW.json do JSONL pod trening.")
    parser.add_argument(
        "--input",
        default=Path(__file__).with_name("RAW.json"),
        type=Path,
        help="Ścieżka do oryginalnego eksportu JSON.",
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).with_name("gemini_aktywnosc_trening.jsonl"),
        type=Path,
        help="Ścieżka do pliku wyjściowego JSONL.",
    )
    parser.add_argument(
        "--messages",
        default=Path(__file__).with_name("gemini_aktywnosc_messages.jsonl"),
        type=Path,
        help="Ścieżka do pliku JSONL w formacie messages gotowym do treningu.",
    )
    parser.add_argument(
        "--markdown",
        default=Path(__file__).with_name("gemini_aktywnosc_przeglad.md"),
        type=Path,
        help="Ścieżka do pliku Markdown do ręcznego przeglądu.",
    )
    parser.add_argument(
        "--report",
        default=Path(__file__).with_name("gemini_aktywnosc_raport.json"),
        type=Path,
        help="Ścieżka do pliku z raportem przetwarzania.",
    )
    parser.add_argument(
        "--rejected",
        default=Path(__file__).with_name("gemini_aktywnosc_odrzucone.jsonl"),
        type=Path,
        help="Ścieżka do pliku JSONL z rekordami odrzuconymi.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        dane = json.load(f)

    if not isinstance(dane, list):
        raise ValueError("Oczekiwano listy rekordów JSON w pliku wejściowym.")

    rekordy, odrzucone, statystyki = iter_records(dane)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rekord in rekordy:
            f.write(json.dumps(asdict(rekord), ensure_ascii=False) + "\n")

    with args.messages.open("w", encoding="utf-8") as f:
        for rekord in rekordy:
            f.write(json.dumps(asdict(to_messages_record(rekord)), ensure_ascii=False) + "\n")

    with args.markdown.open("w", encoding="utf-8") as f:
        f.write("# Gemini aktywność - przegląd rekordów\n\n")
        f.write(f"- źródło: {args.input}\n")
        f.write(f"- rekordy treningowe: {len(rekordy)}\n")
        f.write(f"- rekordy odrzucone: {len(odrzucone)}\n\n")
        for index, rekord in enumerate(rekordy, start=1):
            f.write(format_markdown_record(rekord, index))

    with args.rejected.open("w", encoding="utf-8") as f:
        for rekord in odrzucone:
            f.write(json.dumps(asdict(rekord), ensure_ascii=False) + "\n")

    raport = {
        "input": str(args.input),
        "output": str(args.output),
        "report": str(args.report),
        "counts": {
            **statystyki,
            "wyeksportowane": len(rekordy),
            "odrzucone": len(odrzucone),
        },
        "suggested_next_step": "Użyj gemini_aktywnosc_messages.jsonl jako wejścia do treningu, a gemini_aktywnosc_przeglad.md do ręcznej kontroli jakości.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(raport, f, ensure_ascii=False, indent=2)

    print(f"Zapisano {len(rekordy)} rekordów do {args.output}")
    print(f"Zapisano {len(rekordy)} rekordów messages do {args.messages}")
    print(f"Zapisano przegląd Markdown do {args.markdown}")
    print(f"Zapisano {len(odrzucone)} rekordów odrzuconych do {args.rejected}")
    print(f"Raport: {args.report}")
    print(f"Statystyki: {statystyki}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())