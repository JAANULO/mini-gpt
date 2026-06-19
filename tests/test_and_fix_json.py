import json
import os

def fix_and_test_json(filepath="data/dane.json"):
    if not os.path.exists(filepath):
        print(f"❌ Plik {filepath} nie istnieje.")
        return

    # Wczytaj plik i napraw
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Usuń ostatnią linię jeśli to "*** End Patch"
    while lines and "End Patch" in lines[-1]:
        lines.pop()

    # Połóż linie razem
    content = "".join(lines)

    # Dodaj zamknięcie jeśli brakuje
    content = content.rstrip()
    if not content.endswith("}"):
        if not content.endswith("]"):
            content = content.rstrip(",") + "\n  ]\n}"
        else:
            content = content + "\n}"

    # Weryfikuj JSON
    try:
        data = json.loads(content)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ JSON OK (Naprawiono/Sprawdzono)! Zdań: {len(data['zdania'])}")
    except Exception as e:
        print(f"❌ Błąd formatu JSON: {e}")

if __name__ == "__main__":
    fix_and_test_json()
