import json

# Wczytaj plik i napraw
with open("data/dane.json", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Usuń ostatnią linię jeśli to "*** End Patch"
while lines and "End Patch" in lines[-1]:
    lines.pop()

# Połóż linie razem
content = "".join(lines)

# Dodaj zamknięcie jeśli brakuje
content = content.rstrip()
if not content.endswith("]"):
    content = content.rstrip(",") + "\n  ]\n}\n"
else:
    content = content + "\n}\n"

# Weryfikuj JSON
try:
    data = json.loads(content)
    with open("data/dane.json", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ Naprawiono! Zdań: {len(data['zdania'])}")
except Exception as e:
    print(f"❌ Błąd: {e}")
