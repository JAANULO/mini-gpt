import json

try:
    data = json.load(open('data/dane.json'))
    print(f'✓ JSON OK - {len(data["zdania"])} zdań')
except Exception as e:
    print(f'❌ Błąd: {e}')
