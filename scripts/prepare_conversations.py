import os
import json
import glob
import sys
import time
from datetime import datetime

# Dodajemy główny folder projektu do ścieżki
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mini_gpt.config import cfg

def fix_mojibake(text):
    if not text:
        return ""
    try:
        return text.encode('iso-8859-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def is_valid_text(text, min_length=4):
    if not text:
        return False
    text_stripped = text.strip()
    if len(text_stripped) < min_length:
        return False
    # Pomiń same linki
    if text_stripped.startswith('http') and len(text_stripped.split()) == 1:
        return False
    # Pomiń standardowe wiadomości systemowe Messengera
    system_messages = [
        "przesyła załącznik.",
        "wysłał załącznik.",
        "wysłała załącznik.",
        "ustawił pseudonim",
        "ustawiła pseudonim",
        "zmienił kolor czatu",
        "zmieniła kolor czatu"
    ]
    if any(sm in text_stripped for sm in system_messages):
        return False
    return True

def main():
    USER_NAME = cfg.filter_user_name
    input_dir = cfg.messenger_raw_dir
    output_file = cfg.conversations_jsonl
    
    min_length = cfg.filter_min_length
    max_age_years = cfg.filter_max_age_years
    max_gap_minutes = cfg.filter_max_gap_minutes
    
    # Przelicz próg czasowy w milisekundach
    current_time_ms = time.time() * 1000
    time_threshold_ms = current_time_ms - (max_age_years * 365.25 * 24 * 3600 * 1000)
    max_gap_ms = max_gap_minutes * 60 * 1000
    
    print(f"Konfiguracja przygotowania danych:")
    print(f"  - Nadawca (Ty): '{USER_NAME}'")
    print(f"  - Folder wejściowy: {input_dir}")
    print(f"  - Plik wyjściowy: {output_file}")
    print(f"  - Min. długość wiadomości: {min_length} zn.")
    print(f"  - Maks. wiek wiadomości: {max_age_years} lat(a) (od {datetime.fromtimestamp(time_threshold_ms/1000).strftime('%Y-%m-%d')})")
    print(f"  - Maks. odstęp łączenia wątków: {max_gap_minutes} min.")

    if not os.path.exists(input_dir):
        print(f"[BLAD] Katalog wejściowy {input_dir} nie istnieje. Utwórz go i rozpakuj tam pliki Messengera.")
        return

    # Szukaj plików message_*.json
    file_paths = glob.glob(os.path.join(input_dir, "*", "message_*.json"))
    print(f"Znaleziono {len(file_paths)} plików z wiadomościami.")
    
    if len(file_paths) == 0:
        print("[!] Brak plików JSON Messengera w katalogu wejściowym. Upewnij się, że rozpakowałeś tam podfoldery czatów.")
        return

    # Przetwarzaj katalog po katalogu, by zachować kontekst pojedynczego czatu
    directories = {}
    for fp in file_paths:
        dirname = os.path.dirname(fp)
        if dirname not in directories:
            directories[dirname] = []
        directories[dirname].append(fp)
        
    all_pairs = []
    
    for dirname, files in directories.items():
        all_messages_in_chat = []
        for fp in files:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
                    # Napraw kodowanie w locie
                    for m in messages:
                        m["sender_name"] = fix_mojibake(m.get("sender_name", ""))
                        m["content"] = fix_mojibake(m.get("content", ""))
                    all_messages_in_chat.extend(messages)
            except Exception as e:
                print(f"Błąd czytania pliku {fp}: {e}")
                
        # Sortujemy od najstarszej
        all_messages_in_chat.sort(key=lambda x: x.get("timestamp_ms", 0))
        
        current_context = []
        current_response = []
        previous_timestamp = None
        
        for msg in all_messages_in_chat:
            sender = msg.get("sender_name")
            content = msg.get("content")
            timestamp = msg.get("timestamp_ms", 0)
            
            if not is_valid_text(content, min_length):
                continue
                
            # Filtrowanie po wieku
            if timestamp < time_threshold_ms:
                continue

            # Sprawdzamy odstęp czasowy
            gap_exceeded = False
            if previous_timestamp is not None:
                gap = timestamp - previous_timestamp
                if gap > max_gap_ms:
                    gap_exceeded = True
            
            previous_timestamp = timestamp
            
            if sender == USER_NAME:
                # Jeśli próg czasowy został przekroczony, a mieliśmy coś w buforze
                if gap_exceeded and len(current_response) > 0 and len(current_context) > 0:
                    all_pairs.append({
                        "messages": [
                            {"role": "user", "content": " \n ".join(current_context)},
                            {"role": "assistant", "content": " \n ".join(current_response)}
                        ]
                    })
                    current_context = []
                    current_response = []
                
                current_response.append(content)
            else:
                # To wiadomość od kogoś innego
                # Jeśli próg czasowy został przekroczony lub Janusz skończył odpowiadać
                if gap_exceeded or len(current_response) > 0:
                    if len(current_response) > 0 and len(current_context) > 0:
                        all_pairs.append({
                            "messages": [
                                {"role": "user", "content": " \n ".join(current_context)},
                                {"role": "assistant", "content": " \n ".join(current_response)}
                            ]
                        })
                    current_context = []
                    current_response = []
                
                current_context.append(content)
                
        # Koniec czatu - jeśli na samym końcu Janusz odpowiedział
        if len(current_response) > 0 and len(current_context) > 0:
            all_pairs.append({
                "messages": [
                    {"role": "user", "content": " \n ".join(current_context)},
                    {"role": "assistant", "content": " \n ".join(current_response)}
                ]
            })

    # Zapis
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out:
        for pair in all_pairs:
            out.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
    print(f"Pomyślnie wygenerowano {len(all_pairs)} par konwersacyjnych w formacie 'messages' do {output_file}!")

if __name__ == "__main__":
    main()

