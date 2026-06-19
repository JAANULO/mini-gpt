import os
import json
import glob

def fix_mojibake(text):
    if not text:
        return ""
    try:
        return text.encode('iso-8859-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def is_valid_text(text):
    if not text:
        return False
    text_stripped = text.strip()
    if not text_stripped:
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
    USER_NAME = "Janusz Andrzejewski"
    input_dir = r"C:\Users\atona\Desktop\Nowe_dane\4\your_facebook_activity\messages\inbox"
    output_file = os.path.join("data", "conversations.jsonl")
    
    all_pairs = []
    
    # Szukaj plików message_*.json
    file_paths = glob.glob(os.path.join(input_dir, "*", "message_*.json"))
    print(f"Znaleziono {len(file_paths)} plików z wiadomościami.")
    
    # Przetwarzaj katalog po katalogu, by zachować kontekst pojedynczego czatu
    directories = {}
    for fp in file_paths:
        dirname = os.path.dirname(fp)
        if dirname not in directories:
            directories[dirname] = []
        directories[dirname].append(fp)
        
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
                
        # Wiadomości w plikach Facebooka są od najnowszej do najstarszej (descending)
        # Sortujemy je po timestamp_ms (od najstarszej)
        all_messages_in_chat.sort(key=lambda x: x.get("timestamp_ms", 0))
        
        current_context = []
        current_response = []
        
        for msg in all_messages_in_chat:
            sender = msg.get("sender_name")
            content = msg.get("content")
            
            if not is_valid_text(content):
                continue
                
            if sender == USER_NAME:
                # To wiadomość od Janusza (odpowiedź)
                current_response.append(content)
            else:
                # To wiadomość od kogoś innego
                # Zanim dodamy do kontekstu, sprawdzamy czy Janusz właśnie skończył odpowiadać
                if len(current_response) > 0:
                    # Mamy kompletną parę! Zapisujemy
                    if len(current_context) > 0:
                        context_str = " \n ".join(current_context)
                        response_str = " \n ".join(current_response)
                        
                        all_pairs.append({
                            "source": "messenger",
                            "context": context_str,
                            "response": response_str
                        })
                    
                    # Reset po zapisaniu pary - zaczynamy nowy kontekst
                    current_context = []
                    current_response = []
                
                # Dodajemy wiadomość do kontekstu (jeśli to grupa, będzie tu prefix imienia)
                # Opcjonalnie można dodać "Imię: treść", ale zrobimy samo "treść"
                current_context.append(content)
                
        # Koniec czatu - jeśli na samym końcu Janusz odpowiedział
        if len(current_response) > 0 and len(current_context) > 0:
            context_str = " \n ".join(current_context)
            response_str = " \n ".join(current_response)
            all_pairs.append({
                "source": "messenger",
                "context": context_str,
                "response": response_str
            })

    # Zapis
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out:
        for pair in all_pairs:
            out.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
    print(f"Pomyślnie wygenerowano {len(all_pairs)} par konwersacyjnych do {output_file}!")

if __name__ == "__main__":
    main()
