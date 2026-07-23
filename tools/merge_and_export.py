import os
import sys
import subprocess
import torch
import requests

# Dodajemy główny folder projektu do ścieżki
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mini_gpt.config import cfg

def download_convert_script(target_path):
    url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    print(f"Pobieranie oficjalnego skryptu konwersji llama.cpp z: {url}...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Skrypt konwersji zapisany w: {target_path}")
        return True
    except Exception as e:
        print(f"[BLAD] Nie udało się pobrać skryptu konwersji: {e}")
        return False

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = os.path.join(cfg.ft_output_dir, "lora_adapter")
    merged_model_dir = os.path.join(cfg.ft_output_dir, "merged_model")
    gguf_path = os.path.join(cfg.ft_output_dir, "model-f16.gguf")
    modelfile_path = os.path.join(cfg.ft_output_dir, "Modelfile")
    
    if not os.path.exists(adapter_path):
        print(f"[BLAD] Brak adaptera LoRA w: {adapter_path}. Najpierw uruchom: python tools/finetune.py")
        return

    # 1. Scalanie wag (Merge)
    print("Rozpoczynanie scalania wag adaptera z modelem bazowym...")
    print(f"Ładowanie modelu bazowego: {cfg.ft_base_model}...")
    
    # Ładujemy na CPU, aby nie przekroczyć pamięci GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.ft_base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.ft_base_model, trust_remote_code=True)
    
    print(f"Ładowanie adaptera LoRA z: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Scalanie wag (merge_and_unload)...")
    merged_model = model.merge_and_unload()
    
    print(f"Zapisywanie scalonego modelu do: {merged_model_dir}...")
    os.makedirs(merged_model_dir, exist_ok=True)
    merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_dir)
    print("✅ Scalanie zakończone sukcesem!")

    # Zwolnienie pamięci
    del base_model
    del model
    del merged_model
    import gc
    gc.collect()

    # 2. Automatyczna konwersja do GGUF
    print("\n--- Przygotowanie konwersji do formatu GGUF ---")
    convert_script_path = os.path.join(cfg.ft_output_dir, "convert_hf_to_gguf.py")
    
    # Pobieramy skrypt jeśli nie istnieje
    if not os.path.exists(convert_script_path):
        download_success = download_convert_script(convert_script_path)
    else:
        download_success = True
        
    if download_success:
        print("Instalowanie/sprawdzanie wymaganej biblioteki 'gguf'...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "gguf"], check=True)
        except Exception as e:
            print(f"[OSTRZEZENIE] Nie udało się zainstalować biblioteki 'gguf' automatycznie: {e}")
            
        print(f"Uruchamianie konwersji model-safetensors -> GGUF (f16)...")
        try:
            cmd = [
                sys.executable, 
                convert_script_path, 
                merged_model_dir, 
                "--outfile", 
                gguf_path, 
                "--outtype", 
                "f16"
            ]
            print(f"Uruchamianie komendy: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"✅ Konwersja zakończona pomyślnie! Plik GGUF zapisany w: {gguf_path}")
        except Exception as e:
            print(f"[BLAD] Konwersja do GGUF nie powiodła się: {e}")
            print("Możesz spróbować uruchomić skrypt convert_hf_to_gguf.py ręcznie.")
            return
    else:
        print("[!] Pominięto automatyczną konwersję GGUF z powodu braku skryptu konwertującego.")
        return

    # 3. Generowanie pliku Modelfile dla Ollamy
    print(f"\nGenerowanie pliku Modelfile w: {modelfile_path}...")
    
    # Szablon czatu zgodny z ChatML (używanym przez Qwen)
    chat_template = """TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
\"\"\""""

    system_prompt_escaped = cfg.ft_system_prompt.replace('"', '\\"')

    modelfile_content = f"""FROM {os.path.abspath(gguf_path)}

# Szablon czatu ChatML
{chat_template}

# Persona (System Prompt) Janusza
SYSTEM "{system_prompt_escaped}"

# Parametry generowania
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
        
    print("✅ Plik Modelfile został wygenerowany!")
    print("\n========================================================")
    print("SUKCES! Aby zaimportować model do Ollamy, wykonaj komendę:")
    print(f"  ollama create moj-klon -f {modelfile_path}")
    print("Następnie możesz rozmawiać z nim za pomocą:")
    print("  ollama run moj-klon")
    print("========================================================")

if __name__ == "__main__":
    main()
