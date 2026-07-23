import os
import sys
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Dodajemy główny folder projektu do ścieżki
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mini_gpt.config import cfg

def load_messenger_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Brak pliku danych: {file_path}. Upewnij się, że uruchomiłeś najpierw: python scripts/prepare_conversations.py"
        )
    
    data = []
    print(f"Wczytywanie wiadomości z Messengera z {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Załadowano {len(data)} par konwersacyjnych z Messengera.")
    return data

def load_programming_data(dataset_name, num_samples):
    print(f"Pobieranie angielskiego zbioru zadań programistycznych '{dataset_name}' (próbka: {num_samples})...")
    try:
        raw_ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
        shuffled_ds = raw_ds.shuffle(seed=42)
        subset_ds = shuffled_ds.select(range(min(num_samples, len(shuffled_ds))))
        
        formatted_data = []
        for example in subset_ds:
            instruction = example.get("instruction", example.get("query", ""))
            input_text = example.get("input", "")
            output = example.get("output", example.get("response", example.get("answer", "")))
            
            if not instruction or not output:
                continue
                
            user_content = instruction
            if input_text and input_text.strip():
                user_content += f"\nInput:\n{input_text}"
                
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]
            })
            
        print(f"Pomyślnie przygotowano {len(formatted_data)} zadań programistycznych.")
        return formatted_data
    except Exception as e:
        print(f"[OSTRZEZENIE] Nie udało się pobrać zbioru '{dataset_name}': {e}")
        print("Trening odbędzie się wyłącznie na danych z Messengera.")
        return []

def main():
    # 1. Wczytywanie i przygotowanie danych
    try:
        messenger_data = load_messenger_data(cfg.conversations_jsonl)
    except FileNotFoundError as e:
        print(e)
        return
        
    co_training_data = []
    if cfg.ft_co_training_samples > 0:
        co_training_data = load_programming_data(
            cfg.ft_co_training_dataset, 
            cfg.ft_co_training_samples
        )
        
    system_message = {"role": "system", "content": cfg.ft_system_prompt}
    
    combined_data = []
    for item in messenger_data:
        messages = item.get("messages", [])
        if messages:
            combined_data.append({
                "messages": [system_message] + messages
            })
            
    for item in co_training_data:
        messages = item.get("messages", [])
        if messages:
            combined_data.append({
                "messages": [system_message] + messages
            })
            
    train_dataset = Dataset.from_list(combined_data)
    train_dataset = train_dataset.shuffle(seed=42)
    print(f"Całkowita liczba przykładów treningowych (Messenger + Kod): {len(train_dataset)}")

    # 2. Konfiguracja 4-bit (QLoRA)
    print("Konfigurowanie kwantyzacji 4-bit (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # 3. Ładowanie tokenizera i modelu
    print(f"Ładowanie tokenizera dla modelu: {cfg.ft_base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.ft_base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Ładowanie modelu: {cfg.ft_base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.ft_base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 4. Przygotowanie modelu do QLoRA
    model = prepare_model_for_kbit_training(model)

    # 5. Konfiguracja LoRA
    print("Konfigurowanie adaptera LoRA...")
    peft_config = LoraConfig(
        r=cfg.ft_lora_r,
        lora_alpha=cfg.ft_lora_alpha,
        lora_dropout=cfg.ft_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ]
    )

    # 6. Parametry treningowe
    print("Konfigurowanie parametrów treningowych...")
    training_args = TrainingArguments(
        output_dir=cfg.ft_output_dir,
        per_device_train_batch_size=cfg.ft_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=cfg.ft_lr,
        logging_steps=10,
        num_train_epochs=cfg.ft_epochs,
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none"
    )

    # 7. Inicjalizacja SFTTrainer
    print("Inicjalizacja SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=cfg.ft_max_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 8. Start treningu
    print("Rozpoczynanie fine-tuningu...")
    trainer.train()

    # 9. Zapis adaptera
    adapter_path = os.path.join(cfg.ft_output_dir, "lora_adapter")
    print(f"Trening zakończony! Zapisywanie adaptera LoRA do: {adapter_path}")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print("✅ Gotowe! Uruchom teraz skrypt tools/merge_and_export.py, aby scalić model i wyeksportować do formatu GGUF.")

if __name__ == "__main__":
    main()
