import os
import sys
from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from pydantic import Field

# Dodajemy główny folder projektu do ścieżki
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mini_gpt.config import cfg
from mini_gpt.transformer import MiniGPT
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.utils import DEVICE, hash_data, load_cache, load_export, generate_text

class MiniGPTLLM(LLM):
    """Niestandardowa klasa LangChain integrująca lokalny model Mini-GPT."""
    model: Any = Field(default=None)
    tokenizer: Any = Field(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_local_model()
        
    def _load_local_model(self):
        print("Ładowanie modelu Mini-GPT do przestrzeni LangChain...")
        current_hash = hash_data(cfg.data_file)
        
        tokenizer_temp = Tokenizer()
        if not tokenizer_temp.load(cfg.tokenizer_model):
            raise Exception("Brak tokenizera. Uruchom najpierw trening.")
            
        model_temp = MiniGPT(
            vocab_size=tokenizer_temp.vocab_size,
            embed_dim=cfg.embed_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            max_length=cfg.max_length,
        ).to(DEVICE)
        
        from pathlib import Path
        CACHE_FILE = Path("exports/model_cache.pkl")
        EXPORT_FILE = Path("exports/model_export.pt")
        
        tokenizer_from_cache, cache_ok = load_cache(model_temp, current_hash, CACHE_FILE)
        if cache_ok:
            self.tokenizer = tokenizer_from_cache
            self.model = model_temp
        else:
            tokenizer_export, export_ok = load_export(model_temp, EXPORT_FILE)
            if export_ok:
                self.tokenizer = tokenizer_export
                self.model = model_temp
            else:
                raise Exception("Brak wytrenowanego modelu. Uruchom train.py")
                
        self.model.set_training(False)

    @property
    def _llm_type(self) -> str:
        return "mini-gpt-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Metoda używana przez LangChain do generowania odpowiedzi."""
        # Ze względu na specyfikę naszego modelu ucinamy prompt do ostatniego słowa kluczowego
        words = prompt.strip().split()
        if not words:
            return ""
        
        seed_word = words[-1].replace(":", "").replace("?", "").lower()
        max_new_tokens = kwargs.get("max_tokens", 20)
        
        response = generate_text(self.model, self.tokenizer, seed_word, temperature=0.5, max_tokens=max_new_tokens)
        return response

def main():
    print("--- Demonstracja LangChain + Mini-GPT ---")
    
    # 1. Inicjalizacja
    llm = MiniGPTLLM()
    
    # 2. Szablon
    template = "Jesteś modelem językowym MiniGPT. Dokończ zdanie po słowie: {keyword}"
    prompt = PromptTemplate.from_template(template)
    
    # 3. Łańcuch
    chain = prompt | llm
    
    # 4. Wykonanie
    test_word = "polska"
    print(f"\nUruchamianie łańcucha LangChain dla słowa kluczowego: '{test_word}'")
    
    result = chain.invoke({"keyword": test_word})
    
    print("\n--- WYNIK ---")
    print(f"Formatowany Prompt: {prompt.format(keyword=test_word)}")
    print(f"Wygenerowana kontynuacja: {result}")

if __name__ == "__main__":
    main()
