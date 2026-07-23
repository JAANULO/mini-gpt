import yaml
import os
from pathlib import Path

class Config:
    def __init__(self, config_path="config.yaml"):
        # We need an absolute path to config.yaml depending on where the script is run from
        base_dir = Path(__file__).resolve().parent.parent
        full_path = base_dir / config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku konfiguracji: {full_path}")
            
        with open(full_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
            
    # Paths
    @property
    def data_file(self) -> str: return self._cfg["paths"]["data_file"]
    @property
    def tokenizer_model(self) -> str: return self._cfg["paths"]["tokenizer_model"]
    @property
    def model_export(self) -> str: return self._cfg["paths"]["model_export"]
    @property
    def model_cache(self) -> str: return self._cfg["paths"]["model_cache"]
    @property
    def messenger_raw_dir(self) -> str: return self._cfg["paths"]["messenger_raw_dir"]
    @property
    def conversations_jsonl(self) -> str: return self._cfg["paths"]["conversations_jsonl"]
    
    # Model
    @property
    def embed_dim(self) -> int: return self._cfg["model"]["embed_dim"]
    @property
    def num_layers(self) -> int: return self._cfg["model"]["num_layers"]
    @property
    def num_heads(self) -> int: return self._cfg["model"]["num_heads"]
    @property
    def dropout(self) -> float: return self._cfg["model"]["dropout"]
    @property
    def max_length(self) -> int: return self._cfg["model"]["max_length"]
    
    # Training
    @property
    def epochs(self) -> int: return self._cfg["training"]["epochs"]
    @property
    def lr(self) -> float: return self._cfg["training"]["lr"]
    @property
    def batch_size(self) -> int: return self._cfg["training"]["batch_size"]
    
    # Chat
    @property
    def memory_window(self) -> int: return self._cfg["chat"]["memory_window"]
    @property
    def default_temp(self) -> float: return self._cfg["chat"]["default_temp"]

    # Filtering
    @property
    def filter_user_name(self) -> str: return self._cfg["filtering"]["user_name"]
    @property
    def filter_min_length(self) -> int: return self._cfg["filtering"]["min_length"]
    @property
    def filter_max_age_years(self) -> int: return self._cfg["filtering"]["max_age_years"]
    @property
    def filter_max_gap_minutes(self) -> int: return self._cfg["filtering"]["max_gap_minutes"]

    # Fine-tuning
    @property
    def ft_base_model(self) -> str: return self._cfg["finetuning"]["base_model"]
    @property
    def ft_output_dir(self) -> str: return self._cfg["finetuning"]["output_dir"]
    @property
    def ft_epochs(self) -> int: return self._cfg["finetuning"]["epochs"]
    @property
    def ft_batch_size(self) -> int: return self._cfg["finetuning"]["batch_size"]
    @property
    def ft_max_length(self) -> int: return self._cfg["finetuning"]["max_length"]
    @property
    def ft_lr(self) -> float: return self._cfg["finetuning"]["lr"]
    @property
    def ft_lora_r(self) -> int: return self._cfg["finetuning"]["lora_r"]
    @property
    def ft_lora_alpha(self) -> int: return self._cfg["finetuning"]["lora_alpha"]
    @property
    def ft_lora_dropout(self) -> float: return self._cfg["finetuning"]["lora_dropout"]
    @property
    def ft_co_training_dataset(self) -> str: return self._cfg["finetuning"]["co_training_dataset"]
    @property
    def ft_co_training_samples(self) -> int: return self._cfg["finetuning"]["co_training_samples"]
    @property
    def ft_system_prompt(self) -> str: return self._cfg["finetuning"]["system_prompt"]


# Global singleton
cfg = Config()
