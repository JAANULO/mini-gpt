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

# Global singleton
cfg = Config()
