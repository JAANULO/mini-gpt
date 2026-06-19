import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mini_gpt.tokenizer import Tokenizer

def test_tokenizer_load():
    tokenizer = Tokenizer()
    # Zakładamy że skrypt odpala z root projektu
    model_path = "exports/tokenizer.model"
    if os.path.exists(model_path):
        assert tokenizer.load(model_path) == True
        assert tokenizer.vocab_size > 0

def test_tokenizer_encode_decode():
    tokenizer = Tokenizer()
    model_path = "exports/tokenizer.model"
    if not os.path.exists(model_path):
        pytest.skip("Brak modelu tokenizera, pomijam test")
        
    tokenizer.load(model_path)
    
    text = "ala ma kota"
    ids = tokenizer.encode(text)
    
    assert isinstance(ids, list)
    assert len(ids) > 0
    
    decoded = tokenizer.decode(ids)
    assert isinstance(decoded, str)
    assert len(decoded) > 0

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
