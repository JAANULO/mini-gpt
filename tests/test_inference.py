import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mini_gpt.inference import build_context

def test_build_context():
    history = [("jak masz na imię?", "jestem mini-gpt")]
    new_q = "kto cię stworzył?"
    
    # Wywołanie funkcji
    context = build_context(history, new_q)
    
    # Oczekiwany format do podawania na wejście modelu
    expected = "user jak masz na imię? assistant jestem mini-gpt user kto cię stworzył? assistant"
    assert context == expected, "Kontekst rozmowy został źle zbudowany"

def test_build_context_empty_history():
    history = []
    new_q = "hej"
    
    context = build_context(history, new_q)
    expected = "user hej assistant"
    assert context == expected, "Kontekst dla pustej historii jest błędny"

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
