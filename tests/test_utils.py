import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from mini_gpt.utils import top_k_top_p_sampling

def test_top_k_top_p_sampling():
    # Przygotowanie mockowych logitów: klasa 0 ma najwyższe prawdopodobieństwo
    logits = np.array([10.0, 1.0, 0.1, -2.0])
    
    # Z niską temperaturą algorytm powinien działać jak argmax i zawsze wybierać 0
    idx = top_k_top_p_sampling(logits, temperature=0.01)
    assert idx == 0, "Niska temperatura nie wybrała najbardziej prawdopodobnego tokenu"
    
    # Testowanie zachowania z parametrami top_k i top_p
    idx2 = top_k_top_p_sampling(logits, top_k=2, top_p=0.9, temperature=1.0)
    assert idx2 in [0, 1], "Sampling zwrócił token poza zakresem top-2"

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
