import sentencepiece as spm
import logging
from typing import List

logger = logging.getLogger("mini_gpt")

class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.vocab_size = 0
        self.PAD = 0
        self.UNK = 1

    def load(self, model_path: str = "exports/tokenizer.model") -> bool:
        try:
            self.sp.load(model_path)
            self.vocab_size = self.sp.get_piece_size()
            logger.info(f"Wczytano tokenizer BPE, rozmiar słownika: {self.vocab_size}")
            return True
        except Exception as e:
            logger.error(f"Nie udało się wczytać tokenizera z {model_path}: {e}")
            return False

    @property
    def id_to_word(self):
        # Fake dictionary for UI visualization compatibility (math_step)
        if self.vocab_size == 0:
            return {}
        return {i: self.sp.id_to_piece(i).replace(" ", " ") for i in range(self.vocab_size)}

    def encode(self, text: str) -> List[int]:
        if self.vocab_size == 0:
            return []
        return self.sp.encode_as_ids(text)

    def decode(self, ids: List[int]) -> str:
        if self.vocab_size == 0:
            return ""
        return self.sp.decode_ids(ids)
