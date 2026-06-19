import tiktoken
from typing import List

class BPETokenizer:
    def __init__(self, model_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(model_name)
        self.vocab_size = self.enc.n_vocab  # 词表大小 (50257)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)