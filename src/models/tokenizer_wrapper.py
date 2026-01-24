"""Tokenizer entry point"""
from transformers import AutoTokenizer

class TokenizerWrapper:
    """Tokenizer entry point"""
    def __init__(self, path, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.max_len = max_len

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        return inputs