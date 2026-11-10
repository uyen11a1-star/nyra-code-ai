from tokenizers import Tokenizer

class NyraTokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.post_processor = None
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")

    def encode(self, text, return_tensors=False):
        enc = self.tokenizer.encode(text)
        return enc.ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
