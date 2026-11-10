import torch
from torch.utils.data import Dataset

class NyraDataset(Dataset):
    def __init__(self, token_ids, seq_len=128):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        chunk = self.token_ids[start_idx:end_idx]
        
        # Input: 0 -> n-1, Target: 1 -> n
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
