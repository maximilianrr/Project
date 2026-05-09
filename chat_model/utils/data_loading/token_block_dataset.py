
import torch
from torch.utils.data import Dataset

class TokenBlockDataset(Dataset):
    def __init__(self, sequences, block_size):
        self.examples = []
        for sequence in sequences:
            if len(sequence) <= block_size:
                continue
            for start in range(0, len(sequence) - block_size):
                chunk = sequence[start:start + block_size + 1]
                if len(chunk) == block_size + 1:
                    inputs = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    self.examples.append((inputs, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]