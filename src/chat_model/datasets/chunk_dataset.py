import torch
from torch.utils.data import Dataset

class ChunkChatDataset(Dataset):
    """
    Flattens all conversations into one large token stream with boundary
    tokens, then gives block_size sized chunks for training.
    """

    def __init__(self, conversations, tokenizer, block_size, max_conversations=None, max_blocks=None):
        self.block_size = block_size

        # Limit number of raw conversations
        if max_conversations is not None:
            conversations = conversations[:max_conversations]

        all_ids = []

        for conversation in conversations:
            for message in conversation:
                # Add role boundary token
                role_tokens = tokenizer.encode(f"<|{message['role']}|>")
                all_ids.extend(role_tokens)

                # Add message content
                all_ids.extend(tokenizer.encode(message["content"]))

                # Add end of message token
                end_tokens = tokenizer.encode("<|end|>")
                all_ids.extend(end_tokens)

            # Add end of conversation token
            all_ids.extend(tokenizer.encode("<|endoftext|>"))

        self.all_ids = torch.tensor(all_ids, dtype=torch.long)

        # Limit max number of blocks
        if max_blocks is not None:
            # We need (max_blocks * block_size) tokens, plus 1 extra token for the shifted 'y' target
            max_tokens_needed = (max_blocks * self.block_size) + 1
            self.all_ids = self.all_ids[:max_tokens_needed]

    def __len__(self):
        # Subtract 1 because y is shifted 1 token into the future
        return (len(self.all_ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end   = start + self.block_size

        x = self.all_ids[start : end]
        y = self.all_ids[start + 1 : end + 1].clone()

        return x, y
