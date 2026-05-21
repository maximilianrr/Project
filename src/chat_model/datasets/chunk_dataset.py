import torch
from torch.utils.data import Dataset
from chat_model import config
class ChunkChatDataset(Dataset):
    """
    Flattens all conversations into one large token stream using the tokenizer's
    native render_conversation method.
    
    If loss_masking=True, User prompts, system prompts are 
    masked with -100 so loss is only calculated on the Assistants generated text.
    If loss_masking=False, the model trains to predict every token in the sequence.
    """

    def __init__(self, conversations, tokenizer, block_size, max_conversations=None, max_blocks=None, loss_masking=True):
        self.block_size = block_size
        self.ignore_index = -100
        self.loss_masking = loss_masking

        # Limit number of raw conversations
        if max_conversations is not None:
            conversations = conversations[:max_conversations]

        all_ids = []
        all_targets = []

        for conversation in conversations:
            # Standardize formatting to match what render_conversation expects.
            if isinstance(conversation, list):
                conv_dict = {"messages": conversation}
            else:
                conv_dict = conversation

            # Get tokens ids and training mask
            ids, mask = tokenizer.render_conversation(conv_dict, max_tokens = config.BLOCK_SIZE)

            # If masked take all token_ids flagged with 0 from the render_conversation and set to -100 (pytorch deafult for ignore in loss)
            if self.loss_masking:
                targets = [
                    tok_id if m == 1 else self.ignore_index 
                    for tok_id, m in zip(ids, mask)
                ]
            else:
                # No mask, the target is the token id
                targets = ids

            all_ids.extend(ids)
            all_targets.extend(targets)

        self.all_ids = torch.tensor(all_ids, dtype=torch.long)
        self.all_targets = torch.tensor(all_targets, dtype=torch.long)

        # Limit max number of blocks
        if max_blocks is not None:
            # We need (max_blocks * block_size) tokens, plus 1 extra token for the shifted 'y' target
            max_tokens_needed = (max_blocks * self.block_size) + 1
            self.all_ids = self.all_ids[:max_tokens_needed]
            self.all_targets = self.all_targets[:max_tokens_needed]

    def __len__(self):
        # Subtract 1 because y is shifted 1 token into the future
        return (len(self.all_ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end   = start + self.block_size

        # x is our input tokens
        x = self.all_ids[start : end]
        
        # y is our targets array, shifted by 1 into the future
        y = self.all_targets[start + 1 : end + 1].clone()

        return x, y
    

class ChunkTextDataset(Dataset):
    """
    Flattens a list of raw text documents into one large token stream,
    separated by document boundary tokens, and yields block_size chunks.
    Used for standard raw text pre-training.
    """

    def __init__(self, documents, tokenizer, block_size, max_documents=None, max_blocks=None):
        self.block_size = block_size

        if max_documents is not None:
            documents = documents[:max_documents]

        all_ids = []
        bos_id = tokenizer.get_bos_token_id()

        for doc in documents:
            # Add BOS token to denote the start of a new document
            all_ids.append(bos_id)
            
            # Encode the raw text and append
            all_ids.extend(tokenizer.encode(doc))

        self.all_ids = torch.tensor(all_ids, dtype=torch.long)

        # Limit max number of blocks
        if max_blocks is not None:
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