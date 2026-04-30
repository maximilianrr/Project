import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# # Config
@dataclass
class GPTConfig:
    block_size: int = 1024 # Max context length
    vocab_size: int = 10000 # What is our vocab size?
    n_layer: int = 6       # How many transformer blocks
    n_head: int = 8        # How many transformer heads
    n_embd: int = 512      # Token embed dimension
    dropout: float = 0.2   # Dropout

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        # Mask out future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        v = self.value(x) 
        out = wei @ v 
        return out


class MultiHeadAttention(nn.Module):
    """ 
    Multiple heads of self-attention in parallel 
    
    He did something later in the implemntation of gpt 2.0 where he combined these classes
    Also used something called flash attention that is supposed to be faster than the attention implemention above
    
    """

    def __init__(self, config, head_size):
        super().__init__()
        # List of heads
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_heads)])
        
        # Projection layer to mix the outputs of the heads back together
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Get the results from each head and concat them
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        
        x = self.proj(x)
        out = self.dropout(x)
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  #Larger size inside the feedforward layer as the Attention is all you need paper
            
            # He uses GELU instead of relu in gpt 2.0
            nn.GELU(), 
            
            # nn.ReLU(),
            
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x):
        return self.net(x)

# Transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        head_size = config.n_embd // config.n_head


        self.layer_norm1 = nn.LayerNorm(config.n_embd) # Layer norm are done before attention blocks, different than the original paper
        self.self_att = MultiHeadAttention(config, head_size) 
        self.layer_norm2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.self_att(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

# NanoChat
class NanoChat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core embeddings
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        # The transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        #self.apply(self._init_weights)

    def _init_weights(self, module):
        # Need to decide how we should iniate our wieghts
        return

    def forward(self, input, targets=None):
        B, T = idx.shape
        
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"


        pos = torch.arrange(0,T,dtype = torch.long, device = input.device())
        # Get embeddings
        tok_emb = self.token_embedding_table(input) # (B, T, C)
        pos_emb = self.position_embedding_table(pos) # (T, C)
        
        x = tok_emb + pos_emb # Combine token and position data
        x = self.blocks(x) # pass through the transformer blocks
        x = self.ln_f(x) # Final LayerNorm
        logits = self.output_head(x) # Output probabilities (B, T, vocab_size)

        loss = None
        if targets is not None: # He adds this inside here
            # PyTorch's cross_entropy requires a 2D tensor for logits and 1D for targets
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    @torch.no_grad()
    def generate(self, input, max_new_tokens):
        for _ in range(max_new_tokens):

            input_max_context = input[:,-self.config.block_size] # Take everything in batch and only keep last block_size tokens
            logits, _ = self(input_max_context)

            # Logits.shape = [B,T, vocab_size] So we want all batches last token with all logits (one for every token in vocab)
            logits = logits[:, -1, :]

            #Convert to probabilties for the tokens, looking at last dimension 
            probs = F.softmax(logits, dim= -1) 

            next_token = torch.multinomial(probs, num_samples=1)

            output  = torch.cat((input, next_token), dim = 1)
            return output
