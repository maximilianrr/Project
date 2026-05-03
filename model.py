import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F



class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        
        #self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))  #Register buffer to make sure pytorcgh knows this is part of the model and should be moved to gpu. 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        
        dropout = self.dropout.p if self.training else 0.0

        key = self.key(x)   # (B, T, head_size)
        query = self.query(x) # (B, T, head_size)
        value = self.value(x) 
        
        out = F.scaled_dot_product_attention(query,key,value,dropout_p= dropout, is_causal=True)  #Pytorchs fast attention method, is_causal = True applies the triangle masking. 
 
        """
        B, T, C = x.shape
        
       
        # Compute attention scores
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T), 2nd part is the dividing by square root of head size. 
        
        # Mask out future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) and masking out triangle of values to hide them
        
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        

        out = wei @ v 
        """
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
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        
        # Projection layer to mix the outputs of the heads back together
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Get the results from each head and concat them
        x = torch.cat([h(x) for h in self.heads], dim=-1) # Puts output off all heads into onem massive vector head*head_size
        
        x = self.proj(x) # Is called the projection layer as it the model does a large matrix multiplication with the vector consisting of all heads outputs. giving it opportunity to mix and learn. 
        out = self.dropout(x)
        return out
    
class AllHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd
        self.n_heads = config.n_head
        self.head_size = self.n_embd // self.n_heads
        self.dropout_value = config.dropout
        
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Projection layer to mix the outputs of the heads back together
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x):
        B,T,C = x.shape

        q = self.query(x)  # [B,T, n_embd]
        k = self.key(x) # [B,T, n_embd]
        v = self.value(x) # [B,T, n_embd]

        """
        Remember that head_size = n_embd / n_head
        And we used to do self.key = nn.Linear(config.n_embd, head_size, bias=False), which would createa a [B,T,head_size] matrix for each head. 
        Now we have all heads query,key and values in one big tensor, but in the wrong dimension. Pytorch things its just one VERY big head, so need to make the tensors 4d by splitting up into head
        """

        q = q.reshape(B,T, self.n_heads, self.head_size) 
        k = k.reshape(B,T, self.n_heads, self.head_size)
        v = v.reshape(B,T, self.n_heads, self.head_size)
        """
        NOTE: attn_weight = query @ key.transpose(-2, -1) * scale_factor  -------- From the docs of scaled_dot_product_attention. 
        So they are multiplying the last and 2nd last dimension only. 
        Right now that would be n_heads and head_size. Which is not what we want. We want the time dimension (all the tokens) to be multiplied
        So need to transpose swapping n_heads and T
        """
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        attention = F.scaled_dot_product_attention(q,k,v,dropout_p= self.dropout_value, is_causal=True) # [B,n_heads,T,head_size]

        attention = attention.transpose(1,2).reshape(B,T, self.n_embd)

        out = self.proj(attention)

        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  #Larger size inside the feedforward layer as the Attention is all you need paper, more space to learn
            
            # He uses GELU instead of relu in gpt 2.0
            nn.GELU(), 
            
            # nn.ReLU(),
            
            nn.Linear(4 * config.n_embd, config.n_embd), # Compress back to original
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
        self.self_att = AllHeadAttention(config) 
        self.layer_norm2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.self_att(self.layer_norm1(x)) # Layer norm are done before attention blocks, different than the original paper, supposed to help with vanishing gradients.
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

# NanoChat
class NanoChat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core embeddings
        self.token_embedding_table = nn.Embedding(config.VOCAB_SIZE, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        # The transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.output_head = nn.Linear(config.n_embd, config.VOCAB_SIZE, bias=False)


        self.output_head.weight = self.token_embedding_table.weight # Recommended to do, makes the token probability be similarity between hidden state and token embedding. 
        
        self.apply(self._init_weights)

    def _init_weights(self, module): # From karpathys lets recreate GPT 2.0 video.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        return

    def forward(self, input, targets=None):
        B, T = input.shape
        
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"


        pos = torch.arange(0,T,dtype = torch.long, device = input.device)
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
    def generate(self, input, max_new_tokens = 1, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            
            input_max_context = input[:,-self.config.block_size:] # Take everything in batch and only keep last block_size tokens
            logits, _ = self(input_max_context)

            # Logits.shape = [B,T, vocab_size] So we want all batches last token with all logits (one for every token in vocab)
            logits = logits[:, -1, :]
            if temperature is not None:
                logits = logits / temperature


            if top_k is not None:
                top_k_logits, indices = torch.topk(logits, top_k, dim = -1)
                cutoff = top_k_logits[:, [-1]]
                mask = logits < cutoff

                logits[mask] = -float('Inf')

            #Convert to probabilties for the tokens, looking at last dimension 
            probs = F.softmax(logits, dim= -1) 
  


            next_token = torch.multinomial(probs, num_samples=1)

            input  = torch.cat((input, next_token), dim = 1)
        return input
