import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import time



class KVCache:
    def __init__(self):
        self.k = None
        self.v = None
        self.pos = 0

    def update(self, k, v):
        """Appends new keys and values to the cache and returns the full sequence."""
        if self.k is not None:
            self.k = torch.cat((self.k, k), dim=-2)
            self.v = torch.cat((self.v, v), dim=-2)
        else:
            self.k = k
            self.v = v
        self.pos += k.shape[-2] # Add T to pos to keep track of 
        return self.k, self.v
    
    def get_pos(self):
        return self.pos

    @property
    def length(self):
        """Returns the current sequence length of the cached tokens."""
        if self.k is not None:
            return self.k.shape[-2]
        return 0
    

def precompute_rope_embeddings(head_dim: int, block_size: int, base = 10000):  #10000 seems like a standard value, given our block size of 1024 should be more than enough, only when blocksize gets closer will we need larger value. 
    # theta_i = 1.0 / (base ^ (2i / d))
    pairs = torch.arange(0, head_dim, 2).float()  # [head_dim // 2]
    freq_cache = 1.0 / (base ** (pairs / head_dim))  # Raw frequency values for each pair.  [head_dim // 2]
    m_pos = torch.arange(block_size, device=freq_cache.device)  # How many potential positions there is. [block_size]

    angles = torch.outer(m_pos, freq_cache).float() # Multiply them to get a: [block_size, head_dim // 2]   angles. For each token when going through a head, needs to have head_dim//2 PAIRS of rotations. 

    cos = angles.cos() #[block_size, head_dim // 2] for each token position there is head_dim // 2 cos values. One value per pair inside the attention head.
    sin = angles.sin() #[block_size, head_dim // 2] for each token position there is head_dim // 2 sin values
    return cos, sin

def apply_rope_embeddings(x, cos, sin):
    """
    We want
    In 2d: [x]  * [cos   -sin]  = [x*cos - y*sin] = [x']
           [y]    [sin    cos]    [x*sin + y*cos]   [y']
    Could in theory create the entire n_embd x n_embd matrix but waste of time


    x will have shape [B,T, n_head, head_size]
    cos and sin will have shape [block_size, head_size // 2]

    we can remove all the cos/sin values that are not needed. So just keep everything until T
    then want to split x head_size dimension into 2 so we have 
    x_1 =[B, T, n_head, head_size //2]
    x_2 = [B,T, n_head, head_size //2]

    and we want to multiply the values in head_size // 2 with the values inside sin/cos head_size //2 dimension. 
    So need to reshape cos and sin into [1, T, 1, head_size // 2]

    Then multiply using the formula
    Then concat everything together

    """
    T = x.shape[1]

    #Reshape to match the dimension of q/k
    current_cos = cos.reshape(1, T, 1, -1)
    current_sin = sin.reshape(1,T,1,-1)
    
    split = x.shape[-1] // 2 #Find the middle value of the last dimension aka head_size

    x1 = x[:, :, :, :split]
    x2 = x[:, :, :, split:]

    #NOW can multiply using the formula above, this implicitly makes pairs between x1 and x2. so x1_1 and x2_1 is one pair etc etc

    y1 = (x1 * current_cos) - (x2 * current_sin)
    y2 = (x1 * current_sin) + (x2 * current_cos)

    new_x = torch.cat([y1,y2], dim = -1)
    return new_x

    
class AllHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_embd = config.N_EMB
        self.n_heads = config.N_HEAD
        self.head_size = self.n_embd // self.n_heads
        self.dropout_value = config.DROPOUT
        
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)


        cos, sin = precompute_rope_embeddings(head_dim=self.head_size, block_size= config.BLOCK_SIZE)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)


        # Projection layer to mix the outputs of the heads back together
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.proj.is_residual = True
        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x, cached_kv = None):
        B,T,C = x.shape

        past_length = cached_kv.get_pos() if cached_kv is not None else 0

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


        cos_slice = self.cos[past_length : past_length + T] 
        sin_slice = self.sin[past_length : past_length + T]

        #Now apply the rope mebeddings
        q = apply_rope_embeddings(q, cos_slice, sin_slice)
        k = apply_rope_embeddings(k, cos_slice, sin_slice)


        """
        NOTE: attn_weight = query @ key.transpose(-2, -1) * scale_factor  -------- From the docs of scaled_dot_product_attention. 
        So they are multiplying the last and 2nd last dimension only. 
        Right now that would be n_heads and head_size. Which is not what we want. We want the time dimension (all the tokens) to be multiplied
        So need to transpose swapping n_heads and T
        """
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        if cached_kv is not None:
            k, v = cached_kv.update(k, v)

        dropout_p = self.dropout_value if self.training else 0.0

        is_causal = q.shape[2] > 1
        attention = F.scaled_dot_product_attention(q,k,v,dropout_p= dropout_p, is_causal=is_causal) # [B,n_heads,T,head_size]

        attention = attention.transpose(1,2).reshape(B,T, self.n_embd)

        out = self.proj(attention)

        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.N_EMB, 4 * config.N_EMB),  #Larger size inside the feedforward layer as the Attention is all you need paper, more space to learn
            
            # He uses GELU instead of relu in gpt 2.0
            nn.GELU(), 
            
            # nn.ReLU(),
            
            nn.Linear(4 * config.N_EMB, config.N_EMB), # Compress back to original
            nn.Dropout(config.DROPOUT)
        )
        self.net[2].is_residual = True
        
    def forward(self, x):
        return self.net(x)

# Transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        head_size = config.N_EMB // config.N_HEAD
        self.layer_norm1 = nn.LayerNorm(config.N_EMB) # Layer norm are done before attention blocks, different than the original paper
        self.self_att = AllHeadAttention(config) 
        self.layer_norm2 = nn.LayerNorm(config.N_EMB)
        self.feed_forward = FeedForward(config)

    def forward(self, x, cached_kv = None):
        attn = self.self_att(self.layer_norm1(x), cached_kv=cached_kv) # Layer norm are done before attention blocks, different than the original paper, supposed to help with vanishing gradients.
        x = x + attn 
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

# NanoChat
class NanoChat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core embeddings
        self.token_embedding_table = nn.Embedding(config.VOCAB_SIZE, config.N_EMB)
        #self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd) # No longer needed since we dont use absolute positional embed

        # The transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.N_LAYER)])
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(config.N_EMB)
        self.output_head = nn.Linear(config.N_EMB, config.VOCAB_SIZE, bias=False)

        self.output_head.weight = self.token_embedding_table.weight # Recommended by karpathy to do, makes the token probability be similarity between hidden state and token embedding and also lowers the amount of weights to calculate. 
        
        self._init_weights()

    def _init_weights(self): # Mean of 0 and std = 0.02 is what gpt 2.0 did. 
       for module in self.modules():
            if isinstance(module, nn.Linear):
               std = 0.02
               if hasattr(module, 'is_residual'):
                   std = std * (1.0 / math.sqrt(2 * self.config.N_LAYER))
               torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
               if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input, targets=None, kv_caches = None):
        B, T = input.shape
        
        assert T <= self.config.BLOCK_SIZE, f"Cannot forward sequence of length {T}, block size is {self.config.BLOCK_SIZE}"


        # Get embeddings
        tok_emb = self.token_embedding_table(input) # (B, T, C)
        x = tok_emb

        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, cached_kv = layer_cache)


        x = self.ln_f(x) # Final LayerNorm
        logits = self.output_head(x) # Output probabilities (B, T, vocab_size)

        loss = None
        if targets is not None: 
            # PyTorch's cross_entropy requires a 2D tensor for logits and 1D for targets
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    @torch.no_grad()
    def generate(self, input, max_new_tokens = 200, temperature = 1.0, top_k = None, stop_token_id=None):
        max_prompt_length = self.config.BLOCK_SIZE - max_new_tokens

        if input.shape[1] > self.config.BLOCK_SIZE:
            print(f"Prompt too long! Truncating to {max_prompt_length} tokens.")
            input = input[:, -max_prompt_length:]

        kv_caches = [KVCache() for _ in range(self.config.N_LAYER)]
        next_input = input
        start_time = time.time()
        tokens_generated = 0
        for _ in range(max_new_tokens):
            
            if input.shape[1] >= self.config.BLOCK_SIZE:
                print(f"Context window full ({self.config.BLOCK_SIZE}). Stop generating")
                break


            logits, _ = self(next_input, kv_caches=kv_caches)

            # Logits.shape = [B,T, vocab_size] So we want all batches last token with all logits (one for every token in vocab)
            logits = logits[:, -1, :]
            if temperature is not None and temperature > 0:
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
            tokens_generated += 1
            #Check if a end token was generated and stop generating text
            if stop_token_id is not None and next_token[0].item() == stop_token_id:
                break
            next_input = next_token
        

        end_time = time.time()
        total_time = end_time - start_time
        tokens_per_second = tokens_generated / total_time


        print(f"Generated {tokens_generated} tokens in {total_time:.3f} seconds.")
        print(f"Speed: {tokens_per_second:.2f} tokens/sec\n")
        return input


