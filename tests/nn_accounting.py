
from tests.nn_yaml import ModelConfig, load_yaml_config


def calc_model_params(d: ModelConfig):
    norm = d.d_model
    emb = d.vocab_size * d.d_model
    attn = 4 * d.d_model * d.d_model        # q, k, v, o proj
    swiglu = 3 * d.d_ff * d.d_model         # W1 (gate), W3 (up), W2 (down)
    block = attn + swiglu + 2 * norm              # 2 RMSNorms per block

    llm = norm + 2 * emb + d.num_layers * block  # final norm + tok emb + lm head
    return llm

def calc_activation_params(d: ModelConfig, batch_size: int):
    dk =  d.d_model // d.num_heads
    # commenting out rope cache b/c it is too small  to count and we can add it outside of activation
    # rope_cache = d['seq_len'] * dk 
    rope_pass = batch_size * d.seq_len * dk + batch_size * d.seq_len # rope input and positions

    swiglu = (batch_size * d.seq_len * d.d_ff    # W1 (gate) output
          + batch_size * d.seq_len * d.d_ff      # SiLU(W1 output)
          + batch_size * d.seq_len * d.d_ff      # W3 (up) output
          + batch_size * d.seq_len * d.d_ff      # elementwise gate*up product
          + batch_size * d.seq_len * d.d_model)  # W2 (down) output
    
    rms_norm = batch_size * d.seq_len * d.d_model

    mha = (3 * batch_size * d.seq_len * d.d_model                    # Q, K, V
       + 2 * batch_size * d.num_heads * d.seq_len * d.seq_len       # scores + softmax
       + 2 * batch_size * d.seq_len * d.d_model)                  # attn output + out_proj
    
    logits = batch_size * d.seq_len * d.vocab_size

    entropy = batch_size * d.seq_len * d.vocab_size

    total = rope_pass + d.num_layers * (swiglu + 2 * rms_norm + mha) + rms_norm + logits + entropy
    return total 

def calc_memory_bytes(d: ModelConfig, num_bytes: int = 4):
    num_params = calc_model_params(d)
    mem = num_bytes * num_params
    return mem

def calc_training_memory(d: ModelConfig, num_bytes: int = 4):
    num_params = 4 * calc_model_params(d) # weights + grad + m + v
    mem = num_bytes * num_params 
    return mem  

def calc_peak_memory(d: ModelConfig, b: int, num_bytes: int = 4):
    num_params = calc_model_params(d)
    training_mem = num_bytes * num_params * 4 # multiply by 4 b/c for AdamW we need weights + grad + m + v
    activation_mem_per_batch = num_bytes * calc_activation_params(d, 1)
    activation_mem = b * activation_mem_per_batch
    result = training_mem + activation_mem
    return ( num_params, training_mem, activation_mem_per_batch, result)

def report_peek_memory(label: str, d: ModelConfig, num_bytes: int = 4, avail_mem: int = 80_000_000_000, batch_size: int = 128):
    tokens_budget = batch_size * d.seq_len * d.vocab_size
    num_params, training_mem, activation_mem_per_batch, total = calc_peak_memory(d, batch_size, num_bytes)
    sol = ( avail_mem - training_mem) // activation_mem_per_batch
    print(
    f"{label:>14} *|* "
    f"Num params: {num_params:>14,} | "
    f"Training Memory (MB): {training_mem // 1_000_000:>8,} | "
    f"Activation Memory per batch (MB): {activation_mem_per_batch // 1_000_000:>8,} | "
    f"Total memory for batch={batch_size} (MB): {total:_}"
    )
    print (
    f"Equation: {activation_mem_per_batch:_} * X + {training_mem:_} = {avail_mem:_} | " 
    f"Solution: batch is {sol}"
    )

if __name__ == "__main__":
    _, cfg = load_yaml_config('tests/config/cs336_basic.yaml')
    report_peek_memory("cs336_basic", cfg.model, num_bytes=4, avail_mem=5_000_000_000, batch_size=cfg.train.batch_size)
