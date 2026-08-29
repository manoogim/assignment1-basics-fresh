
def closest_multiple( num, multiplier = 64):
    x = num * 8 / 3
    ans = multiplier * round (x / multiplier)
    return ans

def sample_params_inputs_gpt2xl():
    return {
        'label': 'GPT2 XL',
        'vocab_size': 50_257,
        'seq_len': 1_024,
        'num_layers': 48,
        'd_model': 1_600,
        'num_heads': 25,
        'd_ff': 4288,
    }

def gpt2_small_inputs():
    return {
        'label': 'GPT2 SMALL',
        'vocab_size': 50_257,
        'seq_len': 1_024,
        'num_layers': 12,
        'd_model': 768,
        'num_heads': 12,
        'd_ff': closest_multiple(768),
    }

def gpt2_medium_inputs():
    return {
        'label': 'GPT2 MEDIUM',
        'vocab_size': 50_257,
        'seq_len': 1_024,
        'num_layers': 24,
        'd_model': 1_024,
        'num_heads': 16,
        'd_ff': closest_multiple(1_024),
    }

def gpt2_large_inputs():
    return {
        'label': 'GPT2 LARGE',
        'vocab_size': 50_257,
        'seq_len': 1_024,
        'num_layers': 36,
        'd_model': 1_280,
        'num_heads': 20,
        'd_ff': closest_multiple(1_280),
    }


def calc_model_params(d):
    norm = d['d_model']
    emb = d['vocab_size'] * d['d_model']
    attn = 4 * d['d_model'] * d['d_model']        # q, k, v, o proj
    swiglu = 3 * d['d_ff'] * d['d_model']         # W1 (gate), W3 (up), W2 (down)
    block = attn + swiglu + 2 * norm              # 2 RMSNorms per block

    llm = norm + 2 * emb + d['num_layers'] * block  # final norm + tok emb + lm head
    return llm

def calc_activation_params(d, batch_size):
    dk =  d['d_model'] // d['num_heads']
    # commenting out rope cache b/c it is too small  to count and we can add it outside of activation
    # rope_cache = d['seq_len'] * dk 
    rope_pass = batch_size * d['seq_len'] * dk + batch_size * d['seq_len'] # rope input and positions

    swiglu = (batch_size * d['seq_len'] * d['d_ff']    # W1 (gate) output
          + batch_size * d['seq_len'] * d['d_ff']      # SiLU(W1 output)
          + batch_size * d['seq_len'] * d['d_ff']      # W3 (up) output
          + batch_size * d['seq_len'] * d['d_ff']      # elementwise gate*up product
          + batch_size * d['seq_len'] * d['d_model'])  # W2 (down) output
    
    rms_norm = batch_size * d['seq_len'] * d['d_model']

    mha = (3 * batch_size * d['seq_len'] * d['d_model']                    # Q, K, V
       + 2 * batch_size * d['num_heads'] * d['seq_len'] * d['seq_len']       # scores + softmax
       + 2 * batch_size * d['seq_len'] * d['d_model'])                  # attn output + out_proj
    
    logits = batch_size * d['seq_len'] * d['vocab_size']

    entropy = batch_size * d['seq_len'] * d['vocab_size']

    total = rope_pass + d['num_layers'] * (swiglu + 2 * rms_norm + mha) + rms_norm + logits + entropy
    return total 

def calc_memory_bytes(d, num_bytes=4):
    num_params = calc_model_params(d)
    mem = num_bytes * num_params
    return mem

def calc_training_memory(d, num_bytes=4):
    num_params = 4 * calc_model_params(d) # weights + grad + m + v
    mem = num_bytes * num_params 
    return mem  

def calc_peak_memory(d, b, num_bytes=4):
    num_params = calc_model_params(d)
    training_mem = num_bytes * num_params * 4 # multiply by 4 b/c for AdamW we need weights + grad + m + v
    activation_mem_per_batch = num_bytes * calc_activation_params(d, 1)
    activation_mem = b * activation_mem_per_batch
    result = training_mem + activation_mem
    return ( num_params, training_mem, activation_mem_per_batch, result)

def calc_flops(d):
    attn_flops = 4 * d['d_model']**3
    ffn_flops = 2 * d['d_model'] * d['d_ff'] ** (2)
    total_flops = d['num_layers'] * (attn_flops + ffn_flops)
    return total_flops

def report_param_accounting(label: str, d, num_bytes=4):
    num_params = calc_model_params(d)
    mem = calc_memory_bytes(d, num_bytes)
    flops = calc_flops(d)
    print(
    f"{label:>14} | "
    f"Num params: {num_params:>14,} | "
    f"Memory (MB): {mem // 1_000_000:>8,} | "
    f"Mega Flops: {flops // 1_000_000 :>14,}"
)

def report_peek_memory(label, d, num_bytes=4):
    batch_size = 4
    avail_mem = 80_000_000_000
    num_params, b, a, total = calc_peak_memory(d, batch_size, num_bytes)
    sol = ( avail_mem - b) // a
    print(
    f"{label:>14} | "
    f"Num params: {num_params:>14,} | "
    f"Training Memory (MB): {b // 1_000_000:>8,} | "
    f"Activation Memory per batch (MB): {a // 1_000_000:>8,} | "
    f"Total memory for batch={batch_size} (MB): {total:_}"
    )
    print (
    f"Equation: {a:_} * X + {b:_} = {avail_mem:_} | " 
    f"Solution: batch is {sol}"
    )

def calc_adamw_flops(dm):
    """
    weights, grad, m, v all have N = dm*dm elements.
    Every AdamW op is elementwise -- no matmuls involved.
    """
    N = dm * dm
    weight_decay = N          # p *= (1 - alpha*lambda)
    m_update = 3 * N          # m = beta1*m + (1-beta1)*grad
    v_update = 4 * N          # v = beta2*v + (1-beta2)*grad*grad
    m_hat = N                 # m / (1 - beta1**t)   (scalar denom, negligible)
    v_hat = N                 # v / (1 - beta2**t)
    p_update = 5 * N          # p -= alpha * m_hat / (sqrt(v_hat) + eps)
    result = weight_decay + m_update + v_update + m_hat + v_hat + p_update  # = 15*N
    return result

if __name__ == '__main__':
    d = sample_params_inputs_gpt2xl()
    report_peek_memory(d['label'], d)
    # report_param_accounting(d['label'], d, 2) 
# if __name__ == '__main__':
#     arr = [sample_params_inputs_gpt2xl(), gpt2_small_inputs(), gpt2_medium_inputs(), gpt2_large_inputs()]
#     for dict in arr:
#         report_param_accounting(dict['label'], dict, 2) 


    