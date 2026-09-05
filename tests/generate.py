import torch

from tests.bpe_tokenizer import BpeTokenizer
from tests.nn_loader import load_checkpoint
from tests.nn_loader import load_checkpoint
from tests.nn_transformer import MyTransformer
from tests.nn_yaml import Config, load_yaml_config

def sample(model, input_ids, max_new_tokens, temp, top_k, max_seq_len):
    model.eval()
    for _ in range(max_new_tokens): 
        if input_ids.numel() > max_seq_len:
            print(f"*** Hard stop *** Input_ids length ({input_ids.numel()}) exceeds max_seq_len ({max_seq_len})")    
            return input_ids  
        all_logits = model(input_ids)
        # last token logits
        logits = all_logits[ -1, :]  

        logits = top_k_logits(logits, top_k)    
        logits = logits / temp

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token])
        # print(f'new token: {next_token.item()}, size of input_ids: {input_ids.numel()}')

    return input_ids

def top_k_logits(logits, k):
    if k == 0:
        return logits
    topk_values, topk_indices = torch.topk(logits, k)
    threshold = topk_values[-1]
    epses = torch.ones_like(logits, dtype=logits.dtype) * -1e10
    result = torch.where(logits < threshold, epses, logits)
    return result

def build_model(config: Config):
    model = MyTransformer.from_config(config.model, 'cpu')
    load_checkpoint(model, None, config.gen.model_weights_path, 'cpu')
    model.eval()
    return model

def load_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8', errors='replace') as f:
        prompt_text = f.read()
    return prompt_text

def generate(cfg_path):
    _, config = load_yaml_config(cfg_path)
    
   
    prompt_text = load_prompt(config.gen.prompt_path)
    tokenizer = BpeTokenizer.from_files(config.gen.vocab_folder, config.gen.special_tokens)
    input_tokens = tokenizer.encode(prompt_text)

    # place list of tokens into a tensor of shape (seq_len) and move to device
    input_ids = torch.tensor(input_tokens, device='cpu')
    model = build_model(config)
    output_tokens = sample(model, input_ids, config.gen.max_tokens, config.gen.temp, config.gen.top_k, config.model.seq_len)
    result = tokenizer.decode(output_tokens.tolist())

    return result

if __name__ == "__main__":
    cfg_path = "tests/config/gpt2_tiny.yaml"
    output_text = generate(cfg_path)
    print(output_text)

