import os

import torch
import yaml

from tests.bpe_tokenizer import BpeTokenizer
from tests.nn_loader import load_checkpoint
from tests.nn_transformer import MyTransformer
from tests.nn_yaml import Config, load_yaml_config

def build_model(config: Config):
    model = MyTransformer.from_config(config.model, 'cpu')
    load_checkpoint(config.gen.model_weights_path, model, None, None, 'cpu')
    model.eval()
    return model


def top_k_logits(logits, k):
    if k == 0:
        return logits
    topk_values, topk_indices = torch.topk(logits, k)
    threshold = topk_values[-1]
    epses = torch.ones_like(logits, dtype=logits.dtype) * -1e10
    result = torch.where(logits < threshold, epses, logits)
    return result

@torch.inference_mode()
def sample(model, input_ids, eos_ids: list[int], max_new_tokens: int, temp: float, top_k:int, max_seq_len:int):
    if temp < 0:
        raise ValueError("Temperature must be nonnegative")
    
    for _ in range(max_new_tokens): 
        if input_ids.numel() >= max_seq_len:
            print(f"*** Hard stop *** Input_ids length ({input_ids.numel()}) exceeds max_seq_len ({max_seq_len})")    
            return input_ids  
        all_logits = model(input_ids)
        # last token logits
        logits = all_logits[ -1, :]  

        if temp == 0:
            next_token = logits.argmax(dim=-1).reshape(1)
        else:
            logits = top_k_logits(logits, top_k)    
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token])
        
        is_end_of_stream = next_token.item() in eos_ids
        if is_end_of_stream:
            break

    return input_ids


"""
This code generates text using batch_size=1
"""
def generate(config, model, tokenizer, prompt_text):
    
    input_tokens = tokenizer.encode(prompt_text)
    eos_ids = tokenizer.encode_special()

    # place list of tokens into a tensor of shape (seq_len) and move to device
    input_ids = torch.tensor(input_tokens, device='cpu')

    max_tokens, temp, top_k, seq_len = config.gen.max_tokens, config.gen.temp, config.gen.top_k, config.model.seq_len
    output_tokens = sample(model, input_ids, eos_ids, max_tokens, temp, top_k, seq_len )
    result = tokenizer.decode(output_tokens.tolist())

    return result

def log_config(config):
    desc = (
           f"checkpoint={config.gen.model_weights_path}, "
            f"temp={config.gen.temp}, top_k={config.gen.top_k}, "
            f"max_new_tokens={config.gen.max_tokens}, "
            f"seed={config.run.seed}"
    )
    return desc

def write_generated(config:Config, data: dict):
    y = yaml.safe_dump(data=data, allow_unicode=True, width=120)

    file_name = f'generated_temp{config.gen.temp}_max{config.gen.max_tokens}_seed{config.run.seed}.txt'
    folder = os.path.dirname(config.gen.model_weights_path)
    path = os.path.join(folder, file_name)
    print(f'Saving generated text in: {path}')

    with open(path, 'w') as f:
        f.write(y)

def main(config: Config):
    model = build_model(config)
    tokenizer = BpeTokenizer.from_files(config.gen.vocab_folder, config.gen.special_tokens)

    print(log_config(config)) 

    with open(config.gen.prompt_path) as f:
        prompt_dict = yaml.safe_load(f)

    generated = {'config': log_config(config)}
    for key, prompt_text in prompt_dict.items():

        # make each prompt reproducible 
        torch.manual_seed(config.run.seed)

        output_text = generate(config, model, tokenizer, prompt_text)
        input_output = f'{prompt_text} >>>|<<< {output_text}'
        print(input_output)
        print('_______________________________________________')

        generated[key] = input_output
    return generated


if __name__ == "__main__":
    cfg_path = "tests/config/abla_batch/cs336_owt.yaml"
    _, config = load_yaml_config(cfg_path, None)
  
    generated = main(config)

    write_generated(config, generated)

