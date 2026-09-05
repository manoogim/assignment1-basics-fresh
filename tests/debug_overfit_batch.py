# debug_overfit_batch.py

from tests.bpe_tokenizer import read_tokens_binary
from tests.nn_loader import get_batch
from tests.nn_transformer import MyTransformer
from tests.nn_yaml import load_yaml_config
from tests.nn_adamw import MyAdamW
from tests.nn_utils import compute_loss, clip_gradient

def main(cfg_path):
    _, config = load_yaml_config(cfg_path)

    llm =  MyTransformer.from_config(config.model, config.run.device)          # reuse — same model construction as real training
    llm.train()

    optim = MyAdamW(llm.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999))  # reuse class, different args

    training_tokens = read_tokens_binary(config.data.train_bin, config.data.dtype)
    input_tokens, output_tokens = get_batch(training_tokens, batch_size=4, ctx_len=32, device=config.run.device)  # fetch ONCE

    for step in range(500):
        optim.zero_grad()
        loss = compute_loss(llm, input_tokens, output_tokens)   # same fixed batch every step
        loss.backward()
        clip_gradient(llm.parameters(), 1.0, eps=1e-6)  # keep clipping on
        optim.step()   # no set_lr call at all — fixed lr from construction
        print(f"[{step}] loss={loss.item():.4f}")

if __name__ == '__main__':
    main(r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\tests\config\gpt2_tiny.yaml')