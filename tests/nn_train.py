from argparse import ArgumentParser
import os

import torch

from tests.bpe_tokenizer import get_tokenizer_vocab_size, read_tokens_binary
from tests.nn_adamw import MyAdamW
from tests.nn_loader import get_batch, load_checkpoint, save_checkpoint
from tests.nn_scheduler import MyScheduler
from tests.nn_status_tracker import StatusTracker
from tests.nn_transformer import MyTransformer
from tests.nn_utils import calc_validation_loss, clip_gradient, compute_loss
from tests.nn_yaml import Config, load_yaml_config
TOTAL_TOKEN_BUDGET = 327_680_000

def calc_total_steps(batch_size: int, context_length: int, token_budget: int = TOTAL_TOKEN_BUDGET) -> int:
    return token_budget // (batch_size * context_length)

def build_model(config):
    model = MyTransformer.from_config(config.model, config.run.device)
    model.train()
    if torch.cuda.is_available():
        model.compile()
        StatusTracker.log(f"Model compiled. Resolved device: {config.run.device}, CUDA available: {torch.cuda.is_available()}")
    return model

def build_optimizer(params, config: Config):
    dd = config.optimizer
    optim = MyAdamW(params, dd.lr, dd.weight_decay, dd.betas, dd.eps)
    return optim

def load_tokens(config: Config):
    # validate that the vocab size in the config matches the tokenizer's vocab size
    folder_name = config.data.tokens_folder
    vocab_path = os.path.join(folder_name, 'vocab_readable.txt')
    vocab_size = get_tokenizer_vocab_size(vocab_path)
    if vocab_size != config.model.vocab_size:
        raise ValueError(f'vocab_size mismatch: vocab_readable.txt has {vocab_size} entries, but config.model.vocab_size={config.model.vocab_size} found in {vocab_path}')

    result = {}
    for file_name in ['tokens_train.bin', 'tokens_valid.bin']:
        tokens_file = os.path.join(folder_name, file_name)
        tokens = read_tokens_binary(tokens_file, config.data.dtype)
        # extra validation to prevent crashing if the tokenizer vocab size is smaller than the config.model.vocab_size
        max_id = tokens.max()
        if max_id >= config.model.vocab_size:
            raise ValueError(f'{file_name} max id ({max_id}) exceeds config.model.vocab_size ({config.model.vocab_size})')
        else:
            result[file_name] = tokens
    
    return result['tokens_train.bin'], result['tokens_valid.bin']

def save_checkpoint_cyclic(model, optimizer, iteration, config: Config):
    folder = config.run.output_dir
    os.makedirs(folder, exist_ok=True)

    save_every_steps =  config.run.save_every_steps
    keep_last = config.run.keep_last_ckpts
    slot = iteration // save_every_steps % keep_last   # or a running counter
    out_path = os.path.join(folder, f'ckpt_{slot}.pt')
    save_checkpoint(model, optimizer, iteration, out_path)
    StatusTracker.log(f'Saving checkpoint at step: {iteration} to file: {out_path}')
    return out_path

def resume_checkpoint(model, optimizer, config: Config):
    cpt = config.run.resume_from
    if cpt is not None:
        src = os.path.join(config.run.output_dir, cpt)
        if not os.path.exists(src):
            raise Exception(f'Checkpoint not loaded - file does not exist: {src}')
        
        step = load_checkpoint(model, optimizer, src, config.run.device) + 1

        StatusTracker.log(f'Resuming step: {step} from file: {src}')
    else:
        step = -1
    return step

    
def train(cfg_path):
    raw_cfg, config = load_yaml_config(cfg_path)
    torch.manual_seed(config.run.seed)
    
    num_steps = calc_total_steps(config.train.batch_size, config.model.seq_len, TOTAL_TOKEN_BUDGET)

    sched = MyScheduler(config.scheduler, num_steps)

    training_tokens, validation_tokens = load_tokens(config)

    llm = build_model(config)

    optim = build_optimizer(llm.parameters(), config)
    
    tracker = StatusTracker(num_steps, llm, raw_cfg, config)

    start_step = resume_checkpoint(llm, optim, config) if config.run.resume_from is not None else 0

    for step in range(start_step, num_steps):

        input_tokens, output_tokens = get_batch(training_tokens, config.train.batch_size, config.model.seq_len, config.run.device)

        optim.zero_grad()
        loss = compute_loss(llm, input_tokens, output_tokens)

        # back propagation
        loss.backward()
        grad_norm = clip_gradient(llm.parameters(), config.train.max_norm, config.train.grad_eps)
        lr = sched.calc_learning_rate(step + 1)
        optim.set_lr(lr)
        optim.step()

        log_now = (step > 0 and step % config.run.log_every_steps == 0) or (step == num_steps - 1)
        if log_now:
            tracker.update(step, loss.item(), lr, grad_norm, input_tokens.numel() )

        save_now = (step > 0 and step % config.run.save_every_steps == 0) or (step == num_steps - 1)
        if save_now:
            out_path = save_checkpoint_cyclic(llm, optim, step, config)
            tracker.update_checkpoint(step, out_path)

        eval_now = ( step > 0 and step % config.eval.eval_every_steps == 0) or (step == num_steps - 1)
        if eval_now:
            val_loss = calc_validation_loss(llm, validation_tokens, 
                                            config.eval.batch_size, 
                                            config.model.seq_len, 
                                            config.eval.num_batches, 
                                            config.run.device)
            tracker.update_validation(step, val_loss)

            if config.eval.target_loss is not None and val_loss < config.eval.target_loss:
                StatusTracker.log(f'Training is stopped at step: {step}, because validation loss reached target: {val_loss:.4} <= {config.eval.target_loss}. Regular loss is {loss:.4}.')
                break

    print(f"Training completed. Last step: {step}. Last loss: {loss:.4}. ") # type: ignore


    

def main(cfg_path = 'config/cs336_basic.yaml'):
    StatusTracker.log(f'Using configuration file: {cfg_path}')
    train(cfg_path)

if __name__ == '__main__':
    """
    Usage: 
    python train.py --config config/gpt2-tiny.yaml
    """
    parser = ArgumentParser(description="Train a transformer model.")
    parser.add_argument('-c', '--config', type=str, default='tests/config/cs336_basic.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    main(args.config) # TODO pass path as arg




