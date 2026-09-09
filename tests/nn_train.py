from argparse import ArgumentParser
import itertools
import os

import torch

from tests.bpe_tokenizer import get_tokenizer_vocab_size, read_tokens_binary
from tests.nn_adamw import MyAdamW
from tests.nn_loader import get_batch, load_checkpoint, save_checkpoint
from tests.nn_scheduler import MyScheduler
from tests.nn_status_tracker import StatusTracker
from tests.nn_transformer import MyTransformer
from tests.nn_utils import calc_validation_loss, clip_gradient, compute_loss, derive_ckpt_name
from tests.nn_yaml import Config, load_yaml_config
TOTAL_TOKEN_BUDGET = 327_680_000

def calc_total_steps(batch_size: int, context_length: int, token_budget: int = TOTAL_TOKEN_BUDGET) -> int:
    return token_budget // (batch_size * context_length)

def build_model(config):
    model = MyTransformer.from_config(config.model, config.run.device)
    model.train()
    StatusTracker.log(f'Created transformer model from: {config.model}')
    if config.run.device == 'cuda':
        model.compile()
        StatusTracker.log(f"Model compiled. Resolved device: {config.run.device}, CUDA available: {torch.cuda.is_available()}")
    return model

def build_optimizer(params, config: Config):
    dd = config.optimizer
    optim = MyAdamW(params, dd.lr, dd.weight_decay, dd.betas, dd.eps)
    StatusTracker.log(f'Created AdamW optimizer from: {dd}')
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
        StatusTracker.log(f'Loaded {len(tokens):_} from {file_name}')
    
    return result['tokens_train.bin'], result['tokens_valid.bin']

def save_checkpoint_cyclic(model: MyTransformer, optimizer: MyAdamW, sched: MyScheduler, iteration, tokens_processed: int, config: Config):
    folder = config.run.output_dir
    os.makedirs(folder, exist_ok=True)

    # construct path to ckpt file
    ckpt_name = derive_ckpt_name(iteration, config.run.save_every_steps, config.run.keep_last_ckpts)
    out_path = os.path.join(folder, ckpt_name)

    sched_info = sched.as_dict()
    sched_info['batch_size'] = config.train.batch_size
    save_checkpoint(model, optimizer, sched_info, iteration, tokens_processed, out_path)
    return out_path

def init_run_state(model, optimizer, config: Config, total_steps) -> tuple[int,int,MyScheduler]:
    # validation of cpt suffix to ensure cpt will have a valid file name
    assert config.run.keep_last_ckpts <= 26, f'Numer of saved checkpoints cannot exceed 26, but got: {config.run.keep_last_ckpts}'
    cpt = config.run.resume_from
    if cpt is not None:
        StatusTracker.log(f'Resuming from checkpoint {cpt}')
        src = os.path.join(config.run.output_dir, cpt)        
        if not os.path.exists(src):
            raise Exception(f'Checkpoint not loaded - file does not exist: {src}')
        step, tokens_processed, sched_info = load_checkpoint(model, optimizer, src, config.run.device)
        next_step = step + 1
        sched = MyScheduler.from_state_dict(sched_info)
        StatusTracker.log(f'Resuming from step: {step}, tokens processed: {tokens_processed:_} from file: {src}. Keeping original lr schedule: {sched}')
        # TODO - should we raise or soft warn if new config has different learning schedule
        if sched_info['batch_size'] != config.train.batch_size:
            StatusTracker.log(f"[INFO] Resuming with batch_size={config.train.batch_size}, "
            f"differs from checkpoint's original batch_size={sched_info['batch_size']}. "
            f"Keeping original schedule (warmup_end, cosine_end, minrate, and maxrate). "
            )
    else:
        next_step = 0
        tokens_processed = 0
        sched = MyScheduler.from_config(config.scheduler, total_steps)
        StatusTracker.log(f'Learning Schedule: {sched}')

    return next_step, tokens_processed, sched # type: ignore

def is_cadence_hit (step, interval):
    """
    check if it is time to log, save checkpoint or calc validation loss
    """
    result = (step > 0 )and( step % interval == 0)    
    return result

def train(cfg_path):
    raw_cfg, config = load_yaml_config(cfg_path)
    torch.manual_seed(config.run.seed)
    
    training_tokens, validation_tokens = load_tokens(config)

    llm = build_model(config)

    optim = build_optimizer(llm.parameters(), config)

    total_steps = calc_total_steps(config.train.batch_size, config.model.seq_len, TOTAL_TOKEN_BUDGET)

    start_step, tokens_processed, sched = init_run_state(llm, optim, config, total_steps)
    StatusTracker.log(f"Total steps: {total_steps:_}, Total tokens budget: {TOTAL_TOKEN_BUDGET:_} ")
    tracker = StatusTracker(tokens_processed, TOTAL_TOKEN_BUDGET, total_steps, llm, raw_cfg, config)

    # infinite training loop (no worries it will break based on tokens_processed or validation_loss ;)
    keep_training = True
    for step in itertools.count(start_step):

        input_tokens, output_tokens = get_batch(training_tokens, config.train.batch_size, config.model.seq_len, config.run.device)
        tokens_processed += input_tokens.numel()

        optim.zero_grad()
        loss = compute_loss(llm, input_tokens, output_tokens)

        # back propagation
        loss.backward()
        grad_norm = clip_gradient(llm.parameters(), config.train.max_norm, config.train.grad_eps)
        lr = sched.calc_learning_rate(step + 1)
        optim.set_lr(lr)
        optim.step()

        log_now = is_cadence_hit(step, config.run.log_every_steps)
        if log_now:
            tracker.update(step, loss.item(), lr, grad_norm, tokens_processed )

        save_now = is_cadence_hit(step, config.run.save_every_steps)
        if save_now:
            out_path = save_checkpoint_cyclic(llm, optim, sched, step, tokens_processed, config)
            tracker.update_checkpoint(step, out_path)

        eval_now = is_cadence_hit( step, config.eval.eval_every_steps)
        if eval_now:
            val_loss = calc_validation_loss(llm, validation_tokens, config.eval.batch_size, config.model.seq_len, config.eval.num_batches, config.run.device)
            tracker.update_validation(step, val_loss)

            if config.eval.target_loss is not None and val_loss < config.eval.target_loss:
                StatusTracker.log(f'Training is stopped at step: {step}, because validation loss reached target: {val_loss:.4} <= {config.eval.target_loss}. Regular loss is {loss:.4}.')
                keep_training = False

        if tokens_processed >= TOTAL_TOKEN_BUDGET:
            StatusTracker.log(f'Number of processed tokens: {tokens_processed:_} reached tokens budget: {TOTAL_TOKEN_BUDGET:_}. Now training stops!')
            keep_training = False

        if step >= 1000:
            break
        if not keep_training:
            break   

    # always save everything at the end
    tracker.update(step, loss.item(), lr, grad_norm, tokens_processed ) # type: ignore

    out_path = save_checkpoint_cyclic(llm, optim, sched, step, tokens_processed, config) # type: ignore
    tracker.update_checkpoint(step, out_path) # type: ignore

    print(f"Training completed. Step count: {step}. Tokens processed: {tokens_processed:_}. Last loss: {loss:.4}. ") # type: ignore


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
    
    main(args.config)




