from argparse import ArgumentParser
import itertools
import os
import random
import time

import torch

from tests.bpe_tokenizer import get_tokenizer_vocab_size, read_tokens_binary
from tests.nn_adamw import MyAdamW
from tests.nn_loader import get_batch, load_checkpoint, save_checkpoint
from tests.nn_scheduler import MyScheduler
from tests.nn_status_tracker import StatusTracker
from tests.nn_transformer import MyTransformer
from tests.nn_utils import calc_validation_loss, clip_gradient, compute_loss, derive_ckpt_name, plant_seed
from tests.nn_yaml import Config, load_yaml_config

def calc_total_steps(batch_size: int, context_length: int, token_budget: int) -> int:
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

def maybe_save_best(llm, optim, sched, step, tokens_processed, config, tracker: StatusTracker, validation_tokens):
    start = time.perf_counter()
    threshold = config.run.ckpt_best_below
    val_result, _ = calc_validation_loss(llm, validation_tokens, config.eval.batch_size, config.model.seq_len, config.eval.num_batches, config.eval.eval_seed, config.run.device)
    duration = time.perf_counter() - start
    new_best_val = tracker.update_validation(step, val_result, tokens_processed, duration)

    safe_to_save = new_best_val is not None and (threshold is None or new_best_val < threshold)
    if safe_to_save:
        path = save_checkpoint_file(llm, optim, sched, step, tokens_processed, config, 'best_ckpt.pt')
        tracker.update_checkpoint(step, path, True)
    return new_best_val

def save_checkpoint_file(model: MyTransformer, optimizer: MyAdamW, sched: MyScheduler, iteration, tokens_processed: int, config: Config, ckpt_name: str ):
    # validation of cpt suffix to ensure cpt will have a valid file name
    assert config.run.keep_last_ckpts <= 26, f'Numer of saved checkpoints cannot exceed 26, but got: {config.run.keep_last_ckpts}'

    os.makedirs(config.dict['runs_folder'], exist_ok=True)

    # derive path to ckpt file
    out_path = os.path.join(config.dict['runs_folder'], ckpt_name)

    sched_info = sched.as_dict()
    sched_info['batch_size'] = config.train.batch_size
    sched_info['grad_accum'] = config.train.grad_accum
    sched_info['tokens_processed'] = tokens_processed
    save_checkpoint(model, optimizer,  iteration, out_path, sched_info)
    return out_path

def init_run_state(model, optimizer, config: Config, total_steps) -> tuple[int,int,MyScheduler]:
    cpt = config.run.resume_from
    if cpt is not None:
        StatusTracker.log(f'Resuming from checkpoint {cpt}')
        output_dir = config.dict['runs_folder']
        src = os.path.join(output_dir, cpt)        
        if not os.path.exists(src):
            raise Exception(f'Checkpoint not loaded - file does not exist: {src}')
        step, sched_info = load_checkpoint(src, model, optimizer,  config.run.device)
        next_step = step + 1
        sched = MyScheduler.from_state_dict(sched_info)
        StatusTracker.log(f'Resuming from step: {step}, tokens processed: {sched_info["tokens_processed"]:_} from file: {src}. Keeping original lr schedule: {sched}')
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

def train(raw_cfg, config: Config, training_generator: torch.Generator):

    llm = build_model(config)

    optim = build_optimizer(llm.parameters(), config)

    total_steps = calc_total_steps(config.train.batch_size, config.model.seq_len, config.token_budget)

    start_step, tokens_processed, sched = init_run_state(llm, optim, config, total_steps)

    tracker = StatusTracker(tokens_processed, total_steps, sched.as_dict(), raw_cfg, config, llm.num_params)
    msg=f"Total steps: {total_steps:_}, Total tokens budget: {config.token_budget:_}, effective batch size: {config.train.batch_size}, grad_accum: {config.train.grad_accum}, runs folder: {config.dict['runs_folder']} "
    tracker.log(msg)

    training_tokens, validation_tokens = load_tokens(config)

    # config.train.batch_size stays the EFFECTIVE batch size; add config.train.grad_accum (int, default 1).

    accum = config.train.grad_accum
    assert config.train.batch_size % accum == 0, f'batch_size {config.train.batch_size} must be divisible by grad_accum {accum}'
    micro_batch = config.train.batch_size // accum          # integer

    # step is the OPTIMIZER step index and starts at start_step (matters on resume)
    step, lr, grad_norm = start_step, 0, 0
    mean_loss = torch.zeros(())                              # last completed step's mean loss
    loss_sum = 0.0                                           # becomes a tensor after first add
    load_time = 0.0

    keep_training = True

    optim.zero_grad()
    for micro_step in itertools.count():                     # boundary test is relative, so no start offset

        start_load = time.perf_counter()
        input_tokens, output_tokens = get_batch(training_tokens, micro_batch, config.model.seq_len, training_generator, config.run.device)
        load_time = load_time + (time.perf_counter() - start_load)  # for logging only, not used in any calculations

        tokens_processed += input_tokens.numel()

        raw_loss = compute_loss(llm, input_tokens, output_tokens)
        (raw_loss / accum).backward()                        # scale only for the gradient
        loss_sum = loss_sum + raw_loss.detach()              # for logging

        if (micro_step + 1) % accum != 0:                    # window not complete yet
            continue

        # ---------- optimizer step: same body and order as the old loop ----------
        grad_norm = clip_gradient(llm.parameters(), config.train.max_norm, config.train.grad_eps)
        lr = sched.calc_learning_rate(step + 1)
        optim.set_lr(lr)
        optim.step()
        optim.zero_grad()

        mean_loss = loss_sum / accum                         # mean over the window

        log_now = is_cadence_hit(step, config.run.log_every_steps)
        if log_now:
            tracker.update(step, mean_loss.item(), lr, grad_norm, tokens_processed, load_time)

        eval_now = is_cadence_hit( step, config.eval.eval_every_steps)
        if eval_now:
            maybe_save_best(llm, optim, sched, step, tokens_processed, config, tracker, validation_tokens)

        save_now = is_cadence_hit(step, config.run.save_every_steps)
        if save_now:
            ckpt_name = derive_ckpt_name(step, config.run.save_every_steps, config.run.keep_last_ckpts)
            ckpt_path = save_checkpoint_file(llm, optim, sched, step, tokens_processed, config, ckpt_name)
            tracker.update_checkpoint(step, ckpt_path)

        # stop only on optimizer-step boundaries, so `step` is the last completed step at break
        if tokens_processed >= config.token_budget:
            StatusTracker.log(f'Number of processed tokens: {tokens_processed:_} reached tokens budget: {config.token_budget:_}. Now training stops!')
            keep_training = False

        if config.run.num_steps_dbg is not None and step >= start_step + config.run.num_steps_dbg:
            StatusTracker.log( f"Completed {config.run.num_steps_dbg:_} debug updates. Now training stops!" )
            keep_training = False

        if not keep_training:
            break   

        step += 1
        loss_sum = 0.0
        load_time = 0.0

    tracker.log('Training loop completed. Reporting final loss metrics, and computing final validation loss and uploading best artifact')
    # always print everything at the end 
    # since we are saving periodic checkpoints for the purpose of continuation in case of crash, and this is the end, there is no need to save the final checkpoint
    tracker.update(step, mean_loss.item(), lr, grad_norm, tokens_processed, load_time)

    # save best  checkpoint and upload artifact
    # on the slim chance last step was the best this line would be repeating the same work: calc validation  loss, if best 
    maybe_save_best(llm, optim, sched, step, tokens_processed, config, tracker, validation_tokens)
    tracker.finalize()

    tracker.log(f"Training completed: step={step} | tokens={tokens_processed:_}. ")


def main(args):
    raw_cfg, config = load_yaml_config(args.config, args)
    training_generator = plant_seed(config.run.seed)
    train(raw_cfg, config, training_generator)

if __name__ == '__main__':
    """
    Usage: 
    python train.py -c tests/config/gpt2_tiny.yaml -plr 0.0003
    """
    parser = ArgumentParser(description="Train a transformer model.")
    parser.add_argument('-c', '--config', type=str, default='tests/config/gpt2_tiny.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('-wandbt', '--wandb_tags', type=str, default='')
    parser.add_argument('-plr','--peak_lr', type=float, help='Max learning rate before cosine annealing')
    parser.add_argument('-tb', '--token_budget', type=str, default='extra-small', help="Token budget: 'xs'/'extra-small' or 's'/'small' or 'l'/'large' or 'm'/'medium'")
    parser.add_argument('-wf', '--warmup_frac', type=float, help="Warmup frac of cosine annealing")
    parser.add_argument('-s', '--seed', type=int, help="Prime number to control randomness")
    parser.add_argument('-ga', '--grad_accum', type=int, help="Number of gradient accumulation steps")
    parser.add_argument('-wd', '--weight_decay', type=float, help="Optimizers weight decay factor")
    parser.add_argument('-fm', '--forward_mode', type=str, help='Activation checkpointing: plain | checkpoint (case-insensitive)')
    args = parser.parse_args()
    
    main(args)