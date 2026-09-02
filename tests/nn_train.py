from datetime import datetime
import os

from einops import rearrange
import torch
from torch import nn
import yaml

from tests.bpe_tokenizer import read_tokens_binary
from tests.nn_adamw import MyAdamW
from tests.nn_loader import get_batch, load_checkpoint, save_checkpoint
from tests.nn_status_tracker import StatusTracker
from tests.nn_transformer import MyTransformer
from tests.nn_utils import clip_gradient, cross_entropy, get_lr_cosine_sched, resolve_device


def build_model(cfg):
    dd = cfg['model']
    llm = MyTransformer(dd['vocab_size'], dd['num_layers'], dd['seq_len'], dd['d_model'], dd['num_heads'], dd['d_ff'], device=cfg['run']['device'])
    return llm

def build_optimizer(params, cfg):
    dd = cfg['optimizer']
    optim = MyAdamW(params, dd['lr'], dd['weight_decay'], dd['betas'], dd['eps'])
    return optim

def read_tokens(cfg):
    tokens_file = cfg['data']['train_bin']
    dtype = cfg['data']['dtype']
    x = read_tokens_binary(tokens_file, dtype)
    return x

def get_tokens_batch(tokens_block, cfg):
    input_tokens, output_tokens = get_batch(tokens_block, cfg['train']['batch_size'], cfg['model']['seq_len'], cfg['run']['device'])
    return input_tokens, output_tokens

def calc_learning_rate(iteration, config):
    cfg = config['scheduler']
    lr = get_lr_cosine_sched(iteration, cfg['maxrate'], cfg['minrate'], cfg['tw'], cfg['tc'])
    return lr

def save_checkpoint_cyclic(model, optimizer, iteration, cfg):
    folder = cfg['run']['output_dir']
    os.makedirs(folder, exist_ok=True)

    save_every_steps = cfg['run']['save_every_steps']
    keep_last = cfg['run']['keep_last_ckpts']
    slot = iteration // save_every_steps % keep_last   # or a running counter
    out_path = os.path.join(folder, f'ckpt_{slot}.pt')
    save_checkpoint(model, optimizer, iteration, out_path)
    return out_path

def resume_checkpoint(model, optimizer, cfg):
    cpt = cfg['run']['resume_from']
    if cpt is not None:
        src = os.path.join(cfg['run']['output_dir'], cpt)
        step = load_checkpoint(model, optimizer, src) + 1
    else:
        step = -1
    return step

    
def train(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        cfg['run']['device']  = resolve_device(cfg['run']['device'])

    tokens_block = read_tokens(cfg)

    llm = build_model(cfg)
    llm.train()

    optim = build_optimizer(llm.parameters(), cfg)

    num_steps = cfg['scheduler']['tw'] + cfg['scheduler']['tc']

    tracker = StatusTracker(num_steps, print_every=cfg['run']['save_every_steps'], avg_window=100)

    start_step = resume_checkpoint(llm, optim, cfg) if cfg['run']['resume_from'] is not None else 0
    for step in range(start_step, num_steps):

        input_tokens, output_tokens = get_tokens_batch(tokens_block, cfg)

        optim.zero_grad()
        logits = llm(input_tokens)
        logits = rearrange(logits,'BB CC DD -> (BB CC) DD')
        output_tokens = rearrange(output_tokens, 'BB CC -> (BB CC)')
        loss = cross_entropy(logits, output_tokens) # TODO rearange dims

        # back prpagation
        loss.backward()
        grad_norm = clip_gradient(llm.parameters(), cfg['train']['max_norm'], cfg['train']['grad_eps'])
        lr = calc_learning_rate(step + 1, cfg)
        optim.set_lr(lr)
        optim.step()

        save_now = (step > 0 and step % cfg['run']['save_every_steps'] == 0) or step == num_steps
        if save_now:
            out_path = save_checkpoint_cyclic(llm, optim, step, cfg)
            tracker.update(step, loss.item(), lr, grad_norm, input_tokens.numel(), out_path)

    

def main():
    cfg_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\tests\config\gpt2_tiny.yaml'
    train(cfg_path)

if __name__ == '__main__':
    # python train.py --config config/gpt2-tiny.yaml
    main() # TODO pass path as arg




