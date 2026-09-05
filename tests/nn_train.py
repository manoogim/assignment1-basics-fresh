import os

from tests.bpe_tokenizer import read_tokens_binary
from tests.nn_adamw import MyAdamW
from tests.nn_loader import get_batch, load_checkpoint, save_checkpoint
from tests.nn_status_tracker import StatusTracker
from tests.nn_transformer import MyTransformer
from tests.nn_utils import calc_validation_loss, clip_gradient, compute_loss,  get_lr_cosine_sched
from tests.nn_yaml import Config, load_yaml_config

def build_optimizer(params, config: Config):
    dd = config.optimizer
    optim = MyAdamW(params, dd.lr, dd.weight_decay, dd.betas, dd.eps)
    return optim

def load_tokens(config: Config):
    dtype = config.data.dtype
    training_tokens = read_tokens_binary(config.data.train_bin, dtype)
    validation_tokens = read_tokens_binary(config.data.val_bin, dtype)
    return training_tokens, validation_tokens

def calc_learning_rate(iteration, config: Config):
    cfg = config.scheduler
    lr = get_lr_cosine_sched(iteration, cfg.maxrate, cfg.minrate, cfg.tw, cfg.tc)
    return lr

def save_checkpoint_cyclic(model, optimizer, iteration, config: Config):
    folder = config.run.output_dir
    os.makedirs(folder, exist_ok=True)

    save_every_steps =  config.run.save_every_steps
    keep_last = config.run.keep_last_ckpts
    slot = iteration // save_every_steps % keep_last   # or a running counter
    out_path = os.path.join(folder, f'ckpt_{slot}.pt')
    save_checkpoint(model, optimizer, iteration, out_path)
    return out_path

def resume_checkpoint(model, optimizer, config: Config):
    cpt = config.run.resume_from
    if cpt is not None:
        src = os.path.join(config.run.output_dir, cpt)
        step = load_checkpoint(model, optimizer, src, config.run.device) + 1
    else:
        step = -1
    return step

    
def train(cfg_path):
    raw_cfg, config = load_yaml_config(cfg_path)

    training_tokens, validation_tokens = load_tokens(config)

    llm = MyTransformer.from_config(config.model, config.run.device)
    llm.train()

    optim = build_optimizer(llm.parameters(), config)

    num_steps = config.scheduler.tw + config.scheduler.tc
    tracker = StatusTracker(num_steps, llm, raw_cfg, config)

    start_step = resume_checkpoint(llm, optim, config) if config.run.resume_from is not None else 0

    for step in range(start_step, num_steps):

        input_tokens, output_tokens = get_batch(training_tokens, config.train.batch_size, config.model.seq_len, config.run.device)

        optim.zero_grad()
        loss = compute_loss(llm, input_tokens, output_tokens)

        # back propagation
        loss.backward()
        grad_norm = clip_gradient(llm.parameters(), config.train.max_norm, config.train.grad_eps)
        lr = calc_learning_rate(step + 1, config)
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



    

def main():
    cfg_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\tests\config\gpt2_tiny.yaml'
    train(cfg_path)

if __name__ == '__main__':
    # python train.py --config config/gpt2-tiny.yaml
    main() # TODO pass path as arg




