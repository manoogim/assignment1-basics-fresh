import torch
import yaml

from typing import NamedTuple, Optional

class ModelConfig(NamedTuple):
    vocab_size: int
    seq_len: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int

class OptimizerConfig(NamedTuple):
    type: str
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float

class TrainConfig(NamedTuple):
    batch_size: int
    datatype: str
    max_norm: float
    grad_eps: float

class SchedulerConfig(NamedTuple):
    type: str
    warmup_frac: float # fraction of total_steps at which the warmup phase ends, which also equals length of warmup phase
    cosine_frac: float # fraction of total_steps at which the cosine phase ends, with length of cosine phase = cosine_frac - warmup_frac
    minrate: float
    maxrate: float

class DataConfig(NamedTuple):
    tokens_folder: str
    dtype: str

class RunConfig(NamedTuple):
    ckpt_best_below: float          # # only start tracking "best" once val loss crosses this
    num_steps_dbg: int              # meant for dev purposes, keep it null in prod
    device: str                     # auto | cuda | cpu | mps
    seed: int
    name: str
    out_prefix: str
    save_every_steps: int
    log_every_steps: int
    keep_last_ckpts: int            # keep under 27.. suffix will be a letter a-z
    resume_from: Optional[str]
    avg_window: int
    wandb_enabled: bool

class EvalConfig(NamedTuple):
    num_batches: int
    batch_size: int
    eval_every_steps: int
    target_loss: float

class GenConfig(NamedTuple):
    temp: float
    top_k: int
    max_tokens: int
    prompt_path: str
    vocab_folder: str
    special_tokens: list[str]
    model_weights_path: str

class NamingConfig(NamedTuple):
    active: str
    templates: dict

class Config(NamedTuple):
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    scheduler: SchedulerConfig
    data: DataConfig
    run: RunConfig
    eval: EvalConfig
    gen: GenConfig
    naming: NamingConfig
    
def load_yaml_config(cfg_path, args):
    override_token_budget = args.token_budget
    override_peak_lr = args.peak_lr
    override_warmup_frac = args.warmup_frac
    override_seed = args.seed

    msg = (f'$$$ Using configuration file: {cfg_path} with sweep param overrides peak_lr: {override_peak_lr}, warmup_frac: {override_warmup_frac}, seed={override_seed}, Token budget: {override_token_budget}')
    print(msg)

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    raw['run']['device'] = resolve_device(raw['run']['device'])
    raw['my_path'] = cfg_path

    # overrider peak learning rate
    if override_peak_lr is not None:
        raw['optimizer']['lr'] = override_peak_lr
        raw['scheduler']['maxrate'] = override_peak_lr

    if override_warmup_frac is not None:
        raw['scheduler']['warmup_frac'] = override_warmup_frac

    if override_seed is not None:
        raw['run']['seed'] = override_seed

    return raw, Config(
        model=ModelConfig(**raw['model']),
        optimizer=OptimizerConfig(**raw['optimizer']),
        train=TrainConfig(**raw['train']),
        scheduler=SchedulerConfig(**raw['scheduler']),
        data=DataConfig(**raw['data']),
        run=RunConfig(**raw['run']),
        eval=EvalConfig(**raw['eval']),
        gen = GenConfig(**raw['gen']),
        naming = NamingConfig(**raw['naming'])
    )

def resolve_device(requested: str) -> str:
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

if __name__ == "__main__":
    cfg_path = "tests/config/gpt2_tiny.yaml"
    config = load_yaml_config(cfg_path)
    print(config)