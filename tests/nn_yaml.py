from enum import Enum

import torch
from torch import nn
import yaml

from typing import NamedTuple, Optional

class TokenBudget(Enum):
    EXTRA_SMALL = 40_000_000
    SMALL = 70_000_000
    MEDIUM = 115_000_000
    LARGE = 327_680_000

    @classmethod
    def from_arg(cls, arg: str ) -> "TokenBudget":
        a = arg.lower().strip()
        if a in ('xs', 'extra-small'):
            return cls.EXTRA_SMALL
        elif a in ("s", "small"):
            return cls.SMALL
        elif a in ('m', 'medium'):
            return cls.MEDIUM
        elif a in ("l", "large"):
            return cls.LARGE
        else:
            raise ValueError(f"Invalid token budget: {arg}")
    
class ModelConfig(NamedTuple):
    vocab_size: int
    seq_len: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    forward_mode: str = 'plain'  # plain | checkpoint

class OptimizerConfig(NamedTuple):
    type: str
    weight_decay: float
    betas: tuple[float, float]
    eps: float

class TrainConfig(NamedTuple):
    batch_size: int
    grad_accum: int
    precision: str       # # float32 is CPU-friendly, float16 is GPU-friendly, bfloat16 is TPU-friendly
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
    out_prefix: str
    save_last: bool
    save_every_steps: int
    log_every_steps: int
    keep_last_ckpts: int            # keep under 27.. suffix will be a letter a-z
    resume_from: Optional[str]
    avg_window: int

class EvalConfig(NamedTuple):
    num_batches: int
    batch_size: int
    eval_every_steps: int   # cadence of computing validation loss during training
    target_loss: float      # can be used to break out of training loop
    best_ckpt: str          # for computing robust validation loss
    eval_seed: int          # make random generator deterministic

class GenConfig(NamedTuple):
    temp: float
    top_k: int
    max_tokens: int
    prompt_path: str
    vocab_folder: str
    special_tokens: list[str]
    model_weights_path: str

class WandbConfig(NamedTuple):
    enabled: bool
    proj_name: str
    tags: str | None
    active: str
    name_templates: dict

class Config(NamedTuple):
    token_budget: int 
    dict: dict
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    scheduler: SchedulerConfig
    data: DataConfig
    run: RunConfig
    eval: EvalConfig
    gen: GenConfig
    wandb: WandbConfig

def load_yaml_config(cfg_path, args=None):
    """
    This loads config objects from yaml, and combines with optional overrides.
    Names of overrideable params are combined from training workflow and eval workflow
    """    
    
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
        raw['my_path'] = cfg_path

    device = resolve_device(raw['run']['device'])
    raw['run']['device'] = device

    overrides = get_overrides(args) if args is not None else {}
    wandb_tags = []

    # overrider peak learning rate
    override_peak_lr = overrides.get('peak_lr', None)
    if override_peak_lr is not None:
        raw['scheduler']['maxrate'] = override_peak_lr
        raw['scheduler']['minrate'] = 0.1 * override_peak_lr
        wandb_tags.append(f"lr{override_peak_lr}")

    override_warmup_frac = overrides.get('warmup_frac', None)
    if override_warmup_frac is not None:
        raw['scheduler']['warmup_frac'] = override_warmup_frac
        wandb_tags.append(f"warmup_frac{override_warmup_frac}")

    override_seed = overrides.get('seed', None)
    if override_seed is not None:
        raw['run']['seed'] = override_seed
        wandb_tags.append(f"seed{override_seed}")

    override_weight_decay = overrides.get('weight_decay', None)
    if override_weight_decay is not None:
        raw['optimizer']['weight_decay'] = override_weight_decay
        wandb_tags.append(f"weight_decay{override_weight_decay}")

    override_num_batches = overrides.get('num_batches', None)
    if override_num_batches is not None:
        raw['eval']['num_batches'] = override_num_batches
        wandb_tags.append(f"eval_num_batches{override_num_batches}")

    override_best_ckpt = overrides.get('best_path', None) # for stand-alone eval script, for computing robust validation loss
    if override_best_ckpt is not None:
        raw['eval']['best_ckpt'] = override_best_ckpt
        wandb_tags.append(f"best_ckpt{override_best_ckpt}")

    override_eval_seed = overrides.get('eval_seed', None)
    if override_eval_seed is not None:
        raw['eval']['eval_seed'] = override_eval_seed
        wandb_tags.append(f"eval_seed{override_eval_seed}")

    override_token_budget = overrides.get('token_budget', None)
    if override_token_budget is not None:
        wandb_tags.append(f"token_budget{override_token_budget}")
        token_budget=TokenBudget.from_arg(override_token_budget).value
    else:
        token_budget = 0

    override_grad_accum = overrides.get('grad_accum', None)
    if override_grad_accum is not None:
        raw['train']['grad_accum'] = override_grad_accum
        
        micro_batch_size = raw['train']['batch_size'] // override_grad_accum
        wandb_tags.append(f"grad_accum{override_grad_accum}")
        wandb_tags.append(f"micro_batch_size{micro_batch_size}")

    override_forward_mode = overrides.get('forward_mode', None)
    if override_forward_mode is not None:
        raw['model']['forward_mode'] = override_forward_mode
        wandb_tags.append(f'forward_mode{override_forward_mode}')

    override_resume_from = overrides.get('resume_from', None) 
    if override_resume_from is not None:
        raw['run']['resume_from'] = override_resume_from
        wandb_tags.append(f'resume_from{override_resume_from[-20:]}')

    override_num_steps_dbg = overrides.get('num_steps_dbg', None)
    if override_num_steps_dbg is not None:
        raw['run']['num_steps_dbg'] = override_num_steps_dbg
        wandb_tags.append(f'num_steps_dbg{override_num_steps_dbg}')
        
    dd = {
        'wandb_name': '',
        'runs_folder': ''
    }
    
    raw['wandb']['tags'] = wandb_tags if len(wandb_tags) > 0 else None

    config = Config(
        token_budget=token_budget,
        dict=dd,
        model=ModelConfig(**raw['model']),
        optimizer=OptimizerConfig(**raw['optimizer']),
        train=TrainConfig(**raw['train']),
        scheduler=SchedulerConfig(**raw['scheduler']),
        data=DataConfig(**raw['data']),
        run=RunConfig(**raw['run']),
        eval=EvalConfig(**raw['eval']),
        gen = GenConfig(**raw['gen']),
        wandb = WandbConfig(**raw['wandb'])
    )
    dd['wandb_name'] = resolve_wandb_name(config)
    dd['runs_folder'] = resolve_runs_folder(config, f'{token_budget//1_000_000}mm')
    # dd['runs_folder'] = resolve_runs_folder(config)
    # validations
    accum = config.train.grad_accum
    assert config.train.batch_size % accum == 0, f'batch_size {config.train.batch_size} must be divisible by grad_accum {accum}'
    
    return raw, config


def get_overrides(args):
    overrides = {
        "token_budget": getattr(args, "token_budget", None),
        "peak_lr": getattr(args, "peak_lr", None),
        "warmup_frac": getattr(args, "warmup_frac", None),
        "seed": getattr(args, "seed", None),
        "weight_decay": getattr(args, "weight_decay", None),

        "num_batches": getattr(args, "num_batches", None),
        "best_path": getattr(args, "best_path", None),
        "eval_seed": getattr(args, "eval_seed", None),
        "grad_accum": getattr(args, "grad_accum", None),
        "forward_mode": getattr(args, "forward_mode" , None  ),
        "resume_from": getattr(args,"resume_from", None),
        "num_steps_dbg": getattr(args, "num_steps_dbg", None)
    }
    print("Overrides:", overrides)
    return overrides

def resolve_device(requested: str):
    if requested != 'auto':
        result= requested
    elif torch.cuda.is_available():
        result = 'cuda'
    elif torch.backends.mps.is_available():
        result = 'mps'
    else:
        result = 'cpu'
    return result

def _resolve_runs_folder(config: Config) -> str:
    prefix = f'{config.run.out_prefix}{resolve_wandb_name(config)}'
    folder = f'a{prefix}_{config.token_budget//1_000_000}mm_b{config.train.batch_size}'
    print(f'*** Runs folder: {folder}')
    return folder

def resolve_runs_folder(config: Config, suffix) -> str:
    prefix = f'{config.run.out_prefix}{resolve_wandb_name(config)}'
    folder = f'a{prefix}_{config.token_budget//1_000_000}mm_b{config.train.batch_size}'
 
    folder = f'{prefix}_{suffix}'
    print(f'*** Runs folder: {folder}')
    return folder

def resolve_wandb_name(config: Config) -> str:
    active = config.wandb.active
    templ = config.wandb.name_templates[active]
    return templ.format(config=config)
