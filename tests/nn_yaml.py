from enum import Enum

import torch
import yaml

from typing import NamedTuple, Optional

class TokenBudget(Enum):
    SMALL = 70_000_000
    MEDIUM = 115_000_000
    LARGE = 327_680_000

    @classmethod
    def from_arg(cls, arg: str ) -> "TokenBudget":
        a = arg.lower().strip()
        if a in ("s", "small"):
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
    out_prefix: str
    save_every_steps: int
    log_every_steps: int
    keep_last_ckpts: int            # keep under 27.. suffix will be a letter a-z
    resume_from: Optional[str]
    avg_window: int

class EvalConfig(NamedTuple):
    num_batches: int
    batch_size: int
    eval_every_steps: int
    target_loss: float
    best_ckpt: str
    eval_seed: int

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
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    scheduler: SchedulerConfig
    data: DataConfig
    run: RunConfig
    eval: EvalConfig
    gen: GenConfig
    wandb: WandbConfig
    
def load_yaml_config(cfg_path, args):
    """
    This loads config objects from yaml, and combines with optional overrides.
    Names of overrideable params are combined from training workflow and eval workflow
    """    
    wandb_tags_raw = getattr(args, "wandb_tags", None)
    override_token_budget = getattr(args, "token_budget", None)
    override_peak_lr = getattr(args, "peak_lr", None)
    override_warmup_frac = getattr(args, "warmup_frac", None)
    override_seed = getattr(args, "seed", None)
    override_weight_decay = getattr(args, "weight_decay", None)

    override_num_batches = getattr(args, 'num_batches', None)
    override_best_ckpt = getattr(args, 'best_path', None)
    override_eval_seed = getattr(args, 'eval_seed', None)

    xtra_tags = (
        [tag.strip() for tag in wandb_tags_raw.split(",") if tag.strip()]
        if wandb_tags_raw
        else []
    )

    msg = f"""
$$$ Using configuration file: {cfg_path}
    Sweep tags: {xtra_tags}
    Sweep overrides → Token budget: {override_token_budget}, peak_lr: {override_peak_lr}
    warmup_frac: {override_warmup_frac}, seed: {override_seed}, weight_decay: {override_weight_decay},
    num_batches: {override_num_batches}, best_ckpt: {override_best_ckpt}, eval_seed: {override_eval_seed}
"""
    print(msg.strip())


    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    raw['run']['device'] = resolve_device(raw['run']['device'])
    raw['my_path'] = cfg_path
    raw['wandb']['tags'] = xtra_tags

    # overrider peak learning rate
    if override_peak_lr is not None:
        raw['optimizer']['lr'] = override_peak_lr
        raw['scheduler']['maxrate'] = override_peak_lr

    if override_warmup_frac is not None:
        raw['scheduler']['warmup_frac'] = override_warmup_frac

    if override_seed is not None:
        raw['run']['seed'] = override_seed

    if override_weight_decay is not None:
        raw['optimizer']['weight_decay'] = override_weight_decay

    if override_num_batches is not None:
        raw['eval']['num_batches'] = override_num_batches

    if override_best_ckpt is not None:
        raw['eval']['best_ckpt'] = override_best_ckpt

    if override_eval_seed is not None:
        raw['eval']['eval_seed'] = override_eval_seed

    return raw, Config(
        token_budget=TokenBudget.from_arg(override_token_budget).value, # type: ignore
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

def resolve_device(requested: str) -> str:
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
