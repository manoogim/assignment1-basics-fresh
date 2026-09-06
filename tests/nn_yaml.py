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
    warmup_frac: float
    cosine_frac: float
    minrate: float
    maxrate: float

class DataConfig(NamedTuple):
    tokens_folder: str
    dtype: str

class RunConfig(NamedTuple):
    device: str
    seed: int
    name: str
    output_dir: str
    save_every_steps: int
    log_every_steps: int
    keep_last_ckpts: int
    checkpoint_pattern: str
    resume_from: Optional[str]
    avg_window: int
    wandb_enabled: bool

class EvalConfig(NamedTuple):
    num_batches: int
    batch_size: int
    eval_every_steps: int

class GenConfig(NamedTuple):
    temp: float
    top_k: int
    max_tokens: int
    prompt_path: str
    vocab_folder: str
    special_tokens: list[str]
    model_weights_path: str

class Config(NamedTuple):
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    scheduler: SchedulerConfig
    data: DataConfig
    run: RunConfig
    eval: EvalConfig
    gen: GenConfig
    
def load_yaml_config(cfg_path):
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
        device  = resolve_device(raw['run']['device'])
        print(f"Resolved device: {device}, CUDA available: {torch.cuda.is_available()}")
        raw['run']['device'] = device

    return raw, Config(
        model=ModelConfig(**raw['model']),
        optimizer=OptimizerConfig(**raw['optimizer']),
        train=TrainConfig(**raw['train']),
        scheduler=SchedulerConfig(**raw['scheduler']),
        data=DataConfig(**raw['data']),
        run=RunConfig(**raw['run']),
        eval=EvalConfig(**raw['eval']),
        gen = GenConfig(**raw['gen'])
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