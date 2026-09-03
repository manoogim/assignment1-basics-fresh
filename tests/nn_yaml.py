import yaml

from tests.nn_utils import resolve_device

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
    tw: int
    tc: int
    minrate: float
    maxrate: float

class DataConfig(NamedTuple):
    train_bin: str
    val_bin: str
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

class Config(NamedTuple):
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    scheduler: SchedulerConfig
    data: DataConfig
    run: RunConfig
    eval: EvalConfig

def load_yaml_config(cfg_path):
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
        raw['run']['device']  = resolve_device(raw['run']['device'])

    return raw, Config(
        model=ModelConfig(**raw['model']),
        optimizer=OptimizerConfig(**raw['optimizer']),
        train=TrainConfig(**raw['train']),
        scheduler=SchedulerConfig(**raw['scheduler']),
        data=DataConfig(**raw['data']),
        run=RunConfig(**raw['run']),
        eval=EvalConfig(**raw['eval'])
    )

if __name__ == "__main__":
    cfg_path = "tests/config/gpt2_tiny.yaml"
    config = load_yaml_config(cfg_path)
    print(config)