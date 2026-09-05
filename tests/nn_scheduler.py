from tests.nn_utils import get_lr_cosine_sched
from tests.nn_yaml import SchedulerConfig


class MyScheduler:
    def __init__(self, config: SchedulerConfig, total_steps: int):
        self.type = config.type
        if self.type == 'cosine':
            assert config.warmup_frac + config.cosine_frac <= 1.0, f"warmup_frac + cosine_frac must be < 1.0, but got {config.warmup_frac} and {config.cosine_frac}"
            self.tw = int(config.warmup_frac * total_steps)
            self.tc = int(config.cosine_frac * total_steps)           
            self.minrate = config.minrate
            self.maxrate = config.maxrate
        else:
            self.maxrate = config.maxrate

    def calc_learning_rate(self, iteration):
        if self.type == 'cosine':
            lr = get_lr_cosine_sched(iteration, self.maxrate, self.minrate, self.tw, self.tc)
        else:
            lr = self.maxrate
        return lr
