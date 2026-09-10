from tests.nn_utils import get_lr_cosine_sched
from tests.nn_yaml import SchedulerConfig


class MyScheduler:

    def __init__(self):
        self.type = None
        self.warmup_end = None
        self.cosine_end = None
        self.minrate = None
        self.maxrate = None
        self.total_steps = None  

    def __repr__(self):
        def fmti(x):
            return f"{x:_}" if x is not None else "None"
        def fmtf(x):
            return f"{x:.8f}" if x is not None else "None"

        return (
            f"<warmup_end={fmti(self.warmup_end)}, "
            f"cosine_end={fmti(self.cosine_end)}, "
            f"minrate={fmtf(self.minrate)}, "
            f"maxrate={fmtf(self.maxrate)}, "
            f"total_steps={fmti(self.total_steps)}>"
        )

    
    @classmethod
    def from_config(cls, config: SchedulerConfig, total_steps: int):
        self = MyScheduler()
        self.type = config.type
        if self.type == 'cosine':
            assert config.warmup_frac >= 0, f'Cosine frac is negative: {config.warmup_frac}'
            assert config.warmup_frac <= config.cosine_frac, f"Warmup_frac must finish before cosine_frac finishes (or even starts), but got {config.warmup_frac} and {config.cosine_frac}"
            assert config.minrate <= config.maxrate, f'Min rate {config.minrate} cannot be bigger than max rate {config.maxrate}'
            self.warmup_end = int(config.warmup_frac * total_steps)  # warmup_end_step 
            self.cosine_end = int(config.cosine_frac * total_steps)           
            self.minrate = config.minrate
            self.maxrate = config.maxrate
            self.total_steps = total_steps
        else:
            self.maxrate = config.maxrate
        return self

    @classmethod
    def from_state_dict(cls, dict):
        if (dict['type'] != 'cosine') :
            raise Exception('Only cosine scheduler is supported')  
        self = MyScheduler()
        self.type = 'cosine'
        self.warmup_end = dict['warmup_end']
        self.cosine_end = dict['cosine_end']
        self.minrate = dict['minrate']
        self.maxrate = dict['maxrate']
        self.total_steps = dict['total_steps']       
        return self

    def as_dict(self):
        dict = {}
        dict['type'] = self.type
        dict['warmup_end'] = self.warmup_end
        dict['cosine_end'] = self.cosine_end
        dict['minrate'] = self.minrate
        dict['maxrate'] = self.maxrate
        dict['total_steps'] = self.total_steps
        return dict   

    def calc_learning_rate(self, iteration):
        if self.type == 'cosine':
            lr = get_lr_cosine_sched(iteration, self.maxrate, self.minrate, self.warmup_end, self.cosine_end)
        else:
            lr = self.maxrate
        return lr
