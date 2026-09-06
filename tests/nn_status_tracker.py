import math
import time
import os
import psutil
import wandb

from tests.nn_transformer import MyTransformer
from tests.nn_yaml import Config

def safe_ppl(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')
    
class StatusTracker:
    def __init__(self, total_steps, model: MyTransformer, raw_cfg, config: Config):
        tokens_budget = config.train.batch_size * total_steps * config.model.seq_len
    
        self.total_steps = total_steps
        self.avg_window = config.run.avg_window
        self.log_every_steps = config.run.log_every_steps
        self.model = model

        self.loss_history = []
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = 0

        self.log(f"Total steps: {total_steps:_}, Total tokens budget: {tokens_budget:_} ")

        if config.run.wandb_enabled:
            self.wandb_enabled = True
            wandb.init(project=config.run.name, name=config.run.name, config=raw_cfg)
        else:
            self.wandb_enabled = False

    @classmethod
    def log(cls, msg):
        print(f'@@@ {msg} @@@')

    def update(self, step, loss, lr, grad_norm, batch_tokens):
        # Track loss
        self.loss_history.append(loss)
        if len(self.loss_history) > self.avg_window:
            self.loss_history.pop(0)

        # Compute averages
        avg_loss = sum(self.loss_history) / len(self.loss_history)
        min_loss = min(self.loss_history)

        # Compute perplexity
        perplexity = safe_ppl(loss)
        avg_perplexity = safe_ppl(avg_loss)

        # Timing
        now = time.time()
        time_since_last_update = now - self.last_time
        self.last_time = now

        tokens_processed_since_last_update = self.log_every_steps * batch_tokens
        throughput = tokens_processed_since_last_update / time_since_last_update

        # ETA
        elapsed = now - self.start_time
        steps_done = step + 1
        steps_per_sec = steps_done / elapsed
        eta_seconds = (self.total_steps - steps_done) / steps_per_sec

        # Print periodic status

        print(f"[{step}] loss={loss:.4f} avg_loss({self.avg_window})={avg_loss:.4f} min_loss={min_loss:.4f}")
        print(f"      ppl={perplexity:.2f} avg_ppl={avg_perplexity:.2f}")
        print(f"      lr={lr:.6f} grad_norm={grad_norm:.4f}")
        print(f"      throughput={throughput:.1f} tokens/sec")
        print(f"      elapsed={self._fmt(elapsed)} eta={self._fmt(eta_seconds)}")
        print(f"      rss={self._rss():.2f}GB")
        if self.wandb_enabled:
            wandb.log({
                "loss": loss,
                "avg_loss": avg_loss,
                "perplexity": perplexity,
                "avg_perplexity": avg_perplexity,
                "lr": lr,
                "grad_norm": grad_norm,
                "throughput": throughput,
                "rss": self._rss(),
                "step": step,
                "wallclock_secs": elapsed,
            }, step=step)

        
    def update_checkpoint(self, step, ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"[{step}] Saved checkpoint: {ckpt_path} ({size_mb:.1f}MB)")
        if self.wandb_enabled:
            wandb.log({
                "step": step,
                "checkpoint": ckpt_path,
                "checkpoint_size_mb": size_mb
            }, step=step)

    def update_validation(self, step, val_loss):
        val_ppl = safe_ppl(val_loss)
        print(f"[{step}] Validation loss: {val_loss:.4f}   ppl: {val_ppl:.2f}")
        
        if self.wandb_enabled:
            wandb.log({
                "step": step,
                "validation_loss": val_loss,
                "validation_perplexity": val_ppl
            }, step=step)

    def _rss(self):
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

    def _fmt(self, seconds):
        if seconds <= 0:
            return "--:--:--"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


