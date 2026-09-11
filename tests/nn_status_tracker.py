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
    def __init__(self, tokens_processed, total_token_budget, total_steps, sched_as_dict, raw_cfg, config: Config):
        self.initial_tokens_processed = tokens_processed
        self.total_token_budget = total_token_budget
        self.avg_window = config.run.avg_window

        self.loss_history = []
        self.start_time = time.time()

        if config.run.wandb_enabled:
            run_name = f'{config.run.name}_{config.train.batch_size}'

            # this helps to log actual lengths of each sched phase
            raw_cfg['schedule'] = sched_as_dict

            self.wandb = wandb.init(project=config.run.name, name=run_name, config=raw_cfg)
            self.wandb.summary.update ({"token_budget": total_token_budget, "total_steps": total_steps,"previous_tokens": tokens_processed})
        else:
            self.wandb = None

    @classmethod
    def log(cls, msg):
        print(f'@@@ {msg} !!!')

    def update(self, step, loss, lr, grad_norm, tokens_processed_lifetime):
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
        elapsed = now - self.start_time

        # Tokens processed   
        run_time = now - self.start_time     
        run_tokens = tokens_processed_lifetime - self.initial_tokens_processed      
        run_throughput = run_tokens / run_time

        # ETA
        
        remaining_tokens = (self.total_token_budget - tokens_processed_lifetime)
        eta_seconds = remaining_tokens / run_throughput

        # Print periodic status

        print(f"[{step}] loss={loss:.4f} avg_loss({self.avg_window})={avg_loss:.4f} min_loss={min_loss:.4f}")
        print(f"      ppl={perplexity:.2f} avg_ppl={avg_perplexity:.2f}")
        print(f"      lr={lr:.8f} grad_norm={grad_norm:.4f}")
        print(f"      throughput={run_throughput:.1f} tokens/sec")
        print(f"      tokens_processed={tokens_processed_lifetime:_}")
        print(f"      elapsed={self._fmt(run_time)} eta={self._fmt(eta_seconds)}")
        print(f"      rss={self._rss():.2f}GB")
       
        if self.wandb is not None:
            self.wandb.log({
                "loss": loss,
                "avg_loss": avg_loss,
                "perplexity": perplexity,
                "avg_perplexity": avg_perplexity,
                "lr": lr,
                "grad_norm": grad_norm,
                "throughput": run_throughput,
                "rss": self._rss(),
                "step": step,
                "tokens_processed": tokens_processed_lifetime
            }, step=step)

        
    def update_checkpoint(self, step, ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"[{step}] Saved checkpoint: {ckpt_path} ({size_mb:.1f}MB)")
        if self.wandb is not None:
            self.wandb.log({
                "step": step,
                "checkpoint": ckpt_path,
                "checkpoint_size_mb": size_mb
            }, step=step)

    def update_validation(self, step, val_loss):
        val_ppl = safe_ppl(val_loss)
        print(f"[{step}] Validation loss: {val_loss:.4f}   ppl: {val_ppl:.2f}")
        
        if self.wandb is not None:
            self.wandb.log({
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
    
    def upload_ckpt(self, ckpt_path):

        if self.wandb is not None:
            self.log(f'Start uploading last weights to wandb.')
            # add the actual file to wandb
            artifact_name = 'last_ckpt'   # → "ckpt_a_off.pt"
            metadata={'path': ckpt_path}
        
            artifact = wandb.Artifact(name=artifact_name, type = 'model', metadata=metadata)
            artifact.add_file(ckpt_path, name=os.path.basename(ckpt_path))
            self.wandb.log_artifact(artifact)
            try:
                artifact.wait() # wait without short timeout
                self.log('Completed upload.')
            except Exception as ex:
                self.log(f'Error while waiting to upload weights to wandb. {ex}')
            finally:
                self.log('BYE')
                self.wandb.finish()


