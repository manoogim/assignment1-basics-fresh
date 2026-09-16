import math
import time
import os
import psutil
import wandb

from tests.nn_yaml import Config, load_yaml_config
from tests.upload_artifact import upload_artifact

def safe_ppl(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')
    
class StatusTracker:
    def __init__(self, tokens_processed, total_token_budget, total_steps, sched_as_dict, raw_cfg, config: Config, num_params):

        self.initial_tokens_processed = tokens_processed
        self.total_token_budget = total_token_budget
        self.avg_window = config.run.avg_window

        self.loss_history = []
        self.val_los_hist = {}
        self.start_time = time.time()
        self.min_loss = float('inf')
        self.min_val_loss = float('inf')

        # everything about the best validation point, tracked together
        self.best_val = {
            'validation_loss': None,
            'validation_perplexity': None,
            'step': None,
            'tokens_processed': None,
            'elapsed_seconds': None,
        }
        self.best_ckpt_path = None
 
        # last values seen by update(), for the final run summary
        self.last_update = {}


        # each run is named according to the active name template
        active = config.naming.active
        templ = config.naming.templates[active]
        run_name = templ.format(config = config)

        if config.run.wandb_enabled:
            self.wandb = wandb.init(project=config.run.name, name=run_name, config=raw_cfg)
            for metric in ("loss", "perplexity"):
                self.wandb.define_metric(metric, summary="last")
                self.wandb.define_metric(metric, summary="mean")

            for key in ['checkpoint', 'checkpoint_size_mb']:
                self.wandb.define_metric(key, summary="none") 

            self.wandb.summary.update ({'init': {"token_budget": total_token_budget, "total_steps": total_steps,"previous_tokens": tokens_processed, 'num_params': num_params}})
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

        # Compute window averages
        window_min_loss  = min(self.loss_history)
        if window_min_loss  < self.min_loss:
            self.min_loss = window_min_loss 

        # Compute perplexity
        perplexity = safe_ppl(loss)

        # Timing
        now = time.time()
        run_time  = now - self.start_time

        # Tokens processed   
        run_time = now - self.start_time     
        run_tokens = tokens_processed_lifetime - self.initial_tokens_processed      
        run_throughput = run_tokens / run_time

        # ETA
        
        remaining_tokens = (self.total_token_budget - tokens_processed_lifetime)
        eta_seconds = remaining_tokens / run_throughput

        # Print periodic status

        print(f"[{step}] loss={loss:.4f} | min_loss={self.min_loss:.4f} | ppl={perplexity:.2f} ")
        print(f"      lr={lr:.8f} grad_norm={grad_norm:.4f}")
        print(f"      throughput={run_throughput:.1f} tokens/sec")
        print(f"      tokens_processed={tokens_processed_lifetime:_}")
        print(f"      elapsed={self._fmt(run_time)} eta={self._fmt(eta_seconds)}")
        print(f"      rss={self._rss():.2f}GB")

        # remember for the final run summary / metadata
        self.last_update = {
            'step': step,
            'loss': loss,
            'perplexity': perplexity,
            'tokens_processed': tokens_processed_lifetime,
            'throughput_tokens_per_second': run_throughput,
            'runtime_seconds': run_time,
        }
       
        if self.wandb is not None:
            self.wandb.log({'stats': {
                "loss": loss,
                "perplexity": perplexity,
                "lr": lr,
                "grad_norm": grad_norm,
                "throughput": run_throughput,
                "rss": self._rss(),
                "tokens_processed": tokens_processed_lifetime,
                "elapsed": run_time
            }}, step=step)

    def update_checkpoint(self, step, ckpt_path, is_best=False):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        label = 'BEST checkpoint' if is_best else 'checkpoint'
        print(f"[{step}] Saved {label}: {ckpt_path} ({size_mb:.1f}MB)")

        if is_best:
            self.best_ckpt_path = ckpt_path
        if self.wandb is not None:

            self.wandb.log({
                "checkpoint": ckpt_path,
                "checkpoint_size_mb": size_mb
            }, step=step)

    def update_validation(self, step, val_loss, tokens_processed_lifetime):
        # tracking best loss for reporting at end
        val_ppl = safe_ppl(val_loss)
        new_best = None
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            new_best = val_loss
            elapsed = time.time() - self.start_time
            self.best_val = {
                'validation_loss': val_loss,
                'validation_perplexity': val_ppl,
                'step': step,
                'tokens_processed': tokens_processed_lifetime,
                'elapsed_seconds': elapsed,
                'elapsed_hms': self._fmt(elapsed)
            }
 
        print(f"[{step}] Validation Loss: {val_loss:.4f} | Min val loss: {self.min_val_loss:.4f} | Tokens: {tokens_processed_lifetime:_} | ppl: {val_ppl:.2f}")

        if self.wandb is not None:
            self.wandb.log({'validation': {
                "step": step,
                "validation_loss": val_loss,
                "tokens_processed":tokens_processed_lifetime,
                "validation_perplexity": val_ppl
            }}, step=step)

        return new_best

    def finalize(self):
        """
        Report the best point, write it to the wandb summary as one dict, 
        and upload the BEST checkpoint.
        """
        self.report_best_loss()

        if self.wandb is not None:
            self.wandb.summary.update({'best_loss': self.best_val}) 
            
            # At the end of training report time
            self.wandb.summary["stats.elapsed_hms"] = self._fmt(self.last_update['runtime_seconds'])

            if self.best_ckpt_path is None:
                self.log('No best checkpoint was saved during this run - skipping artifact upload.')

            elif not os.path.exists(self.best_ckpt_path):
                self.log(f'Best checkpoint path missing on disk, skipping upload: {self.best_ckpt_path}')

            else:                
                self.upload_best_artifact(self.best_ckpt_path)

    def upload_best_artifact(self, ckpt_path):      
        if self.wandb is not None:
            self.log(f'Start uploading best weights to wandb.')
            # add the actual file to wandb, and attach final summary to metadata
            metadata = {key: value for key, value in self.wandb.summary._as_dict().items() if not key.startswith("_") }
            upload_artifact(self.wandb, 'best_ckpt', ckpt_path, metadata=metadata, artifact_type='model')
            self.log('Upload complete.')

    def report_best_loss(self):
        if self.best_val['step'] is None:
            self.log('No validation was run - no best validation loss to report.')
        else:
            self.log(f"Min val loss: {self.best_val['validation_loss']:.6f} "
                 f"at step {self.best_val['step']}, "
                 f"tokens {self.best_val['tokens_processed']:_}, "
                 f"elapsed {self._fmt(self.best_val['elapsed_seconds'])} "
                 f"| Min train loss: {self.min_loss:.6f}")
        return self.best_val['step']

    def _rss(self):
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

    def _fmt(self, seconds):
        if seconds <= 0:
            return "--:--:--"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


if __name__ == '__main__':
    _, conf = load_yaml_config('tests/config/gpt2_tiny.yaml')
    active = conf.naming.active
    templ = conf.naming.templates[active]
    name = templ.format(config = conf)
    print(f'name: {name}')
    