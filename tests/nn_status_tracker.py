import math
import time
import os
from typing import NamedTuple
import psutil
import wandb

from tests.nn_yaml import Config, load_yaml_config
from tests.upload_artifact import build_metadata, upload_artifact

def safe_ppl(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

def fmt_hms( seconds):
    if seconds <= 0:
        return "--:--:--"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

class LastUpdate(NamedTuple):
    step: int
    time: float
    loss: float
    min_window_loss: float
    tokens_processed: int
    throughput_tokens_per_second: float
    runtime_seconds: float
    eta_seconds: float
    lr: float
    grad_norm: float
    rss_gb: float

    def as_dict(self):
        return self._asdict()
    
    def render(self) -> str:
        return (
            f"[{self.step}] loss={self.loss:.4f} | min_loss={self.min_window_loss:.4f} | ppl={safe_ppl(self.loss):.2f}\n"
            f"      lr={self.lr:.8f} grad_norm={self.grad_norm:.4f}\n"
            f"      throughput={self.throughput_tokens_per_second:.1f} tokens/sec\n"
            f"      tokens_processed={self.tokens_processed:_}\n"
            f"      elapsed={fmt_hms(self.runtime_seconds)} eta={fmt_hms(self.eta_seconds)}\n"
            f"      rss={self.rss_gb:.2f}GB"
        )

class BestValidationLoss(NamedTuple):
    validation_loss: float
    step: int | None
    tokens_processed: int
    elapsed_seconds: float
    eval_time: float

    def render(self) -> str:
        return (
            f"[{self.step}] Validation Loss: {self.validation_loss:.4f} | "
            f"ppl: {safe_ppl(self.validation_loss):.2f} |"
            f"Tokens: {self.tokens_processed:_} | "
            f"elapsed_seconds: {self.elapsed_seconds} |"
            f"eval_time: {self.eval_time}"
        )

class StatusTracker:
    def __init__(self, tokens_processed, total_token_budget, total_steps, sched_as_dict, raw_cfg, config: Config, num_params):

        self.initial_tokens_processed = tokens_processed
        self.total_token_budget = total_token_budget
        self.avg_window = config.run.avg_window

        self.loss_history = []
        self.start_time = time.time()
        self.min_loss = float('inf')

        # everything about the best validation point, tracked together
        self.best_val = BestValidationLoss(float('inf'), None, 0, 0.0, 0.0)
        self.best_ckpt_path = None
 
        # last values seen by update(), for the final run summary
        self.last_update = None


        # each run is named according to the active name template
        active = config.naming.active
        templ = config.naming.templates[active]
        run_name = templ.format(config = config)

        if config.run.wandb_enabled:
            raw_cfg['schedule'] = sched_as_dict

            self.wandb = wandb.init(project=config.run.name, name=run_name, config=raw_cfg)
            for metric in ("stats.loss", "stats.perplexity"):
                self.wandb.define_metric(metric, summary="last")
                self.wandb.define_metric(metric, summary="mean")

            for metric in ['checkpoint', 'checkpoint_size_mb']:
                self.wandb.define_metric(metric, summary="none") 

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

        # Timing
        now = time.time()
        run_time  = now - self.start_time

        # Tokens processed   
        run_time = now - self.start_time     
        run_tokens = tokens_processed_lifetime - self.initial_tokens_processed      
        run_throughput = run_tokens / run_time

        if self.last_update is not None:
            window_seconds = now - self.last_update.time
            window_tokens = tokens_processed_lifetime - self.last_update.tokens_processed
            window_throughput = window_tokens / window_seconds if window_seconds > 0 else 0.0
        else:
            window_throughput = run_throughput
        # ETA
        
        remaining_tokens = (self.total_token_budget - tokens_processed_lifetime)
        eta_seconds = remaining_tokens / window_throughput

        # print(f"[{step}] loss={loss:.4f} | min_loss={self.min_loss:.4f} | ppl={perplexity:.2f} ")
        # print(f"      lr={lr:.8f} grad_norm={grad_norm:.4f}")
        # print(f"      throughput={run_throughput:.1f} tokens/sec" )
        # print(f"      tokens_processed={tokens_processed_lifetime:_} | tokens_remaining={remaining_tokens:_} | eta_seconds={eta_seconds:_}")
        # print(f"      elapsed={fmt_hms(run_time)} eta={fmt_hms(eta_seconds)}")
        # print(f"      rss={self._rss():.2f}GB")

        # remember for the final run summary / metadata and print periodic status
        self.last_update = LastUpdate(step=step, time=now, loss=loss, min_window_loss=self.min_loss,
                                      tokens_processed=tokens_processed_lifetime,throughput_tokens_per_second=window_throughput, 
                                      runtime_seconds=run_time, eta_seconds=eta_seconds, lr = lr, grad_norm=grad_norm, rss_gb=self._rss())
        upd_msg = self.last_update.render()
        print(upd_msg)
        # self.last_update = {
        #     'step': step,
        #     'loss': loss,
        #     'perplexity': perplexity,
        #     'tokens_processed': tokens_processed_lifetime,
        #     'throughput_tokens_per_second': run_throughput,
        #     'runtime_seconds': run_time,
        # }
       
        if self.wandb is not None:
            self.wandb.log({'stats': self.last_update.as_dict()}, step=step)

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

    def update_validation(self, step, val_loss, tokens_processed_lifetime, duration):
        # tracking best loss for reporting at end
        val_ppl = safe_ppl(val_loss)
        new_best = None
        if val_loss < self.best_val.validation_loss:
            new_best = val_loss
            elapsed = time.time() - self.start_time
            self.best_val = BestValidationLoss(val_loss, step, tokens_processed_lifetime, elapsed, duration)
 
        print(f"[{step}] Validation Loss: {val_loss:.4f} | Min val loss: {self.best_val.validation_loss:.4f} | Tokens: {tokens_processed_lifetime:_} | ppl: {val_ppl:.2f} | Eval time: {fmt_hms(duration)}")

        if self.wandb is not None:
            self.wandb.log({'validation': {
                "step": step,
                "validation_loss": val_loss,
                "tokens_processed":tokens_processed_lifetime,
                "validation_perplexity": val_ppl,
                "eval_time": duration
            }}, step=step)

        return new_best

    def finalize(self):
        """
        Report the best point, write it to the wandb summary as one dict, 
        and upload the BEST checkpoint.
        """
        best_loss_msg = 'No validation was run - no best validation loss to report.' if self.best_val.step == -1 else f'FINAL BEST LOSS: {self.best_val.render()}'
        self.log(best_loss_msg)

        if self.wandb is not None:
            self.wandb.summary.update({"best_validation_loss": self.best_val._asdict() })
            # At the end of training report time
            self.wandb.summary.update({'final_metrics': self.last_update.as_dict()}) # type: ignore
            # self.wandb.summary[" elapsed_hms"] = fmt_hms(self.last_update.runtime_seconds)

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
            metadata = build_metadata(self.wandb)
            upload_artifact(self.wandb, 'best_ckpt', ckpt_path, metadata=metadata, artifact_type='model')
            self.log('Upload complete.')

    def _rss(self):
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


if __name__ == '__main__':
    _, conf = load_yaml_config('tests/config/gpt2_tiny.yaml')
    active = conf.naming.active
    templ = conf.naming.templates[active]
    name = templ.format(config = conf)
    print(f'name: {name}')
    