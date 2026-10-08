import time
import os
from typing import NamedTuple
import psutil
import torch
import wandb

from tests.nn_utils import calc_eta, compute_safe_ppl, fmt_hms
from tests.nn_yaml import Config, load_yaml_config
from tests.upload_artifact import build_metadata, upload_artifact

class LastUpdate(NamedTuple):
    step: int
    time: float
    loss: float
    min_window_loss: float
    tokens_processed: int
    throughput_tokens_per_second: float
    runtime_seconds: float
    eta_seconds: float
    eta_seconds_dbg: float | None
    lr: float
    grad_norm: float
    rss_gb: float
    load_time: float

    def as_dict(self):
        return self._asdict()
    """
    wandb.define_metric("optim_step")
wandb.define_metric("*", step_metric="optim_step")
wandb.log({"optim_step": step, "train/loss": loss, "val/loss": val, "tokens_seen": tokens}, commit=True)
    """
    def render(self) -> str:
        load_fraction_time = self.load_time / self.runtime_seconds if self.runtime_seconds > 0 else 0.0
        eta_dbg = f'eta_dbg={fmt_hms(self.eta_seconds_dbg)}' if self.eta_seconds_dbg is not None else ''
        return (
            f"[{self.step}] loss={self.loss:.4f} | min_loss={self.min_window_loss:.4f} | ppl={compute_safe_ppl(self.loss):.2f}\n"
            f"      lr={self.lr:.8f} grad_norm={self.grad_norm:.4f}\n"
            f"      throughput={self.throughput_tokens_per_second:.1f} tokens/sec\n"
            f"      tokens_processed={self.tokens_processed:_}\n"
            f"      elapsed={fmt_hms(self.runtime_seconds)} | eta={fmt_hms(self.eta_seconds)} | {eta_dbg} \n"
            f"      rss={self.rss_gb:.2f}GB | load_wait_time={self.load_time:.2f}s | load_time_fraction={load_fraction_time:.2%}"
        )

class BestValidationLoss(NamedTuple):
    validation_loss: float
    val_ppl: float
    step: int | None
    tokens_processed: int
    elapsed_seconds: float

    def render(self) -> str:
        return (
            f"[{self.step}] Validation Loss: {self.validation_loss:.4f} | "
            f"ppl: {self.val_ppl:.2f} |"
            f"Tokens: {self.tokens_processed:_} | "
            f"elapsed_seconds: {self.elapsed_seconds} |"
        )

class StatusTracker:
    def __init__(self, tokens_processed, total_steps, sched_as_dict, raw_cfg, config: Config, num_params):

        self.initial_tokens_processed = tokens_processed
        self.total_token_budget = config.token_budget
        self.avg_window = config.run.avg_window

        self.loss_history = []
        self.start_time = time.time()
        self.min_loss = float('inf')

        # everything about the best validation point, tracked together
        self.best_val = BestValidationLoss(float('inf'), float('inf'), None, 0, 0.0)
        self.best_ckpt_path = None
        self.eval_time = 0.0
 
        # last values seen by update(), for the final run summary
        self.last_update = None

        msg=f"Total steps: {total_steps:_}, Total tokens budget: {config.token_budget:_}, effective batch size: {config.train.batch_size}, grad_accum: {config.train.grad_accum}, runs folder: {config.dict['runs_folder']} "
        StatusTracker.log(msg)
        if config.run.num_steps_dbg is not None:
            self.dbg_token_budget = config.token_budget * config.run.num_steps_dbg // total_steps
            msg = f'DBG total steps: {config.run.num_steps_dbg} | DBG tokens budget: {self.dbg_token_budget:_}'
            StatusTracker.log(msg)
        else:
            self.dbg_token_budget = None

        if config.run.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        if config.wandb.enabled:
            raw_cfg['schedule'] = sched_as_dict

            # each run is named according to the active name template
            active = config.wandb.active
            templ = config.wandb.name_templates[active]
            run_name = templ.format(config = config)
            self.log(f'Wandb project: {config.wandb.proj_name} | Wandb name: {run_name}')

            self.wandb = wandb.init(project=config.wandb.proj_name, name=run_name, config=raw_cfg, tags=config.wandb.tags)
            for metric in ("stats.loss", "stats.perplexity"):
                self.wandb.define_metric(metric, summary="last")
                self.wandb.define_metric(metric, summary="mean")

            for metric in ['checkpoint', 'checkpoint_size_mb', 'best_validation_loss.eval_time']:
                self.wandb.define_metric(metric, summary="none") 

            self.wandb.summary.update ({'init': {"token_budget": config.token_budget, "total_steps": total_steps,"previous_tokens": tokens_processed, 'num_params': num_params}})
        else:
            self.wandb = None

    @classmethod
    def log(cls, msg):
        print(f'@@@ {msg} !!!')

    def update(self, step, loss, lr, grad_norm, tokens_processed_lifetime, load_time):
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

        if self.last_update is not None and self.last_update.step == step:
            # edge case when final step coincides with the last loop pass
            window_throughput = self.last_update.throughput_tokens_per_second
        elif self.last_update is not None:
            window_seconds = now - self.last_update.time
            window_tokens = tokens_processed_lifetime - self.last_update.tokens_processed
            window_throughput = window_tokens / window_seconds if window_seconds > 0 else 0.0
        else:
            window_throughput = run_throughput
        # ETA
        eta_seconds = calc_eta(self.total_token_budget, tokens_processed_lifetime, window_throughput)
        tokens_processed_this_run = tokens_processed_lifetime - self.initial_tokens_processed
        eta_seconds_dbg = calc_eta(self.dbg_token_budget, tokens_processed_this_run, window_throughput) if self.dbg_token_budget is not None else None
 
        # remember for the final run summary / metadata and print periodic status
        self.last_update = LastUpdate(step=step, time=now, loss=loss, min_window_loss=self.min_loss,
                                      tokens_processed=tokens_processed_lifetime,throughput_tokens_per_second=window_throughput, 
                                      runtime_seconds=run_time, eta_seconds=eta_seconds, eta_seconds_dbg=eta_seconds_dbg, 
                                      lr = lr, grad_norm=grad_norm, rss_gb=self._rss(), load_time=load_time)
        upd_msg = self.last_update.render()
        print(upd_msg)
       
        if self.wandb is not None:
            self.wandb.log({'stats': self.last_update.as_dict()}, step=step)
    
    def update_checkpoint(self, step, ckpt_path, kind):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        # label = 'BEST checkpoint' if is_best else 'checkpoint'
        print(f"[{step}] Saved {kind.upper()} checkpoint: {ckpt_path} ({size_mb:.1f}MB)")
        is_best = kind.lower().startswith('best')
        if is_best:
            self.best_ckpt_path = ckpt_path

    def update_validation(self, step, val_result, tokens_processed_lifetime, duration):
        val_loss, val_ppl = val_result['val_loss'], val_result['val_ppl']
        # tracking best loss for reporting at end
        new_best = None
        self.eval_time += duration
        if val_loss < self.best_val.validation_loss:
            new_best = val_loss
            elapsed = time.time() - self.start_time
            self.best_val = BestValidationLoss(val_loss, val_ppl, step, tokens_processed_lifetime, elapsed)
 
        print(f"[{step}] Validation Loss: {val_loss:.4f} | Min val loss: {self.best_val.validation_loss:.4f} | Tokens: {tokens_processed_lifetime:_} | ppl: {val_ppl:.2f} | Eval time: {fmt_hms(self.eval_time)}")

        if self.wandb is not None:
            self.wandb.log({'validation': {
                "step": step,
                "validation_loss": val_loss,
                "tokens_processed":tokens_processed_lifetime,
                "validation_perplexity": val_ppl,
                "eval_time": self.eval_time
            }}, step=step)

        return new_best

    def finalize(self, last_ckpt_path = None):
        try:
            self._finalize(last_ckpt_path)
        except Exception as ex:
            print(ex)
        finally:
            pass

    def _finalize(self, last_ckpt_path = None):
        """
        Report the best point, write it to the wandb summary as one dict, and upload the BEST checkpoint.
        Optionally, upload the last chkpt
        """
        best_loss_msg = 'No validation was run - no best validation loss to report.' if self.best_val.step == -1 else f'FINAL BEST LOSS: {self.best_val.render()}'
        self.log(best_loss_msg)

        if self.wandb is not None:
            summary = {
                'best_validation_loss': self.best_val._asdict() ,
                'final_metrics': self.last_update.as_dict(), # type: ignore
                'cuda_metrics': {
                    "gpu/max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "gpu/max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
                    } ,
            }
            self.wandb.summary.update(summary)

            if self.best_ckpt_path is None:
                self.log('No best checkpoint was saved during this run - skipping artifact upload.')

            elif not os.path.exists(self.best_ckpt_path):
                self.log(f'Best checkpoint path missing on disk, skipping upload: {self.best_ckpt_path}')

            else:                
                self.upload_best_artifact(self.best_ckpt_path)
            
            if last_ckpt_path is not None:
                self.log('Uploading last checkpoint')
                self.upload_milestone_artifact(last_ckpt_path, self.last_update.step, 'last') # type: ignore

    def upload_best_artifact(self, ckpt_path):      
        if self.wandb is not None:
            self.log(f'Start uploading best weights to wandb.')
            # add the actual file to wandb, and attach final summary to metadata
            metadata = build_metadata(self.wandb)
            upload_artifact(self.wandb, 'best_ckpt', ckpt_path, metadata=metadata, artifact_type='model')
            self.log('Upload complete.')

    def upload_milestone_artifact(self, ckpt_path, step, tag):
        if self.wandb is not None:
            self.log(f'Start uploading {tag} weights to wandb.')
            metadata = {
                'kind': tag,
                'step': step,
            }
            artifact_name = f'{tag}_{step}'
            upload_artifact(self.wandb, artifact_name, ckpt_path, metadata, 'model')
            self.log(f'Uploaded {tag} at step {step}')

    def _rss(self):
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


if __name__ == '__main__':
    _, conf = load_yaml_config('tests/config/gpt2_tiny.yaml', 0.0001)
    active = conf.wandb.active
    templ = conf.wandb.name_templates[active]
    name = templ.format(config = conf)
    print(f'name: {name}')
    