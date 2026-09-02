import time
import os
import psutil
import torch

class StatusTracker:
    def __init__(self, total_steps, print_every=100, avg_window=100):
        self.total_steps = total_steps
        self.print_every = print_every
        self.avg_window = avg_window

        self.loss_history = []
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = 0

    def update(self, step, loss, lr, grad_norm, batch_tokens, ckpt_path):
        # Track loss
        self.loss_history.append(loss)
        if len(self.loss_history) > self.avg_window:
            self.loss_history.pop(0)

        # Compute averages
        avg_loss = sum(self.loss_history) / len(self.loss_history)
        min_loss = min(self.loss_history)

        # Timing
        now = time.time()
        step_time = now - self.last_time
        self.last_time = now

        throughput = batch_tokens / step_time

        # ETA
        elapsed = now - self.start_time
        steps_done = step + 1
        steps_per_sec = steps_done / elapsed
        eta_seconds = (self.total_steps - steps_done) / steps_per_sec if hasattr(self, "total_steps") else 0

        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        # Print periodic status
        if step % self.print_every == 0:
            print(f"[{step}] loss={loss:.4f} avg_loss({self.avg_window})={avg_loss:.4f} min_loss={min_loss:.4f}")
            print(f"      lr={lr:.6f} grad_norm={grad_norm:.4f}")
            print(f"      throughput={throughput:.1f} tokens/sec")
            print(f"      elapsed={self._fmt(elapsed)} eta={self._fmt(eta_seconds)}")
            print(f"      rss={self._rss():.2f}GB")
            print(f"[{step}] Saved checkpoint: {ckpt_path} ({size_mb:.1f}MB)")
        

    def _rss(self):
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

    def _fmt(self, seconds):
        if seconds <= 0:
            return "--:--:--"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
