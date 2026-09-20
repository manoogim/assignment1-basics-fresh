# Experiment Log

This document describes training attempts of a transformer model, trained on dataset tinystoriesGPT4.
Hardware: CPU and GPU
OS: Windows and Linux and Notebook
Tokens Budget: 327_680_000
* Input Dataset: tinystories_GPT4, consisting of bpe encodings of the text storead as tokens_train.bin, and tokens_valid.bin
* Training config: tests/config/css336.yaml
* Final weights: TBD 

## Infrastructure

### Windows

I wanted to train with batch_size=128, but my laptop RAM was not sufficient so I used batch_size=48. 
After 600 steps, Microsoft update arrived around 1am and rebooted my laptop, but I saved the run at [wandb](https://wandb.ai/manoogim-personal/cs336-basic-windows/runs/6eocbsd9/overview?nw=nwusermanoogim).

#### Code Refactoring

This crash induced me to switch training to a remote VM but I wanted to resume from the last checkpoint `ckpt_a.pt`.
For this so I refactored my checkpointing to allow batch size to be different when resuming from a checkpoint.

* Training loop stopping criteria was previously based on predermined number of steps. 
* But progress of N steps with batch=48 is not the same as progress of same number of steps with batch = 128: number of processed tokens is a better measure of progress
* Still, we need to keep track of steps count, because it is used as input to learning schedule, and as a checkpoint milestone.
* At the same time I made it so that the training stops when target loss is reached, in this case <= 2.0 (configurable)

### Linux

I rented a spot-based VM on Google Cloud, where it resumed from `step=600` in `ckpt_a.pt`, and trained slowly on CPU, with batch_size=128.
At step 1440 it stopped with reaching validation loss 1.975.
Code recorded final number of processed tokens at that step: 30_814_208, see logs [here](https://wandb.ai/manoogim-personal/cs336-basic-linux/runs/jwfstvnt/logs?nw=nwusermanoogim)
Since it didn't exhaust the full token budget we don't know what might happen after step 1440

```
@@@ Training is stopped at step: 1440, because validation loss reached target: 1.975 <= 2.0. Regular loss is 2.026. !!!

[1440] loss=2.0264 avg_loss(100)=2.1547 min_loss=1.9997
2026-09-10 02:41:08 ppl=7.59 avg_ppl=8.63
2026-09-10 02:41:08 lr=0.00029906 grad_norm=0.3854
2026-09-10 02:41:08 throughput=646.8 tokens/sec
2026-09-10 02:41:08 tokens_processed=30_814_208
2026-09-10 02:41:08 elapsed=06:28:25 eta=127:30:07
2026-09-10 02:41:08 rss=2.13GB
2026-09-10 02:41:10
[1440] Saved checkpoint: runs/cs336_tinystories/ckpt_d.pt (259.8MB)
2026-09-10 02:41:10 Training completed. Step count: 1440. Tokens processed: 30_814_208. Last loss: 2.026.
```
Conclusion: Did training on CPU achieve target validation loss of 2.0 with the budget of 40mm tokens?
The logs support that it strongly beat the target with about 30.8mm of tokens. We don't know if 1.45 would be reached b/c we didn't exhaust the tokens budget in this run.

### Kagle Notebook

On Kagle notebook I trained first with large token budget 360mm tokens and 40mm tokens.

## Large Budget
Config and summary log and charts at [this link](https://wandb.ai/manoogim-personal/cs336-basic-kagle/runs/atdpws22?nw=nwusermanoogim).
Number of steps was 10,000. Min loss of 1.56 was reached near the end at step 9900, with almost full token budget.
Warmup period was short and the curve kept plunging until step 700, then a brief phase of moderate downtrending while gently oscilating until step 3700, and the tail was oscilating slightly while staying under 1.6. Interestingly at the last step it trended up so the recorded loss at the end was 1.56. I am planning to enhance code such that it captures weights for the step where validation loss is smallest. 

This run achieved the 2.0 loss milestone, but didn't achieve 1.45 milestone.

## Small Budget (40mm), Large Batch (128)
Config and summary log at this [link](https://wandb.ai/manoogim-personal/cs336-basic-kagle-40mm/runs/yqi4535n/overview?nw=nwusermanoogim).
Number of steps was only 1220. Min of 2.20 was reached at step 920 and after that it oscilated in a narrow band, ending at 2.25. 
The warmup period was tiny and curve plunged for the first 100 steps, and until step 900 trended down moderately,  with occassional an gentle peak, and the tail was in a narrow band until the end.
The warmup phase so short that it wasn't even plotted. 

Since validation loss never got under 2.20, this run reached neither of the two target losses.

*Next Steps*

Remember my Windows + Linux stitched run, where I started with batch=48 on Windows and continued with batch=128 on Linux. 
In both cases number of steps was computed based on the large token budget and learning schedule was appropriated accordingly.
In that run loss target of 1.95 was achieved after only 30mm tokens processed, but for non-stitched large run it took more than 300mm tokens for loss to get under 2.0

## Small Budget, Small Batch (48)
Target loss under 2.0 was reached at 38mm tokens. But warmup period is still non-consequential, and the tail is over 70% of the loss curve.
Link [here](https://wandb.ai/manoogim-personal/cs336-basic-kagle-40mm-b48/runs/ff5v8ele/logs?nw=nwusermanoogim)

## Small Budget (40mm), Large Batch (128), More Warmup
I increased warmup phase thinking that maybe gradient was stuck in a bad area and didn't get a chance to slope down. While the val loss curve got smoother, it didn't get under 2.2
In conclusion, the warmup ablation was a dead end.
So far with small budget of tokens, large batch is not producing good result, regardless of the warmup. 
It progressed only for my stitched batch but that was with large budget of tokens (and consequently more steps).
Link [here](https://wandb.ai/manoogim-personal/cs336-basic-kagle-40mm-b48-warm/runs/3x6yiisb)
Next progression is to increase number of steps, but fbefore that let's first I want to do two more batch size ablations in between 48 and 128 to confirm that progress of val loss is also in between. 
Batch size=96 confirms the trend with recording smaller val loss , but still above 2.0 link [here](https://wandb.ai/manoogim-personal/cs336-basic-kagle-40mm-b96/runs/bv1bj4am?nw=nwusermanoogim)
Batch size=64 confirms trend, slightly above 2
Batch size = 130 runninf -> OOM
Batch size = 32 -> under 2 [link](https://wandb.ai/manoogim-personal/cs336-basic-kagle-40mm-b96/runs/4t1xssgv?nw=nwusermanoogim)
batch_size = 16 -> ~ 1.8 [link](https://wandb.ai/manoogim-personal/cs336-basic-kagle-40mm-b96/runs/c7o8d5r8?nw=nwusermanoogim)
batch size = 8 ?

Tracks (loss/ppl/lr/grad_norm/throughput/rss vs. step and wall-clock).

## Hyperparameter Search
| Run | lr | warmup_frac | batch_size | steps_to_target | val_loss | notes | wandb |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | [link] |

### Observations
- Prose: what patterns you saw, what you concluded, what you tried next and why.

## Ablations
### RoPE vs. no positional encoding
- Config diff, result, plot, conclusion.
### SwiGLU vs. ReLU FFN
- ...

## Final Model
- Chosen hyperparameters, full-budget run result, generated text samples.