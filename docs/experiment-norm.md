# RMS Norm Experiments
What happens when we remove all RMS norms?
Can we still train with the baseline learning rate?
Can we get stability with smaller learning rate?

*Baseline setup*
```
Steps: 1,220
Tokens budget: 40,000,000
Batch: 128
peak_lr = 0.0055
Norm: RMS (pre-layer)
Embedding: rope
```

I conducted tests with 3 learning rates. For large lr=0.0055, gradients exploded soon after start, and all numbers were Nan.
For medium lr=0.0020, there was a spike, but recovered, and a reasonable validation loss resulted, but it was worse than the baseline.
For small lr=0.0005 there were no spikes, training completed, validation loss was reasonable but worse than baseline.

When lr is large, it means that training is taking large steps, causing weights and logits to become larger.
The effect compounds with each layer and eventually overflow can occur.

## Ablation: large peak_lr = 0.0055, without norm

Around 100 steps, this run started exploding and became completely non-finite by step 140
| ![Gradient explosion when lr is large, and there is no normalization.](nangrad.svg) | ![Loss cannot be computed when lr is large and there is no norm.](nanloss.svg) |
|-----------------|-----------------|


## Ablation: medium peak_lr = 0.0020, without norm

Training completed, but there was a spike and it recovered.

| Step | LR        | Gradient norm | Validation loss |
|------|-----------|---------------|------------------|
| 80   | ~0.00305  | 1.125         | 2.998            |
| 100  | 0.003805  | 448.0         | 3.172            |
| 120  | 0.004558  | 5.05×10¹⁸     | 1.69×10²⁰        |
| 140  | 0.005312  | NaN           | NaN              |

| ![Validation Loss](loss-no-norm.svg) | ![](grad-no-norm.svg) |
|---------------|---------------|
| ![](val-loss-no-norm.svg) | ![](img4.svg) |



| Step | LR        | Gradient norm | Validation loss |
|------|-----------|---------------|------------------|
| 320  | 0.001885  | 0.704         | 2.238            |
| 340  | 0.001858  | 0.673         | 2.268            |
| 360  | 0.001828  | 11,841.37     | 2.289            |
| 380  | 0.001796  | 1.239         | 2.199            |
| 400  | 0.001761  | 0.826         | 2.224            |
| 420  | 0.001724  | 0.546         | 2.191            |


## Ablation: small peak_lr = 0.00005, without norm
Completed.

## Final Summary

| Run           | Description | Norm | Peak LR | Final train loss | Mean train loss | Final val loss ↓ | Final grad norm |
|---------------|-------------|------|---------|------------------|------------------|-------------------|------------------|
| rms_lr0.0055  | Baseline            | rms  | 0.0055  | 1.6240           | 2.0891           | 1.6133            | 0.1167           |
| none_lr0.0055 | Diverged             | none | 0.0055  | NaN              | NaN              | NaN               | NaN              |
| none_lr0.0005 | Completed            | none | 0.0005  | 1.9025           | 2.4317           | 1.8752            | 0.6985           |
| none_lr0.002  | Spiked, but recovered            | none | 0.002   | 1.7109           | 2.5293           | 1.6915            | 0.3378           |

## Ablation: post-layer norm vs pre-layer norm
The original Transformer paper used post-norm (normalize results of attention, and results of ffn), and we are currently using pre-norm (normalize inputs to attention, and inputs to ffn). Let's see what happens when we use pre-norm.

First run result was for peak_lr=0.0055. Both best loss and final loss are both better with pre-norm approach. The loss curve in pre-norm (original, blue) stays above the loss curve in post-norm (orange), except for a short period at the beginning.

![Current approch vs Original Transformer approach](post-norm.svg)

Then I also tried for smaller learning rates: 0.002 and 0.0005, which shows mostly equal or post-layer slightly better.


| Peak LR | Pre-norm val loss | Post-norm val loss | Pairwise winner | Relative difference |
|---------|--------------------|---------------------|------------------|----------------------|
| 0.0005  | 1.9487             | 1.9293              | Post-norm        | 1.00% lower          |
| 0.0020  | 1.6534             | 1.6482              | Post-norm        | 0.31% lower          |
| 0.0055  | 1.6133             | 1.6740              | Pre-norm         | 3.77% lower          |


# Embedings Experiments #
We will next investigate the impact of the position embeddings on the performance of the model. Specifically, we will compare our base model (with RoPE) with not including position embeddings at all (NoPE).

| Learning rate | RoPE loss | No-position loss | RoPE advantage |
|---------------|-----------|------------------|----------------|
| 0.0005        | 1.9487    | 2.0954           | 7.00% lower    |
| 0.0020        | 1.6534    | 1.7316           | 4.51% lower    |
| 0.0055        | 1.6133    | 1.7220           | 6.31% lower    |


RoPE clearly outperforms no positional embedding at all three learning rates.