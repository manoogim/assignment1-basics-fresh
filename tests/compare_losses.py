import numpy as np

import numpy as np


def compare_paired(candidate_losses, baseline_losses):
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    baseline = np.asarray(baseline_losses, dtype=np.float64)
    print(f'mean of candidate_losses: {candidate_losses.mean()}')
    print(f'mean of baseline_losses: {baseline_losses.mean()}')
    print(f'diff of means: {(baseline_losses - candidate_losses).mean()}')

    if candidate.shape != baseline.shape:
        raise ValueError("Paired arrays must have the same shape")

    if candidate.ndim != 1:
        raise ValueError("Expected one-dimensional arrays")

    if len(candidate) < 2:
        raise ValueError("At least two paired losses are required")

    # Positive means candidate loss is lower than baseline loss.
    improvements = baseline - candidate

    nn = len(improvements)
    mean_improvement = improvements.mean()
    paired_std = improvements.std(ddof=1)
    paired_se = paired_std / np.sqrt(nn)

    # Normal approximation; appropriate for n=100.
    margin = 1.96 * paired_se

    return {
        "num_pairs": nn,
        "candidate_mean": candidate.mean(),
        "baseline_mean": baseline.mean(),
        "mean_improvement": mean_improvement,
        "paired_std": paired_std,
        "paired_se": paired_se,
        "ci95_low": mean_improvement - margin,
        "ci95_high": mean_improvement + margin,
    }


candidate_path = 'artifacts/last_1699-v0/batch_losses_50.npy' 
baseline_path =  'artifacts/last_1349-v0/batch_losses_50.npy' 

result = compare_paired(candidate_losses=np.load(candidate_path), baseline_losses=np.load(baseline_path))
print(f'*** Paired comparison between candidate: {candidate_path} and baseline: {baseline_path} ***')
for key, value in result.items():
    print(f"{key}: {value}")