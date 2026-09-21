"""
loss_curve_noise_analysis.py

Analyze noise in validation-loss sweeps exported from Weights & Biases (or any
CSV with a "Step" column plus one value column per run).

What it computes, for a set of runs you believe are otherwise-identical
(e.g. same peak_lr, different seed) or comparable (e.g. an LR sweep):

  1. Tail statistics per run: mean/std of validation loss once training has
     flattened out (e.g. after the cosine anneal reaches its floor).
  2. Seed-to-seed noise: the log-ratio standard deviation between pairs of
     runs, which tells you the % noise band to compare against real gaps.
  3. Rank-crossing check: whether curves ever cross (useful for confirming
     a clean LR sweep, or spotting a run that's an outlier).
  4. Threshold analysis: given a target loss, how many standard deviations
     away is the current tail mean, the (normal-approx) probability a single
     checkpoint dips below it, and how many independent checkpoints you'd
     need before that becomes likely -- i.e. whether hitting the target is
     plausible from noise/luck alone or requires a real improvement in the
     underlying mean.

Usage (CLI):
    python loss_curve_noise_analysis.py run.csv --tail-start-step 1700 --threshold 1.45

Usage (as a library):
    from loss_curve_noise_analysis import load_wandb_csv, tail_stats, seed_noise, threshold_report
    runs = load_wandb_csv("run.csv")
    stats = {name: tail_stats(steps, vals, tail_start_step=1700) for name, (steps, vals) in runs.items()}
"""

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import norm
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_wandb_csv(path):
    """
    Load a W&B CSV export using only the stdlib csv module (no pandas).
    Keeps only the main value column per run (drops the __MIN / __MAX
    shadow columns W&B adds when a run is grouped).

    Returns: dict of {run_name: (steps: np.ndarray, values: np.ndarray)}
    Rows with a missing/blank/non-numeric value in a given run's column are
    dropped for that run only (matches pandas' dropna behavior).
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "Step" not in fieldnames:
            raise ValueError("Expected a 'Step' column in the CSV.")

        value_cols = [c for c in fieldnames
                      if c != "Step" and not (c.endswith("__MIN") or c.endswith("__MAX"))]

        raw = {col: {"steps": [], "values": []} for col in value_cols}
        for row in reader:
            step_raw = row.get("Step", "")
            if step_raw in (None, ""):
                continue
            try:
                step = float(step_raw)
            except ValueError:
                continue
            for col in value_cols:
                v_raw = row.get(col, "")
                if v_raw in (None, ""):
                    continue
                try:
                    v = float(v_raw)
                except ValueError:
                    continue
                raw[col]["steps"].append(step)
                raw[col]["values"].append(v)

    runs = {}
    for col in value_cols:
        name = col.split(" - ")[0].strip()
        runs[name] = (np.array(raw[col]["steps"], dtype=float),
                      np.array(raw[col]["values"], dtype=float))
    return runs


# --------------------------------------------------------------------------
# Tail statistics (the "settled" region of training you want to compare on)
# --------------------------------------------------------------------------

@dataclass
class TailStats:
    name: str
    n_points: int
    mean: float
    std: float
    min: float
    min_step: float
    fallback_used: bool = False   # True if tail_start_step couldn't be honored
    single_point: bool = False    # True if only 1 point available (std undefined)

    def __str__(self):
        flag = ""
        if self.single_point:
            flag = "  [SINGLE POINT -- no std, treat as unreliable]"
        elif self.fallback_used:
            flag = "  [fallback: tail_start_step unreachable for this run]"
        std_str = f"{self.std:.4f}" if self.std is not None else "  n/a "
        return (f"{self.name:20s} n={self.n_points:3d}  mean={self.mean:.4f}  "
                f"std={std_str}  min={self.min:.4f} (step {self.min_step:.0f}){flag}")


def tail_stats(steps, values, tail_start_step=None, tail_frac=0.15, name="run",
                min_tail_points=5):
    """
    Compute mean/std/min over the flat tail of a run.

    tail_start_step: absolute step to start the tail window at (e.g. where
        your schedule reaches its floor). Takes precedence if given.
    tail_frac: fallback -- use the last `tail_frac` fraction of logged points
        if tail_start_step is not given, OR if tail_start_step leaves too few
        points for this particular run (e.g. the run was logged less
        frequently, stopped early, or the floor was reached right at the
        last point / after logging stopped).
    min_tail_points: minimum points needed to trust a std estimate. If fewer
        than this are available even after falling back, the run is still
        reported (so it isn't silently dropped from a multi-run comparison)
        but flagged so you don't treat its std as meaningful.

    This never raises for "too few points" -- some runs legitimately have
    zero or one logged step after their schedule flattens (e.g. anneal floor
    reached right at the end of logging, or a shorter run). Instead it falls
    back to the last `tail_frac` of points, and flags the result so you can
    see which runs' stats are trustworthy vs. estimated from very little data.
    """
    steps = np.asarray(steps)
    values = np.asarray(values)
    fallback_used = False

    if tail_start_step is not None:
        mask = steps >= tail_start_step
        if mask.sum() < min_tail_points:
            fallback_used = True
            cutoff_idx = max(0, int(len(steps) * (1 - tail_frac)))
            mask = np.arange(len(steps)) >= cutoff_idx
    else:
        cutoff_idx = max(0, int(len(steps) * (1 - tail_frac)))
        mask = np.arange(len(steps)) >= cutoff_idx

    # Absolute last resort: even the fraction-based window is empty/too small
    # (e.g. a very short run) -- just take whatever points exist, down to 1.
    if mask.sum() == 0:
        mask = np.zeros(len(steps), dtype=bool)
        mask[-1:] = True
        fallback_used = True

    v = values[mask]
    s = steps[mask]
    n = int(mask.sum())
    single_point = n < 2

    return TailStats(
        name=name,
        n_points=n,
        mean=float(v.mean()),
        std=(float(v.std(ddof=1)) if n >= 2 else None), # type: ignore
        min=float(v.min()),
        min_step=float(s[np.argmin(v)]),
        fallback_used=fallback_used,
        single_point=single_point,
    )


# --------------------------------------------------------------------------
# Seed-to-seed / run-to-run noise
# --------------------------------------------------------------------------

def seed_noise(steps_a, values_a, steps_b, values_b, min_step=None, min_points=3,
                tail_frac_fallback=0.15):
    """
    Estimate the noise band between two runs meant to be comparable
    (e.g. same config, different seed), as the std of the log-ratio
    log(a/b) at matching steps once both have settled.

    Returns (mean_log_ratio, std_log_ratio, n_points_used, fallback_used).
    std_log_ratio is ~ the fractional (%) noise you should expect between
    two "identical" runs.

    If min_step leaves fewer than min_points overlapping steps (e.g. the
    anneal floor is reached right at/after the last logged point for one of
    the runs), this falls back to the last tail_frac_fallback of the
    overlapping steps instead of raising, and flags that it did so.
    """
    steps_a, values_a = np.asarray(steps_a), np.asarray(values_a)
    steps_b, values_b = np.asarray(steps_b), np.asarray(values_b)

    common = np.intersect1d(steps_a, steps_b)
    fallback_used = False

    if min_step is not None:
        restricted = common[common >= min_step]
        if len(restricted) < min_points:
            fallback_used = True
        else:
            common = restricted

    if fallback_used or (min_step is None and len(common) >= min_points):
        pass  # common already set appropriately above when no fallback needed

    if len(common) < min_points:
        # last resort: take the tail fraction of whatever overlap exists at all
        cutoff = max(0, int(len(common) * (1 - tail_frac_fallback)))
        common = common[cutoff:] if len(common) > 0 else common
        fallback_used = True

    if len(common) < 2:
        return None, None, len(common), True  # not enough data even after fallback

    a = np.array([values_a[steps_a == s][0] for s in common])
    b = np.array([values_b[steps_b == s][0] for s in common])
    log_ratio = np.log(a / b)
    return float(log_ratio.mean()), float(log_ratio.std(ddof=1)), len(common), fallback_used


def crossing_check(runs_ordered):
    """
    runs_ordered: list of (name, steps, values), pre-sorted in the order
    you expect them to rank (e.g. by peak_lr, descending).

    Returns the number of rank inversions across all pairs at all common
    steps, and a list of (step, name_i, name_j) for any inversion found.
    Zero inversions confirms a clean, non-crossing sweep.
    """
    names = [r[0] for r in runs_ordered]
    # build step -> value lookup dicts (pandas-free stand-in for a Series)
    lookups = []
    step_sets = []
    for name, steps, values in runs_ordered:
        d = {float(s): float(v) for s, v in zip(steps, values)}
        lookups.append(d)
        step_sets.append(set(d.keys()))

    common_steps = sorted(set.intersection(*step_sets)) if step_sets else []

    inversions = []
    for step in common_steps:
        row = [lookups[k][step] for k in range(len(lookups))]
        for i, j in itertools.combinations(range(len(row)), 2):
            if row[i] > row[j]:
                inversions.append((step, names[i], names[j]))

    return len(inversions), inversions


# --------------------------------------------------------------------------
# Threshold / target-loss analysis
# --------------------------------------------------------------------------

def _normal_sf(z):
    """Survival function P(Z > z) for standard normal, scipy if available."""
    if _HAVE_SCIPY:
        return float(norm.sf(z))
    # crude fallback via erf, good enough for reporting
    import math
    return 0.5 * math.erfc(z / math.sqrt(2))


def threshold_report(stats: TailStats, threshold: float):
    """
    Given a run's tail stats and a target loss threshold, report:
      - z: how many std the tail mean is above the threshold
      - p_single: probability a single checkpoint dips below threshold
                  (normal approximation -- treats checkpoints as iid, which
                  is optimistic since real checkpoints are autocorrelated)
      - n_for_even_odds: approx number of independent checkpoints needed
                  for a ~50% chance at least one dips below threshold
    """
    if stats.std is None or stats.std == 0:
        return {
            "run": stats.name,
            "tail_mean": stats.mean,
            "tail_std": stats.std,
            "threshold": threshold,
            "z_std_above_threshold": None,
            "p_single_checkpoint_below": None,
            "checkpoints_needed_for_50pct_chance": None,
        }

    z = (stats.mean - threshold) / stats.std
    p_single = _normal_sf(z)
    # solve 1 - (1-p)^n = 0.5  =>  n = ln(0.5) / ln(1-p)
    if p_single <= 0 or p_single >= 1:
        n_for_even_odds = float("inf")
    else:
        n_for_even_odds = np.log(0.5) / np.log(1 - p_single)

    return {
        "run": stats.name,
        "tail_mean": stats.mean,
        "tail_std": stats.std,
        "threshold": threshold,
        "z_std_above_threshold": z,
        "p_single_checkpoint_below": p_single,
        "checkpoints_needed_for_50pct_chance": n_for_even_odds,
    }


def print_threshold_report(report):
    print(f"\n--- Threshold analysis: {report['run']} vs target {report['threshold']} ---")
    if report["tail_std"] is None:
        print(f"  tail mean:  {report['tail_mean']:.4f}   tail std: n/a (too few points)")
        below = report["tail_mean"] < report["threshold"]
        print(f"  no std available -- can't do a probabilistic estimate.")
        print(f"  mean is {'BELOW' if below else 'ABOVE'} threshold, but with n<2 points "
              f"treat this as anecdotal, not a real signal.")
        return
    print(f"  tail mean:  {report['tail_mean']:.4f}   tail std: {report['tail_std']:.4f}")
    print(f"  gap to threshold: {report['z_std_above_threshold']:.2f} std")
    p = report["p_single_checkpoint_below"]
    n = report["checkpoints_needed_for_50pct_chance"]
    print(f"  P(single checkpoint < threshold) ~= {p:.2e}")
    if np.isfinite(n):
        print(f"  checkpoints needed for ~50% chance of one dipping below: ~{n:,.0f}")
        note = ("plausible from noise alone" if n < 200 else
                 "needs a real improvement in the mean, not just noise/luck")
        print(f"  => {note}")
    else:
        print("  => essentially impossible from noise alone at this mean/std")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_paths", nargs="+",help="One or more W&B CSV exports")
    ap.add_argument("--tail-start-step", type=float, default=None,
                    help="Absolute step where the schedule has flattened (e.g. anneal floor)")
    ap.add_argument("--tail-frac", type=float, default=0.15,
                    help="Fallback: fraction of logged points to treat as the tail")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Target validation loss to run threshold analysis against")
    ap.add_argument("--check-crossings", action="store_true",
                    help="Check for rank inversions across all runs, in the order given by column order")
    args = ap.parse_args()

    all_runs = {}
    for path in args.csv_paths:
        all_runs.update(load_wandb_csv(path))

    print(f"Loaded {len(all_runs)} run(s): {list(all_runs.keys())}\n")

    stats_by_run = {}
    print("--- Tail statistics ---")
    for name, (steps, values) in all_runs.items():
        st = tail_stats(steps, values, tail_start_step=args.tail_start_step,
                         tail_frac=args.tail_frac, name=name)
        stats_by_run[name] = st
        print(st)

    if len(all_runs) >= 2:
        print("\n--- Pairwise seed/run noise (log-ratio std over the tail) ---")
        names = list(all_runs.keys())
        for a, b in itertools.combinations(names, 2):
            steps_a, values_a = all_runs[a]
            steps_b, values_b = all_runs[b]
            mean_lr, std_lr, n_used, fb = seed_noise(steps_a, values_a, steps_b, values_b,
                                                      min_step=args.tail_start_step)
            if std_lr is None:
                print(f"  {a} vs {b}: skipped (fewer than 2 overlapping points even after fallback)")
                continue
            flag = "  [fallback: widened window]" if fb else ""
            print(f"  {a} vs {b}: mean={mean_lr:+.4f}  std={std_lr:.4f}  "
                  f"(~{std_lr*100:.1f}% noise band, n={n_used}){flag}")

    if args.check_crossings:
        ordered = [(name, *all_runs[name]) for name in all_runs]
        n_inv, inversions = crossing_check(ordered)
        print(f"\n--- Crossing check (order = column order in file) ---")
        print(f"  rank inversions found: {n_inv}")
        for step, i, j in inversions[:20]:
            print(f"    step {step:.0f}: {i} > {j}")

    if args.threshold is not None:
        for name, st in stats_by_run.items():
            print_threshold_report(threshold_report(st, args.threshold))



if __name__ == "__main__":
    sys.argv = [
        "program_name",   # argv[0] is ignored by argparse
        r"C:\Users\Melissa\Downloads\wandb_export_2026-09-20T20_39_07.191-04_00.csv",
        "--tail-start-step", "1800",
        "--tail-frac", "0.15",
        "--threshold", "1.45",
    ]

    sys.exit(main())

