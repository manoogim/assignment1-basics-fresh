"""
Grab a truncated slice of a large corpus file and profile pre-tokenization
on it, so you can extrapolate time/memory before committing to a full run.

Usage:
    python sample_and_profile.py <path_to_full_corpus> [--mb 500] [--workers 4]

What it does:
    1. Copies the first N MB of the source file to a new sample file,
       trimmed to end at a special-token boundary (so you don't cut a
       document in half -- matters less for this quick test, but keeps
       things consistent with how your real pretokenize() works).
    2. Runs pretokenize_parallel on just that sample.
    3. Reports: sample size, wall time, peak memory (parent process,
       via tracemalloc), number of unique pre-tokens found.
    4. Extrapolates a rough full-corpus time estimate by linear scaling.

"""

import argparse
import os
import time
import tracemalloc

# --- adjust this import to match your project layout ---
from tests.bpe_loader import pretokenize


def make_sample(source_path: str, sample_path: str, target_mb: int, special_token: str) -> int:
    """Copy the first target_mb megabytes of source_path to sample_path,
    trimmed back to the last special-token boundary found so the sample
    doesn't end mid-document. Returns the actual sample size in bytes."""
    target_bytes = target_mb * 1024 * 1024
    special_bytes = special_token.encode("utf-8")

    with open(source_path, "rb") as src:
        data = src.read(target_bytes)

    # trim back to the last special-token boundary, if any, so the
    # sample doesn't end mid-document
    last_boundary = data.rfind(special_bytes)
    if last_boundary != -1:
        data = data[: last_boundary + len(special_bytes)]

    with open(sample_path, "wb") as dst:
        dst.write(data)

    return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", default=r"C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\data\owt_train.txt", help="Path to the full corpus file")
    parser.add_argument("--mb", type=int, default=2228, help="Sample size in MB (default: 2,000)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel chunks/workers")
    parser.add_argument("--special-token", default="<|endoftext|>", help="Document delimiter")
    parser.add_argument(
        "--sample-path",
        default=None,
        help="Where to write the truncated sample (default: alongside source, suffixed _sample)",
    )
    args = parser.parse_args()

    source_size = os.path.getsize(args.source_path)
    sample_path = args.sample_path or (args.source_path + f"_sample_{args.mb}mb.txt")

    print(f"Source file: {args.source_path}  ({source_size / 1e9:.2f} GB)")
    print(f"Writing {args.mb} MB sample to: {sample_path}")

    actual_bytes = make_sample(args.source_path, sample_path, args.mb, args.special_token)
    print(f"Sample written: {actual_bytes / 1e6:.1f} MB (trimmed to doc boundary)\n")

    print(f"Pretokenizing sample with {args.workers} workers ...")

    tracemalloc.start()
    t0 = time.perf_counter()

    pretokens = pretokenize(sample_path, [args.special_token], desired_chunks=args.workers)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    num_unique_pretokens = len(pretokens)

    print("\n--- Results on sample ---")
    print(f"Sample size:            {actual_bytes / 1e6:.1f} MB")
    print(f"Wall time:               {elapsed:.1f} s")
    print(f"Peak memory (parent):    {peak / 1e6:.1f} MB  (note: excludes worker subprocess memory)")
    print(f"Unique pre-token entries (pre-merge across workers): {num_unique_pretokens:,}")

    # --- rough linear extrapolation to full corpus ---
    scale = source_size / actual_bytes
    est_full_time = elapsed * scale

    print("\n--- Extrapolated full-corpus estimate (linear scaling, rough) ---")
    print(f"Full corpus size:        {source_size / 1e9:.2f} GB  ({scale:.1f}x sample)")
    print(f"Estimated pretok time:   {est_full_time:.0f} s  (~{est_full_time / 60:.1f} min)")
    print(
        "Note: memory does NOT scale purely linearly -- vocabulary diversity "
        "(unique pre-token count) grows sub-linearly with corpus size for "
        "natural text, but peak RAM still needs its own check on a larger "
        "sample before trusting this number for memory."
    )

    print(f"\nSample file left at: {sample_path} (delete manually if you don't need it)")


if __name__ == "__main__":
    main()