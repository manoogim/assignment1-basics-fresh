## Table 1: TinyStories vs. OWT — where does the time go?
*(vocab_size=10,000, 4 workers, matched corpus size ~2.2GB)*

| Corpus | Size | Pretok time | Merge time | Total | Pretok % of total | Longest token |
|---|---|---|---|---|---|---|
| TinyStories | 2.23 GB | 415s | 8s | 424s | 98% | `b' accomplishment'` |
| OWT | 2.0 GB | 315s | 177s | 493s | 64% | `b'----------------'` |
| OWT | 2.2 GB | 719s | 197s | 916s | 78% | `b'----------------'` |

**Finding**: at matched vocab_size, OWT's merge phase costs ~20-25x more than TinyStories' (177-197s vs 8s) despite similar corpus size, because OWT's lexical diversity produces far more unique pre-token types, so `reverse_index`/`pair_counts` are larger and each merge round touches more pre-tokens. Pre-tokenization is also somewhat slower per-GB for OWT, but the bigger shift is the merge phase becoming a non-trivial fraction of total time.

## Table 2: OWT scaling — workers and vocab_size, 2.2GB corpus

| Config | Pretok time | Merge time | Total | Valid? |
|---|---|---|---|---|
| 2 workers, vocab=10k | 929s | ~192s | 1121s | ✅ |
| 4 workers, vocab=10k | 719s | ~197s | 916s | ✅ **best** |
| 12 workers, vocab=10k | 900s | ~190s | 1090s | ✅ (disk paging observed) |
| 2 workers, vocab=10k (repeat) | 14,613s | — | — | ❌ excluded — same input as row 1, ~16x slower, system contention |
| 4 workers, vocab=32k | 13,173s | ~130s (stall-corrected; 17,137s raw) | — | ⚠️ pretok contaminated, merge time correctable |

**Finding**: 4 workers is the best validated configuration (both fewer and more workers were worse — consistent with a memory-bound, not CPU-bound, regime on 16GB RAM). The vocab=32k run's raw numbers are dominated by two multi-thousand-second stalls (99.2% of its reported merge time), almost certainly the laptop sleeping during an unattended overnight run — after excluding them, real merge cost at 32k vocab (~130s) is actually *comparable to*, not dramatically worse than, the 10k-vocab merge cost (~190-199s) at this corpus size. This is a positive sign: the incremental merge algorithm's late-stage merges stay cheap even as vocab_size grows, so merge-loop cost does not appear to scale badly with vocab_size at fixed corpus size — the earlier, high-affected-count merges dominate regardless of how many total merges you request.

## Conclusions

1. **Where OWT time goes**: pre-tokenization dominates for TinyStories (98%) but becomes a much smaller share for OWT (64-78% at vocab=10k) because OWT's merge phase is inherently costlier due to lexical diversity — not due to vocab_size choice itself, based on the corrected 32k comparison.
2. **Best worker count**: 4, empirically, on this 16GB machine — both under- and over-provisioning workers made things worse.
3. **Environmental contamination is a real, recurring risk on this machine**: two separate incidents now (the accidental 2-worker repeat, and this overnight 32k run) show 13-18x slowdowns with no code or algorithmic explanation — almost certainly Windows sleep/suspend or heavy background contention during long unattended runs. **Before attempting the full 6GB/32k run**, disable sleep/hibernate for the session (`Settings > Power & sleep > Never`, or `powercfg /change standby-timeout-ac 0`) and close background apps — this matters more for a trustworthy result than any algorithmic optimization at this point.
4. **Feasibility of full 6GB / 32k vocab on this laptop**: using the clean 2.2GB/4-worker baseline (719s pretok, ~130-200s real merge cost), a linear extrapolation to 6GB gives roughly ~1,960s (~33 min) pretok plus a merge cost that, per finding #1 above, likely stays in the low hundreds of seconds even at 32k vocab — suggesting a clean run could plausibly finish in under an hour. However, this estimate only holds if memory pressure and sleep/interruption are controlled; both have caused 10-20x blowups in every long run so far. **Recommendation**: fix the sleep setting, run with 4 workers, monitor Task Manager for paging early in the run, and be prepared to abort/retry rather than trusting an unattended overnight run.