from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tests.bpe_tokenizer import BpeTokenizer
from tests.nn_loader import load_checkpoint
from tests.nn_train import build_model, load_tokens
from tests.nn_yaml import load_yaml_config

def reconstruct_offsets(
    num_tokens, seed=29, ctx_len=1024,
    microbatch_size=4, grad_accum=32,
    first_update=90, last_update=140,
):
    g = torch.Generator().manual_seed(seed)
    high = num_tokens - ctx_len - 1  # match current get_batch exactly
    if high <= 0:
        raise ValueError("Dataset too short for the current sampler")

    rows = []
    for update in range(last_update + 1):  # zero-based, as in training
        for microbatch in range(grad_accum):
            for sequence in range(microbatch_size):
                start = torch.randint(
                    low=0, high=high, size=(1,), generator=g
                ).item()
                if update >= first_update:
                    rows.append((update, microbatch, sequence, start))

    return np.array(rows, dtype=np.int64)

# Use the same training token array that get_batch receives.
# offsets = reconstruct_offsets(len(x))
# Columns: update, microbatch, sequence, start

def show_offset_text(training_tokens, config, single_offset_arr):
    tokenizer = BpeTokenizer.from_files(config.gen.vocab_folder, config.gen.special_tokens)
    for start in single_offset_arr:
        token_ids = training_tokens[start : start + config.model.seq_len + 1].tolist()
        print(f"\n--- start={start} ---")
        print(tokenizer.decode(token_ids))

def show_windows_text(x, offsets, config, update_steps_arr= [94]):
    tokenizer = BpeTokenizer.from_files(config.gen.vocab_folder, config.gen.special_tokens)
    for update in update_steps_arr:
        show_windows(x, config, offsets, tokenizer, update=update, microbatch=0)

def show_windows(x, config,  offsets, tokenizer, update, microbatch=0):
    # Columns: update, microbatch, sequence, start
    selected = offsets[
        (offsets[:, 0] == update)
        & (offsets[:, 1] == microbatch)
    ]

    for upd, mb, seq, start in selected:
        start = int(start)
        # 1024 input tokens + the extra token used for shifted targets
        ids = x[start:start + config.model.seq_len + 1].tolist()

        text = tokenizer.decode(ids)  # adapt to your tokenizer's API
        print(
            f"\n--- update={int(upd)}, microbatch={int(mb)}, "
            f"sequence={int(seq)}, offset={start} ---"
        )
        print(text)

def compare_updates(model, logits_fn, x, offsets, device, from_selected, to_selected,  ctx_len=1024):
    """
    offsets columns: update, microbatch, sequence, start
    logits_fn(inputs) -> logits [B, T, V], without softmax
    x: original training token array
    """
    selected = pd.DataFrame(
        offsets, columns=["update", "microbatch", "sequence", "start"]
    )
    selected = selected[selected["update"].between(from_selected, to_selected)]
    results = []

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for (update, microbatch), group in selected.groupby(["update", "microbatch"], sort=True):
                group = group.sort_values("sequence")
                starts = group["start"].tolist()

                windows = np.stack([
                    x[int(s):int(s) + ctx_len + 1] for s in starts
                ]).astype(np.int32, copy=False)

                inputs = torch.tensor(
                    windows[:, :-1], dtype=torch.int32, device=device
                )
                # Cross-entropy targets must be int64.
                targets = torch.tensor(
                    windows[:, 1:], dtype=torch.long, device=device
                )

                logits = logits_fn(inputs)
                assert logits.shape[:2] == targets.shape

                token_losses = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)

                sequence_losses = token_losses.mean(dim=1).cpu().tolist()

                for sequence, start, loss in zip(
                    group["sequence"], starts, sequence_losses
                ):
                    results.append({
                        "update": int(update),
                        "microbatch": int(microbatch),
                        "sequence": int(sequence),
                        "start": int(start),
                        "loss": float(loss),
                    })

                if microbatch == 31:
                    print(f"Scored update {update}", flush=True)
    finally:
        model.train(was_training)

    sequences = pd.DataFrame(results)
    microbatches = (
        sequences.groupby(["update", "microbatch"])["loss"]
        .mean().rename("loss").reset_index()
    )

    updates = sequences.groupby("update").agg(
        mean_loss=("loss", "mean"),
        max_sequence_loss=("loss", "max"),
    )
    updates["max_microbatch_loss"] = (
        microbatches.groupby("update")["loss"].max()
    )
    return sequences, microbatches, updates

def display_updates(sequences, microbatches, updates, from_selected, to_selected):
    print(updates.round(4).to_string())

    print("\nHighest-loss updates:")
    print(updates.nlargest(10, "mean_loss").round(4))

    print("\nHighest-loss sequences:")
    print(sequences.nlargest(12, "loss").to_string(index=False))

    # Local files: retain offsets so high-loss windows can be decoded later.
    sequences.to_csv(f"spike_sequence_losses_{from_selected}-{to_selected}.csv", index=False)
    updates.to_csv(f"spike_update_losses_{from_selected}-{to_selected}.csv")


def main(args):
    _, config = load_yaml_config(args.config, args)

    training_tokens, _ = load_tokens(config)
    show_offset_text(training_tokens, config, [1_022_856_199, 497_533_245])
    return


    llm = build_model(config)
    load_checkpoint(config.run.resume_from, llm, None, None, config.run.device) # type: ignore

    offsets = reconstruct_offsets(
            len(training_tokens),
            seed=config.run.seed,
            ctx_len=config.model.seq_len,
            microbatch_size=config.train.batch_size // config.train.grad_accum,
            grad_accum=config.train.grad_accum,
            first_update=args.first_update,
            last_update=args.last_update,
        )
    
    print('Starting to compare updates')

    sequences, microbatches, updates = compare_updates(llm, llm, training_tokens, offsets, config.run.device, args.from_selected, args.to_selected, config.model.seq_len)
    display_updates(sequences, microbatches, updates, args.from_selected, args.to_selected)

if __name__ == '__main__':
    """
    Usage: 
    python train.py -c tests/config/gpt2_tiny.yaml -plr 0.0003
    """
    parser = ArgumentParser(description="Train a transformer model.")
    parser.add_argument('-c', '--config', type=str, default='tests/config/abla_batch/cs336_owt.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('-rf', '--resume_from', type=str, help='Start training from a saved checkpoint.')
    parser.add_argument('-fu', '--first_update', type=int, default=90, help='First update step to recompute')
    parser.add_argument('-lu', '--last_update', type=int, default=140, help='Last update to recompute')
    parser.add_argument('-fs', '--from_selected', type=int, default=90, help='Start of focus range')
    parser.add_argument('-ts', '--to_selected', type=int, default=140, help='End of focus range')
    parser.add_argument('-e', '--exit', type=str, help='y or n to exit right away', default='n')
    parser.add_argument('-s', '--show', type=str, help='y or n to exit right away', default='y')
   
    args = parser.parse_args()

    
    if args.exit == 'y':
        print (f'{args}')
    else:
        main(args)