from collections import defaultdict
import json
import os
import pickle
import time

from tests.bpe_heap import HeapEntry, add_new_pair, init_heap_queue, pop_best_pair
from tests.bpe_loader import pretokenize
from tests.bpe_pretoken import PreToken
from tests.bpe_types import PairFrequencyTable, ReverseIndex
from tests.common import gpt2_bytes_to_unicode

def apply_one_merge(best_pair, affected_ids, pretokens, pair_counts, reverse_index, heap):
    """Orchestrates: calls replace_symbol for each affected pretoken,
    aggregates deltas, updates reverse index, then mutates heap and global frequencies once."""
    total_delta: PairFrequencyTable = defaultdict(int)

    for pid in affected_ids:
        pre_token = pretokens[pid]
        delta, adds, removes = pre_token.replace_symbol(best_pair)

        for pair in adds:
            reverse_index[pair].add(pid)

        for pair in removes:
            reverse_index[pair].remove(pid)
            if len(reverse_index[pair]) == 0:
                del reverse_index[pair] # fully consumed by this merge

        for pair, d in delta.items():
            new_freq = d * pre_token.freq
            total_delta[pair] += new_freq

    # single application pass
    for pair, d in total_delta.items():
        new_count = pair_counts.get(pair, 0) + d
        pair_counts[pair] = new_count
        if new_count <= 0:
            del pair_counts[pair]
        else:
            add_new_pair(heap, pair, new_count)


def train( input_path: str | os.PathLike, vocab_size: int, special_tokens_arr: list[str], num_workers=1, log_every=None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
# Input
    # input_path: str Path to a text file with BPE tokenizer training data.
    # vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    # special_tokens: list[str] A list of strings to add to the vocabulary. During training, treat them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.
# Output
    # vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    # merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.

    pretokens: list[PreToken] = pretokenize(input_path, special_tokens_arr, desired_chunks=num_workers)

    pair_counts: PairFrequencyTable = defaultdict(int)
    reverse_index: ReverseIndex = defaultdict(set)
    for pid in range(len(pretokens)):
        pretoken = pretokens[pid]
        byte_pairs = pretoken.as_byte_pairs()
        for bp in byte_pairs:
            pair_counts[bp] += pretoken.freq
            reverse_index[bp].add(pid)

    heap: list[HeapEntry] = init_heap_queue(pair_counts)

    num_merges = vocab_size - 256 - len(special_tokens_arr)
    vocab = {i : special_tokens_arr[i].encode('utf-8') for i in range(len(special_tokens_arr))}
    nn = len(vocab)
    for i in range(256):
        vocab[nn+i] = bytes([i])  # initial byte vocabulary
    merges = []

    for jj in range(num_merges):
        freq, best_pair = pop_best_pair(heap, pair_counts)
        if best_pair is None:
            break
        # Update the reverse index and the heap with the new merged token
        affected_ids = [x for x in reverse_index[best_pair]]
        new_symbol = best_pair.a + best_pair.b
        apply_one_merge(best_pair, affected_ids, pretokens, pair_counts, reverse_index, heap)
        vocab[len(vocab)] = new_symbol
        merges.append((best_pair.a, best_pair.b))
        if (log_every is not None and (jj % log_every == 0 ) or jj == num_merges - 1):          
            print(
                f"merge {jj+1}/{num_merges}  "
                f"pair={best_pair}  freq={freq}  "
                f"vocab_size={len(vocab)}  "
                f"time={time.strftime("%Y-%m-%d %H:%M:%S")}"
            )
    return vocab, merges

def serialize(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], prefix: str, metadata: dict):
    """Serializes the vocabulary and merges and metadata to disk."""

    path = f"out/{prefix}"
    os.makedirs(path, exist_ok=True)

    metadata_path = f"{path}/config.json"
    print(metadata)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4) 

    vocab_path = f"{path}/vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    byte_to_char = gpt2_bytes_to_unicode()

    def readable(b: bytes) -> str:
        return "".join(byte_to_char[byte] for byte in b)

    with open(f"{path}/vocab_readable.txt", "w", encoding="utf-8") as f:
        for token_id, token_bytes in sorted(vocab.items()):
            f.write(f"{token_id}\t{readable(token_bytes)}\n")

    merges_path = f"{path}/merges.pkl"
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Serialized config, vocab and merges to {path}")

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens_arr: list[str], num_workers=1, prefix = None, log_every=None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    t0 = time.perf_counter()
    vocab, merges = train(input_path, vocab_size, special_tokens_arr, num_workers=num_workers, log_every=log_every) 
    elapsed = time.perf_counter() - t0

    print(f"BPE Training took {elapsed:.1f} seconds. Vocab size={len(vocab)}, merge pairs={len(merges)}")
    max_id, max_value = max(vocab.items(), key=lambda kv: len(kv[1]))
    print(f"Longest word is {max_value}")
    if prefix is not None:
        metadata = {
            "prefix": prefix,
            "num_workers": num_workers,
            "input_path": str(input_path),
            "vocab_size": vocab_size,
            "special_tokens": special_tokens_arr,
            "num_merges": len(merges),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trained_sec": f"{elapsed:.1F} seconds",
            "longest_word": max_value.decode('UTF-8')
        }

        serialize(vocab, merges, prefix, metadata)
    return vocab, merges

def train_tiny_stories():
    corpus_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\tests\fixtures\tinystories_sample_5M.txt'
    special_token = "<|endoftext|>"
    vocab_size = 10000
    vocab, merges = train_bpe(corpus_path, vocab_size, [special_token], num_workers=4, prefix='tinystories_5M', log_every=500) 

def train_low_lower():
    corpus_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\tests\fixtures\low_lower_bpe.txt'
    special_token = "<|endoftext|>"
    vocab_size = 300
    
    vocab, merges = train_bpe(corpus_path, vocab_size, [special_token], prefix='lowlower')

    

if __name__ == "__main__":
    # train_low_lower()
    train_tiny_stories()

