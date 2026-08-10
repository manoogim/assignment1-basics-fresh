from collections import defaultdict
import os

from tests.bpe_heap import HeapEntry, add_new_pair, init_heap_queue, pop_best_pair
from tests.bpe_loader import pretokenize
from tests.bpe_pretoken import PreToken
from tests.bpe_types import PairFrequencyTable, ReverseIndex

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


def train_bpe( input_path: str | os.PathLike, vocab_size: int, special_tokens_arr: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
# Input
    # input_path: str Path to a text file with BPE tokenizer training data.
    # vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    # special_tokens: list[str] A list of strings to add to the vocabulary. During training, treat them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.
# Output
    # vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    # merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    
    pretokens: list[PreToken] = pretokenize(input_path, special_tokens_arr)
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
    return vocab, merges

if __name__ == "__main__":

    corpus_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\tests\fixtures\low_lower_bpe.txt'
    special_token = "<|endoftext|>"
    vocab_size = 300
    vocab, merges = train_bpe(corpus_path, vocab_size, [special_token])
    print(vocab)
    print(merges)


