# this code is adopted from https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

import os
import time
from typing import BinaryIO
import regex as re

from tests.bpe_types import GPT2_SPLIT_PATTERN, FrequencyTable
from multiprocessing import Pool
from collections import Counter, defaultdict

from tests.bpe_pretoken import PreToken

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def find_chunk_boundaries_v2(input_path: str, desired_num_chunks: int, split_special_token: bytes) -> list[tuple[int,int]]:
    with open(input_path, 'rb') as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token)
        start_end_pairs = zip(chunk_boundaries[:-1], chunk_boundaries[1:])
        return list(start_end_pairs)

def pretokenize_chunk(start, end, input_path, special_tokens_arr) -> FrequencyTable:
    t0 = time.perf_counter()
    with open(input_path, 'rb') as f:
        f.seek(start)
        raw = f.read(end - start)
        t1 = time.perf_counter()

        chunk_text = raw.decode("utf-8", errors="strict")
        del raw
        t2 = time.perf_counter()

        # 1. Segment on special tokens — this pattern exists ONLY to split
        #    documents apart; it is not the pattern that produces pre-tokens.
        special_pattern = "|".join(re.escape(t) for t in special_tokens_arr)
        doc_arr = re.split(special_pattern, chunk_text) if special_pattern else [chunk_text]
        t3 = time.perf_counter()

        # 2. Pre-tokenize each document independently with the GPT-2 regex —
        #    this is the pattern that actually produces pre-tokens.
        compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)
   
        frequency: FrequencyTable = defaultdict(int)
        for doc_txt in doc_arr:
            matches = re.finditer(compiled_pattern, doc_txt)
            for m in matches:
                chunk = m.group()
                symbol = tuple(bytes([b]) for b in chunk.encode("utf-8"))
                # symbol = tuple(ch.encode("utf-8") for ch in chunk)
                frequency[symbol] += 1  # Use the encoded bytes tuple as the key for frequency counting
        t4 = time.perf_counter()

        print(
        f"[worker pid={os.getpid()}] "
        f"bytes={end-start:,}  "
        f"read={t1-t0:.1f}s  decode={t2-t1:.1f}s  split={t3-t2:.1f}s  "
        f"regex_scan={t4-t3:.1f}s  total={t4-t0:.1f}s"
    )
    return frequency

def pretokenize_serial(input_path, special_tokens_arr, start_end_pairs) -> list[FrequencyTable]:  

    results = []
    for start, end in start_end_pairs:
        freq_tbl = pretokenize_chunk(start, end, input_path, special_tokens_arr)
        results.append(freq_tbl)
    return results

    return worker_results
def pretokenize_parallel(input_path, special_token_arr, start_end_pairs, num_workers=4) -> list[FrequencyTable]:
    args = [(start, end, input_path, special_token_arr) for start, end in start_end_pairs]
    with Pool(processes=num_workers) as pool:
        worker_results = pool.starmap(pretokenize_chunk, args)
    return worker_results

def pretokenize(input_path, special_tokens_arr, desired_chunks=4, num_workers=1) -> list[PreToken]:
    if len(special_tokens_arr) == 0:
        raise Exception('At least one special token required.')

    if desired_chunks % num_workers != 0:
        raise Exception('Number of chunks must be divisible by number of workers.')
    partial_counts: list[FrequencyTable] = []
    start_end_pairs = find_chunk_boundaries_v2(input_path, desired_chunks, special_tokens_arr[0].encode('utf-8'))
    if num_workers == 1:
        partial_counts = pretokenize_serial(input_path, special_tokens_arr, start_end_pairs)
    else:
        partial_counts = pretokenize_parallel(input_path, special_tokens_arr, start_end_pairs, num_workers)

 # Merge all partial frequency tables
    global_counts = Counter()
    for partial_count in partial_counts:
        global_counts.update(partial_count)

    result = [PreToken(list(symbol_tuple), freq=count) for symbol_tuple, count in global_counts.items()]
    return result


if __name__ == "__main__":

    corpus_path = 'C:\\Users\\Melissa\\cs336\\assignment1-basics\\tests\\fixtures\\tinystories_sample.txt'
    # corpus_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\tests\fixtures\low_lower_bpe.txt'
    # corpus_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\data\TinyStoriesV2-GPT4-train.txt'
 
    desired_num_chunks = 4
    special_token = "<|endoftext|>"
    num_workers = 2
    pretokens = pretokenize(corpus_path, [special_token], desired_chunks=desired_num_chunks, num_workers=num_workers)
    print(f"\n\nPreTokens: {pretokens}")



