# this code is adopted from https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

import os
import pathlib
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
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="strict")

        # 1. Segment on special tokens — this pattern exists ONLY to split
        #    documents apart; it is not the pattern that produces pre-tokens.
        special_pattern = "|".join(re.escape(t) for t in special_tokens_arr)
        doc_arr = re.split(special_pattern, chunk_text) if special_pattern else [chunk_text]

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

    return frequency

def pretokenize_serial(input_path, special_tokens_arr, start_end_pairs) -> list[FrequencyTable]:  
    if len(start_end_pairs) != 1:
        raise Exception("Expected exactly one start-end pair for serial pretokenization.")
    start = start_end_pairs[0][0]
    end = start_end_pairs[0][1]
    freq_tbl = pretokenize_chunk(start, end, input_path, special_tokens_arr)
    return [freq_tbl]

def pretokenize_parallel(input_path, special_token_arr, start_end_pairs) -> list[FrequencyTable]:
    args = [(start, end, input_path, special_token_arr) for start,end in start_end_pairs]

    with Pool() as pool:
        worker_results = pool.starmap(pretokenize_chunk, args)

    return worker_results

def pretokenize(input_path, special_tokens_arr, desired_chunks=1) -> list[PreToken]:
    if len(special_tokens_arr) == 0:
        raise Exception('At least one special token required')
    start_end_pairs = find_chunk_boundaries_v2(input_path, desired_chunks, special_tokens_arr[0].encode('utf-8'))
    partial_counts: list[FrequencyTable] = []
    if desired_chunks == 1:
        partial_counts = pretokenize_serial(input_path, special_tokens_arr, start_end_pairs)
    else:
        partial_counts = pretokenize_parallel(input_path, special_tokens_arr, start_end_pairs)

 # Merge all partial frequency tables
    global_counts = Counter()
    for partial_count in partial_counts:
        global_counts.update(partial_count)

    result = [PreToken(list(symbol_tuple), freq=count) for symbol_tuple, count in global_counts.items()]
    return result


## Usage
if __name__ == "__main__":

    corpus_path = 'C:\\Users\\Melissa\\cs336\\assignment1-basics\\tests\\fixtures\\tinystories_sample.txt'
    # corpus_path = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\tests\fixtures\low_lower_bpe.txt'
 
    desired_num_chunks = 4
    special_token = "<|endoftext|>"
    pretokens = pretokenize(corpus_path, [special_token], desired_num_chunks)
    print(f"\n\nPreTokens: {pretokens}")



