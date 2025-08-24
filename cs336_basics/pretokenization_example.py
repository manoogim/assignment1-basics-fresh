import os
import pathlib
from typing import BinaryIO
import regex as re

from tests.common import GPT2_SPLIT_PATTERN
from multiprocessing import Pool
from collections import Counter, defaultdict


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
    
def pretokenize(input_path, special_tokens_arr, desired_chunks):
    if len(special_tokens_arr) > 1:
        raise Exception('Only one special token allowed')
    special_token_text = special_tokens_arr[0]
    
    start_end_pairs = find_chunk_boundaries_v2(input_path, desired_chunks, special_token_text.encode('utf-8'))
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    ids = []
    distinct_words = Counter()
    combined_positions = defaultdict(set)
    for start, end in start_end_pairs:
        chunk_ids, counters, positions = pretokenize_chunk(start, end, input_path, special_token_text)
        ids.extend(chunk_ids)
        distinct_words += counters
        for k,v in positions.items():
            combined_positions[k].update(v)
        print(f"end={end} ++++++++++++++++++++++++++++++++++++++++++++++++")
    return ids, distinct_words

def pretokenize_chunk(start, end, input_path, special_token_text):
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
        doc_arr = chunk_text.split(special_token_text)
        print(f"\n!!!!!!!!!!!chunk bytes size={len(chunk_text.encode('utf-8'))}")
        compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)
        ids = []
        distinct_words = Counter()
        positions_dict = defaultdict(set)
        word_idx = 0
        for doc_txt in doc_arr:
            matches = re.finditer(compiled_pattern, doc_txt)
            for m in matches:
                chunk = m.group()
                try:
                    chunk_bytes = tuple(chunk.encode('utf-8'))
                    byte_tokens = tuple(bytes([b]) for b in chunk_bytes)
                    ids.append(byte_tokens)
                    distinct_words[byte_tokens] += 1
                    positions_dict[byte_tokens].add(word_idx)
                    word_idx += 1
                except Exception as ex:
                    print(ex)

    return ids, distinct_words, positions_dict
    
def pretokenize_parallel(input_path, special_tokens_arr, desired_chunks):
    if len(special_tokens_arr) > 1:
        raise Exception('Only one special token allowed')
    special_token_text = special_tokens_arr[0]
    
    start_end_pairs = find_chunk_boundaries_v2(input_path, desired_chunks, special_token_text.encode('utf-8'))
    args = [(start, end, input_path, special_token_text) for start,end in start_end_pairs]
    positions = defaultdict(set)
    with Pool() as pool:
        results = pool.starmap(pretokenize_chunk, args)
        # ids elements are results of worker threads - we need to flatten
        chunked_ids, counters, positions_dicts = zip(*results)
        ids = [x for worker_list in chunked_ids for x in worker_list]
        distinct_words = sum(counters, Counter())
        for pos in positions_dicts:
            for k,v in pos.items():
                positions[k].update(v) 
        return ids, distinct_words, dict(positions )        

## Usage
if __name__ == "__main__":
    corpus_path = 'C:\\Users\\Melissa\\cs336\\assignment1-basics\\tests\\fixtures\\tinystories_sample.txt'
    corpus_path = 'C:\\Users\\Melissa\\cs336\\assignment1-basics\\tests\\fixtures\\low_lower_bpe.txt'
 
    desired_num_chunks = 3
    special_token = "<|endoftext|>"
    ids = pretokenize_parallel(corpus_path, [special_token], desired_num_chunks)
    ids2 = pretokenize_parallel(corpus_path, [special_token], desired_num_chunks+1)
    miss = [x for x in ids if not x in ids2]
    miss2 = [x for x in ids2 if not x in ids]
    from collections import Counter
    c1 = Counter(tuple(x) for x in ids)
    c2 = Counter(tuple(x) for x in ids2)
    missed = c1 - c2
    missed2 = c2 - c1
    print(f"Number of ids: {len(ids)} and {len(ids2)}")

# tests\fixtures\address.txt
# C:\Users\Melissa\cs336\assignment1-basics\tests\fixtures\address.txt
# with open('C:\\Users\\Melissa\\cs336\\assignment1-basics\\tests\\fixtures\\tinystories_sample.txt', "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token
#         print(f"end={end}")
#         print('++++++++++++++++++++++++++++++++++++++++++++++++')
