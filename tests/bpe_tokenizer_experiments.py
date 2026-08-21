import os
import random
import time

import numpy as np

from tests.bpe_tokenizer import from_pkl, read_tokens_binary, write_tokens_binary

def sample_docs(file_path, end_token="<|endoftext|>", num_items=10):
    buf = []
    reservoir = [None] * num_items
    samples = 0

    start = time.perf_counter()
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB
            if not chunk:
                break

            buf.append(chunk)
            joined = "".join(buf)
            parts = joined.split(end_token)

            # All complete docs except the last partial one
            for doc in parts[:-1]:
                if samples < num_items:
                    reservoir[samples] = doc
                else:
                    choice = random.randrange(0, samples + 1)
                    if choice < num_items:
                        reservoir[choice] = doc                     

                samples += 1

            # Keep the last partial doc
            buf = [parts[-1]]


    # Remove any None entries if file had fewer than num_items docs
    my_time = time.perf_counter() - start
    print(f'Selected {num_items} from {file_path} in {my_time:.0f} seconds.')
    return [d for d in reservoir if d is not None]

# in case we want first 10 docs (faster)
def stream_10_docs(file_path, end_token="<|endoftext|>", max_docs=10):
    buf = []
    docs = []

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)  # read 1MB at a time
            if not chunk:
                break

            buf.append(chunk)
            joined = "".join(buf)

            # Split by end_token
            parts = joined.split(end_token)

            # All but the last part are complete docs
            for doc in parts[:-1]:
                docs.append(doc)
                if len(docs) >= max_docs:
                    return docs

            # Keep the last partial doc in buffer
            buf = [parts[-1]]

    # If file ends without enough docs
    if buf and len(docs) < max_docs:
        docs.append(buf[0])

    return docs

def calc_ratio(doc_text, tokenizer):
    raw_bytes = doc_text.encode('utf-8')
    bytes_count = len(raw_bytes)
    tokens = tokenizer.encode(doc_text)
    token_count = len(tokens)

    # 5. Calculate compression ratio (Bytes per Token)
    bytes_per_token = bytes_count / token_count
    msg = (f'Raw Bytes : {bytes_count}, '
    f'Total Tokens: {token_count}, '
    f"Compression Efficiency: {bytes_per_token:.2f} bytes/token")

    print(msg)
    return bytes_per_token


def tinystories_inputs():
    subject= 'Compression results for Tinystories'
    dataset_train = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\data\TinyStoriesV2-GPT4-train.txt'
    dataset_valid = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\data\TinyStoriesV2-GPT4-valid.txt'
    vocab_folder = r"C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\out\tinystories_GPT4"
    special_token = '<|endoftext|>'
    return subject, dataset_train, dataset_valid, vocab_folder, special_token

def owt_inputs():
    subject= 'Compression results for OpenWebText'
    dataset_train = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\data\owt_train.txt'
    dataset_dev = r'C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\data\owt_valid.txt'
    vocab_folder = r"C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\out\owt"
    special_token = '<|endoftext|>'
    return subject, dataset_train, dataset_dev, vocab_folder, special_token



def report_comprehension():
    for subject, dataset_train, _, vocab_folder, special_token in [tinystories_inputs(), owt_inputs()]:
        print(f'Subject: {subject}')
        tokenizer = from_pkl(vocab_folder, [special_token])
        ten_stories = sample_docs(dataset_train)
        ratios = [calc_ratio(doc, tokenizer) for doc in ten_stories ]
        my_max = max(ratios)
        my_min = min(ratios)
        my_avg = sum (ratios) / len(ratios)
        print(f'Range: {my_min:.2f} - {my_max:.2f}, Avg: {my_avg:.2f}')


def serialize_tokens():
    for _, dataset_train, dataset_valid, vocab_folder, special_token in [tinystories_inputs(), owt_inputs()]: 
        tokenizer = from_pkl(vocab_folder, [special_token])
        write_tokens_binary(dataset_train, tokenizer, os.path.join(vocab_folder,'tokens_train.bin'))
        write_tokens_binary(dataset_valid, tokenizer, os.path.join(vocab_folder,'tokens_valid.bin'))   
# def serialize_tinystories_tokens ():
#     _, dataset_train, dataset_valid, vocab_folder, special_token = tinystories_inputs()
#     tokenizer = from_pkl(vocab_folder, [special_token])
#     write_tokens_binary(dataset_train, tokenizer, os.path.join(vocab_folder,'tokens_train.bin'))
#     # write_tokens_binary(dataset_valid, tokenizer, os.path.join(vocab_folder,'tokens_valid.bin'))

if __name__ == '__main__':
    serialize_tokens()
