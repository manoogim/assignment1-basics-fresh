import regex as re

from tests.bpe_types import GPT2_SPLIT_PATTERN


class BpeTokenizer:
    def __init__(self, vocab: dict[int,bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str]=None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)


    def decode(self, ids: list[int]) -> str:
        part_bytes = []
        for id in ids:
            if id in self.vocab.keys():
                part_bytes.append(self.vocab[id])
            else:
                raise ValueError(f'Invalid token:M {id}')
        text_bytes = b''.join(part_bytes)
        result = text_bytes.decode('utf-8', errors='replace')
        return result

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        symbols = [bytes([b]) for b in text_bytes]

        while len(symbols) >= 2:
            pairs = ((symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1))
            best_pair = min(
                (p for p in pairs if p in self.merge_ranks),
                key=lambda p: self.merge_ranks[p],
                default=None,
            )
            if best_pair is None:
                break
            symbols = merge_symbols(symbols, best_pair)

        return [self.reverse_vocab[s] for s in symbols]


    def encode_iterable(self, iterable):
        for chunk in iterable:
            yield from self.encode(chunk)

    def encode(self, text: str) -> list[int]:
        if len(self.special_tokens) > 0:
            tmp = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "(" + "|".join(re.escape(k) for k in tmp) + ")"
            special_chunks = re.split(special_pattern, text)
        else:
            special_chunks = [text]
        ids =[]
        for part in special_chunks:
            if part in self.special_tokens:
                ids.append(self.reverse_vocab[part.encode('utf-8')])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
def merge_symbols(symbols: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    new_symbols = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            new_symbols.append(pair[0] + pair[1])
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return new_symbols
