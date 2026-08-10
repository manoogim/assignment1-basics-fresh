from collections import defaultdict
from typing import NamedTuple


def show(symbols: tuple[bytes, ...] | list[bytes]) -> str:
    return "|".join(s.decode("utf-8", errors="replace") for s in symbols)

# Immutable mapping byte tuples to their frequencies, holds initial counting of word frequencies 
FrequencyTable = dict[tuple[bytes, ...], int]
    
class BytesPair(NamedTuple):
    a: bytes
    b: bytes
    def __repr__(self):
        return f"({show([self.a])}, {show([self.b])})"


# reverse mapping of byte pairs to their positions in the frequency table
ReverseIndex = defaultdict[BytesPair,set[int]]

# true source of truth for the frequency of byte pairs during merges
PairFrequencyTable = dict[BytesPair, int]

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
