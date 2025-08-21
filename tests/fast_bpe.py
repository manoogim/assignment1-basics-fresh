import heapq
from collections import Counter, defaultdict
from typing import NamedTuple, Tuple, List, Dict, Set, Optional
from dataclasses import dataclass
# from collections import Counter
# from typing import Dict, Set, Tuple
from cs336_basics.pretokenization_example import pretokenize
from tests.common import FIXTURES_PATH

def dict_diff(d1, d2, path=""):
    diffs = []

    # Keys present in both
    for key in d1.keys() & d2.keys():
        if d1[key] != d2[key]:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                diffs.extend(dict_diff(d1[key], d2[key], path + f"{key}.\n"))
            else:
                diffs.append(f"Changed: {path}{key} | {ppp(key,d1[key])} -> {ppp(key,d2[key])}\n")

    # Keys only in d1
    for key in d1.keys() - d2.keys():
        diffs.append(f"Removed: {path}{key} | was {ppp(key, d1[key])}\n")

    # Keys only in d2
    for key in d2.keys() - d1.keys():
        diffs.append(f"Added: {path}{key} | now {ppp(key, d2[key])}\n")

    for d in diffs:
        print(d)
    return diffs



class ReverseBytes:
    def __init__(self, b):
        self.b = b  # b should be a bytes or tuple of ints

    def __lt__(self, other):
        # Reverse the normal comparison
        return self.b > other.b

    def __gt__(self, other):
        return self.b < other.b
    
    def __eq__(self, other):
        return self.b == other.b

    def __le__(self, other):
        # Reverse the normal comparison
        return self.b >= other.b

    def __ge__(self, other):
        return self.b <= other.b
    
    def __repr__(self):
        return f"ReverseBytes({self.b})"

class BytesPair(NamedTuple):
    a: bytes
    b: bytes

    def __repr__(self):
        return f"({self.a!r}, {self.b!r})"


from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Position:
    word: int
    left: Optional[BytesPair]
    right: Optional[BytesPair]

    def __hash__(self):
        # Only hash by word & pos so left/right can differ without affecting equality
        return hash((self.word))

    def __eq__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        # Equality ignores left/right — good for set lookup/removal
        return (self.word) == (other.word)

    def __lt__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        # Sort order is based only on word & pos
        return (self.word) < (other.word)


def ppp(key: BytesPair, positions: set[Position]) -> str:
    sorted_positions = sorted(positions, key=lambda p: (p.word))
    lines = [f"*** {key} **** ["]
    for p in sorted_positions:
        lines.append(
            f"  Position(word={p.word}, "
            f"left={p.left}, right={p.right}),"
        )
    lines.append("===========================")
    lines.append("")  # final newline
    return "\n".join(lines)


def pppdf( freq_dict: Counter[BytesPair], pos_dict: Dict[BytesPair,Set[Position]]):
    for key in freq_dict.keys():
        print(ppp(key, pos_dict[key]))

class HeapEntry(NamedTuple):
    freq: int
    tie_breaker: ReverseBytes
    byte_pair: BytesPair

Corpus = List[List[bytes]]


def add_freq(pair_freq_dict: Counter[BytesPair], pair: BytesPair):
    pair_freq_dict[pair] += 1

def del_freq(pair_freq_dict: Counter[BytesPair], pair: BytesPair):
    pair_freq_dict[pair] -= 1

def add_position(positions_dict: Dict[BytesPair, Set[Position]], pair: BytesPair, position: Position):
    positions_dict[pair].add(position)

def del_position(positions_dict: Dict[BytesPair, Set[Position]], word_idx, byte_idx, pair: BytesPair):
    pos = Position(word_idx, byte_idx, None, None)
    positions_dict[pair].remove(pos)

def init_frequencies(corpus: Corpus) -> Tuple[Counter[BytesPair], Dict[BytesPair, Set[Position]]]:
    pair_freq_dict: Counter[BytesPair] = Counter()
    positions_dict: Dict[BytesPair, Set[Position]] = defaultdict(set)

    def handle_word(word_idx, bytes_pair_list: list[BytesPair] ):
        nn = len(bytes_pair_list)
        for jj in range(nn):
            bp = bytes_pair_list[jj]          
            left_bp = None if jj == 0 else bytes_pair_list[jj - 1]
            right_bp = None if jj == nn - 1 else bytes_pair_list[jj + 1]

            position = Position(word_idx, left_bp, right_bp)
            add_position(positions_dict, bp, position)
            # if bp == (b'i',b'n'):
            #     asciiword = ''.join(x.decode() for x in word)
            #     print(asciiword)
            add_freq(pair_freq_dict, bp)
        pass
    
    for ii in range(len(corpus)):
        word = corpus[ii]
        word_len = len(word)
        bytes_pair_list = [BytesPair(word[i], word[i + 1]) for i in range(word_len - 1)]
        
        handle_word(ii, bytes_pair_list)
    return pair_freq_dict, positions_dict

def init_heap_queue(pair_freq_dict: Dict[BytesPair, int]) -> List[HeapEntry]:
    # build max-heap using negative frequency
    freq_heap: list[HeapEntry] = []
    for p, f in pair_freq_dict.items():
        # for tie-breaking: pick lexicographically larger
        rank = ReverseBytes(p)
        tup = HeapEntry (-f, rank, p)
        freq_heap.append(tup)
    
    heapq.heapify(freq_heap)
    return freq_heap


def pop_best_pair(myheap, pair_freqs: Counter[BytesPair]):
    while myheap:
        freq_neg, r, pair = heapq.heappop(myheap)
        if -freq_neg == pair_freqs[pair]:
            return freq_neg, r, pair
    return 0, None, None

def merge2( corpus: Corpus, top_pair: BytesPair, new_idx: int, new_token, positions_dict: Dict[BytesPair,Set[Position]]):
    new_var = positions_dict[top_pair]
    new_var1 = list(new_var)
    sorted_positions = sorted(new_var1)
    for pos in sorted_positions: 
        if pos.word > len(corpus) -1:
            continue
        word = corpus[pos.word]
        #    merge the new token 
        def merge_all_pairs(word, target_pair, new_token):
            i = 0
            merged = []
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == target_pair:
                    merged.append(new_token)
                    i += 2  # Skip the merged pair
                else:
                    merged.append(word[i])
                    i += 1
            if i == len(word) - 1:
                merged.append(word[-1])  # Append last token if not part of a pair
            return merged
        # TODO consider avoiding new word creation by choosing mutable structure for word word to 
        # new_word = word[:loc] + [new_token] + word[loc+2:]
        new_word = merge_all_pairs(word, top_pair, new_token)
        corpus[pos.word] = new_word

def train_fast_bpe(corpus_raw, num_merges):
    # count token pairs, and produce locations map
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = []
    corpus = [[bytes([id]) for id in word] for word in corpus_raw]
    pair_freq_dict, positions_dict = init_frequencies(corpus)
    heap_queue = init_heap_queue(pair_freq_dict)

    # 1st iter
    for epoch in range(num_merges):
        top_freq, rank, top_pair = pop_best_pair(heap_queue, pair_freq_dict)
        print(f"Epoch= {epoch}, top_pair / freq = {top_pair}/{top_freq}")
        if not top_pair:
            print('Breaking early - no more pairs')
            break
        new_idx = len(vocab)
        new_token = b''.join(top_pair)

        # Add updates incrementally
        # new_pair_freqs, new_position_dict = init_frequencies(corpus)
        affected_pairs = merge(corpus, top_pair, new_idx, new_token, pair_freq_dict, positions_dict)
        # dict_diff(new_position_dict, positions_dict)
        for pair in affected_pairs:
            freq = pair_freq_dict[pair]
            if freq > 0:
                entry = HeapEntry(freq, ReverseBytes(pair), pair)
                heapq.heappush(heap_queue, entry)

        # update artifacts
        vocab[new_idx] = new_token
        merges.append(top_pair)
    
    return vocab, merges

def train_fast_bpe2 (corpus, vocab, num_merges):

    merges = []

    pair_freq_dict, positions_dict = init_frequencies(corpus)

    heap_queue = init_heap_queue(pair_freq_dict)

    for epoch in range(num_merges):
        top_freq, rank, top_pair = pop_best_pair(heap_queue, pair_freq_dict)
        print(f"Epoch= {epoch}, best pair/freq = {top_pair} / {top_freq}")
        if not top_pair:
            print('Breaking early - no more pairs')
            break
        new_idx = len(vocab)
        new_token = b''.join(top_pair)
        vocab[new_idx] = new_token
        merge2(corpus, top_pair, new_idx, new_token, positions_dict)

        merges.append(top_pair)

        # full reset after each merge
        # import copy

        # prev_freq = copy.deepcopy(pair_freq_dict)
        # prev_pos = copy.deepcopy(positions_dict)

        pair_freq_dict, positions_dict = init_frequencies(corpus)
        # dict_diff(prev_pos, positions_dict)
        heap_queue = init_heap_queue(pair_freq_dict)

    
    return merges

if __name__ == "__main__":
    corpus_path = FIXTURES_PATH / "low_lower_bpe.txt"
    special_tokens = ["<|endoftext|>"]
    ids = pretokenize(corpus_path, special_tokens, 4)
    num_merges = 6
    vocab, merges = train_fast_bpe(ids, num_merges)
    print(merges)