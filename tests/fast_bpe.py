import heapq
from collections import Counter, defaultdict
from typing import NamedTuple, Tuple, List, Dict, Set, Optional
from dataclasses import dataclass
# from collections import Counter
# from typing import Dict, Set, Tuple
from cs336_basics.pretokenization_example import pretokenize
from tests.common import FIXTURES_PATH


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

class HeapEntry(NamedTuple):
    freq: int
    tie_breaker: ReverseBytes
    byte_pair: BytesPair

class BpeWord(NamedTuple):
    label: str          # text chunk, used for readability only, ex 'train'
    value: tuple[bytes] # ex: (b't', b'r', b'a', b'i', b'n')
    idx: int            # word index in the corpus (primary key)

Corpus = List[BpeWord]


def word_from_bytes(idx, list_or_tuple):
    try:
        full_bytes = b''.join(list_or_tuple)
        label = full_bytes.decode('utf-8')
        return BpeWord(label, tuple(list_or_tuple), idx)
    except Exception as ex:
        raise ex

PairPositionsMap = defaultdict[BytesPair,list[int]] 

def ppp(key: BytesPair, positions_map:PairPositionsMap) -> str:
    sorted_positions = sorted(positions_map[key])
    lines = [f"*** {key} **** ["]
    for p in sorted_positions:
        lines.append(
            f"  {p}, "
        )
    lines.append("]\n")
    return "".join(lines)

def pppf(pairs: set[BytesPair], positions_map: PairPositionsMap):
    for bp in pairs:
        print(ppp(bp, positions_map))
    pass

def list_byte_pairs (word_bytes, start_loc):
    result = [(ii, BytesPair(word_bytes[ii], word_bytes[ii+1])) for ii in range(start_loc, len(word_bytes) - 1)]
    return result

def init_frequencies(distinct_words: Counter[tuple[bytes]], positions: dict[tuple[bytes], set[int]]):
    """
    inputs:
        distinct_words is a dict which shows how many times each word appears in the entire corpus
        positions is a dict which shows in which corpus positions those words appear

    outputs:
        dict which shows how many times each distinct byte_pair appears in the corpus
        dict which shows set if word idx in corpus where each byte pair appears
        note: count of the positions in second dict should equal the frequency in first dict

    """
    pair_freq_dict: Counter[BytesPair] = Counter()
    pair_positions_dict: PairPositionsMap = defaultdict(list)
    for dw in distinct_words:
        word_len = len(dw)
        for ii in range(word_len - 1):
            bp = BytesPair(dw[ii], dw[ii+1])
            cnt = distinct_words[dw]
            pair_freq_dict[bp] += cnt
            word_positions = positions[dw]
            pair_positions_dict[bp].extend(word_positions)

    return pair_freq_dict, pair_positions_dict

    
def init_heap_queue(freq_heap: list[HeapEntry], pair_positions: PairPositionsMap) :
    # build max-heap using negative frequency
    
    for p, pair_locations in pair_positions.items():
        # for tie-breaking: pick lexicographically larger
        rank = ReverseBytes(p)
        f = len(pair_locations)
        tup = HeapEntry (-f, rank, p)
        freq_heap.append(tup)
    
    heapq.heapify(freq_heap)


def pop_best_pair(myheap, pair_positions_map: PairPositionsMap):
    while myheap:
        freq_neg, r, pair = heapq.heappop(myheap)
        pair_positions = pair_positions_map[pair]
        if -freq_neg == len(pair_positions):
            return freq_neg, r, pair
    return 0, None, None

def create_new_word(old_word: BpeWord, top_pair: BytesPair, new_token: bytes ):
    old_bytes = old_word.value
    new_bytes = []
    ii = 0
    locations = []
    while ii < len(old_bytes):
        if ii + 1 < len(old_bytes) and old_bytes[ii] == top_pair.a and old_bytes[ii + 1] == top_pair.b:
            new_bytes.append(new_token)
            locations.append(ii)
            ii += 2
        else:
            new_bytes.append(old_bytes[ii])
            ii += 1
        pass
    result = word_from_bytes(old_word.idx, new_bytes)
    return result, locations

def collect_affected_pairs(word: BpeWord, same_words, affected_boundaries, receiver: List[tuple[int,BytesPair]]):
    start_loc = max(0, affected_boundaries[0] - 1) # TODO - consider upper boundary also
    pairs_tuples = list_byte_pairs(word.value, start_loc)
    for _, pt in pairs_tuples:
        tup = (word.idx, pt)
        for idx in same_words:
            tup = (idx, pt)
            receiver.append(tup)

def frequency_aware_removal(old_positions: list[int], top_pair_counter: Counter[int]):
    """
    Frequency aware removal of positions need to support words with repeating pairs, ex: 'in' in 'training'
    """
    keep_positions = []
    for x in old_positions:
        if top_pair_counter[x] > 0:
            top_pair_counter[x] -= 1
        else:
            keep_positions.append(x)
    return keep_positions

def adjust_positions(removed_pairs:list[tuple[int,BytesPair]], added_pairs: list[tuple[int,BytesPair]], top_pair_positions, 
                     positions_map: PairPositionsMap):
    # top_pair_counter: Counter = Counter(top_pair_positions)
    delta_positions = PairPositionsMap()

    for word_idx, bp in added_pairs:
        new_positions = positions_map[bp]
        new_positions.append(word_idx) 
        delta_positions[bp] = new_positions

    for word_idx, bp in removed_pairs:
        old_positions = positions_map[bp]
        try:
            old_positions.remove(word_idx)
            delta_positions[bp] = old_positions
        except Exception as ex:
            # print(f'No-op as pair {bp} already removed from corpus word {word_idx} ')
            pass
    
    return delta_positions

def merge(corpus: Corpus, top_pair: BytesPair, new_token, pair_positions_map: PairPositionsMap, 
          word_positions: dict[str, set[int]]):

    # the need for copying is b/c we are adding/removing/modifying positions being iterated over TODO - check if are missing something
    top_pair_positions = [x for x in pair_positions_map[top_pair]]
    
    pairs_removed: list[tuple[int,BytesPair]] = list()
    pairs_added: list[tuple[int,BytesPair]] = list()
    same_words = set()
    new_word = corpus[0] # hmmm better would be None
    for word_idx in top_pair_positions:
        if word_idx in same_words:
            corpus[word_idx] = new_word
            continue
        else :
            corpus_word = corpus[word_idx]
            same_words = word_positions[corpus_word.label].copy()

        new_word = corpus[word_idx]

        new_word, locs = create_new_word(new_word, top_pair, new_token)
        if len(locs) > 0:
            collect_affected_pairs(corpus[word_idx], same_words, locs, pairs_removed)
            collect_affected_pairs(new_word, same_words, locs, pairs_added)

        corpus[word_idx] = new_word
    
    delta_positions = adjust_positions(pairs_removed, pairs_added, top_pair_positions, pair_positions_map)
    return delta_positions

def train_fast_bpe (corpus_raw: list[tuple[bytes]],  distinct_words: Counter, word_positions: dict[tuple,set[int]], vocab, num_merges):

    merges = []
    corpus: Corpus = [word_from_bytes(idx, bytes_tuple) for idx, bytes_tuple in enumerate(corpus_raw)]
    word_positions2 = {x.label : word_positions[x.value] for x in corpus}

    pair_freq_dict, pair_positions_map = init_frequencies(distinct_words, word_positions)

    heap_queue = []
    init_heap_queue(heap_queue, pair_positions_map)

    for epoch in range(num_merges):
        top_freq, _, top_pair = pop_best_pair(heap_queue, pair_positions_map)
        print(f"Epoch= {epoch}, best pair/freq = {top_pair} / {-top_freq}")
        if not top_pair:
            print('Breaking early - no more pairs')
            break
        new_idx = len(vocab)
        new_token = b''.join(top_pair)
        vocab[new_idx] = new_token
        
        delta_positions_map = merge(corpus, top_pair, new_token, pair_positions_map, word_positions2)
        print(f'Affected byte pairs in this epoch = {len(delta_positions_map)}')
        merges.append((top_pair.a, top_pair.b))

        try:
            init_heap_queue(heap_queue, delta_positions_map)
        except Exception as ex:
            print(ex)
    
    return merges

if __name__ == "__main__":
    corpus_path = FIXTURES_PATH / "low_lower_bpe.txt"
    special_tokens = ["<|endoftext|>"]
    ids = pretokenize(corpus_path, special_tokens, 4)
    num_merges = 6
    vocab, merges = train_fast_bpe(ids, num_merges)
    print(merges)