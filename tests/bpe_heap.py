import heapq
from typing import NamedTuple
from .bpe_types import BytesPair, PairFrequencyTable

 # for tie-breaking, to prefer lexicographically larger
class ReverseBytes:
    def __init__(self, b):
        self.b = b  # b should be a bytes or tuple of ints

    def __lt__(self, other):
        # Reverse the normal comparison
        return self.b > other.b
    
    def __eq__(self, other):
        return self.b == other.b


class HeapEntry(NamedTuple):
    freq: int
    # order of fields matters, so the tie breaker must be before the byte pair, so that the heapq will use it for tie-breaking
    tie_breaker: ReverseBytes
    byte_pair: BytesPair
   

    def __repr__(self):
        result = f"(byte_pair={repr(self.byte_pair)}, count={-self.freq})"
        return result

def init_heap_queue(pair_freqs: PairFrequencyTable) -> list[HeapEntry]:
     # build max-heap using negative frequency, and heapify once at the end for efficiency
    heap = [HeapEntry(-freq, ReverseBytes(p), p) for p, freq in pair_freqs.items()]
    heapq.heapify(heap)
    return heap

def add_new_pair(myheap, new_pair: BytesPair, freq: int):
    rank = ReverseBytes(new_pair)
    tup = HeapEntry(-freq, rank, new_pair)
    # Add the new pair to the heap while maintaining the heap property
    heapq.heappush(myheap, tup)

# Remove entry with max frequency. 
def pop_best_pair(myheap, freq_table: PairFrequencyTable):
    while myheap:
        freq_neg, _, pair = heapq.heappop(myheap)
        live_count = freq_table.get(pair,0)
        if freq_neg == -live_count:
            return freq_neg, pair
    return 0, None



if __name__ == "__main__":
    # Example usage
    pair_freqs = {
        BytesPair(b'a', b'b'): 5,
        BytesPair(b'c', b'd'): 3,
        BytesPair(b'e', b'f'): 8,
    }

    heap = init_heap_queue(pair_freqs)
    print("Initial heap:", heap)
