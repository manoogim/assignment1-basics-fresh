# Non-hashable, mutable type for for training phase - do not use for counting frequencies
from collections import Counter, defaultdict

from tests.bpe_types import BytesPair, PairFrequencyTable, show


class PreToken:
    __slots__ = ("symbols", "freq")
    def __init__(self, symbols: list[bytes], freq: int):
        self.symbols = symbols
        self.freq = freq
    def __repr__(self):
        return f"<{show(self.symbols)} x{self.freq}>"

    def as_byte_pairs(self) -> list[BytesPair]:
        result = [BytesPair(self.symbols[i], self.symbols[i+1]) for i in range(len(self.symbols)-1)]
        return result

    def replace_symbol(self, pair: BytesPair) -> tuple[PairFrequencyTable, set, set]:
        """Given the pair being merged, update my symbols with the new pair.
        Return (pair_count_delta, add_pairs, remove_pairs)."""
        symbols = self.symbols
        new_symbols = []
        jj = 0
        while jj < len(symbols):
            if jj < len(symbols) - 1 and symbols[jj] == pair.a and symbols[jj + 1] == pair.b:
                new_symbols.append(pair.a + pair.b)
                jj += 2
            else:
                new_symbols.append(symbols[jj])
                jj += 1

        old_counts = Counter(self.as_byte_pairs())
        self.symbols = new_symbols
        new_counts = Counter(self.as_byte_pairs())

        pair_count_delta: PairFrequencyTable = defaultdict(int)
        add_pairs = set()
        remove_pairs = set()

        for p in old_counts.keys() | new_counts.keys():
            delta = new_counts[p] - old_counts[p]
            if delta == 0:
                continue
            pair_count_delta[p] = delta
            if new_counts[p] == 0:
                remove_pairs.add(p)
            elif old_counts[p] == 0:
                add_pairs.add(p)

        return pair_count_delta, add_pairs, remove_pairs

    def replace_symbol2(self, pair: BytesPair) -> tuple[PairFrequencyTable, set, set]:
    # """Given the pair being merged, update my symbols with the new pair.
    # return ( pair_count_delta, add_pairs, remove_pairs)."""   
        symbols: list[bytes] = self.symbols
        new_symbols = []
        pair_count_delta: PairFrequencyTable = defaultdict(int)
        add_pairs = set()
        remove_pairs = set()
        
        jj = 0
        while jj < len(symbols):
            is_replacing = jj < len(symbols) - 1 and symbols[jj] == pair.a and symbols[jj + 1] == pair.b
            if is_replacing:
                new_symbols.append(pair.a + pair.b)
                jj += 2  # Skip the next symbol as it's part of the merged pair
            else:
                new_symbols.append(symbols[jj])
                jj += 1
        old_pairs = self.as_byte_pairs()
        self.symbols = new_symbols
        new_pairs = self.as_byte_pairs()
        removed_pairs = [x for x in old_pairs if x not in new_pairs]
        added_pairs = [x for x in new_pairs if x not in old_pairs]
        for p in removed_pairs:
            pair_count_delta[p] -= 1
            remove_pairs.add(p)
        for p in added_pairs:
            pair_count_delta[p] += 1
            add_pairs.add(p)

        return pair_count_delta, add_pairs, remove_pairs
    

if __name__ == "__main__":
    chunk = 'mississippi'
    chunk = 'training'
    # chunk = 'newest'
    chunk = 'abxab'
    symbol = tuple(bytes([b]) for b in chunk.encode("utf-8"))
    pt = PreToken(list(symbol), freq=1)
    print(pt)
    pairs = pt.as_byte_pairs()
    print(pairs)
    best_pair = pairs[1]

    pair_count_delta, _, _ = pt.replace_symbol(best_pair)
    print(pt)
    print(pt.as_byte_pairs())
    print(pair_count_delta)
