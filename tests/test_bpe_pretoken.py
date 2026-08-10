"""
Unit tests for PreToken.replace_symbol.

Adjust the import below to match your actual module path
(e.g. `from tests.bpe_pretoken import PreToken` /
`from tests.bpe_types import BytesPair` depending on how your
package is laid out).
"""

import pytest

from tests.bpe_pretoken import PreToken
from tests.bpe_types import BytesPair


def make_pretoken(word: str, freq: int = 1) -> PreToken:
    symbols = [bytes([b]) for b in word.encode("utf-8")]
    return PreToken(symbols, freq)


def test_replace_symbol_repeated_pair_non_adjacent_occurrences():
    """'mississippi', merge ('s','s') -> two separate, non-overlapping
    occurrences of the target pair itself."""
    pt = make_pretoken("mississippi")
    pair = BytesPair(b"s", b"s")

    delta, add_pairs, remove_pairs = pt.replace_symbol(pair)

    assert pt.symbols == [b"m", b"i", b"ss", b"i", b"ss", b"i", b"p", b"p", b"i"]

    assert dict(delta) == {
        BytesPair(b"i", b"s"): -2,
        BytesPair(b"s", b"s"): -2,
        BytesPair(b"s", b"i"): -2,
        BytesPair(b"i", b"ss"): 2,
        BytesPair(b"ss", b"i"): 2,
    }
    assert add_pairs == {BytesPair(b"i", b"ss"), BytesPair(b"ss", b"i")}
    assert remove_pairs == {
        BytesPair(b"i", b"s"),
        BytesPair(b"s", b"s"),
        BytesPair(b"s", b"i"),
    }
    # unaffected pairs like (m,i), (i,p), (p,p), (p,i) must not appear at all
    untouched = {
        BytesPair(b"m", b"i"),
        BytesPair(b"i", b"p"),
        BytesPair(b"p", b"p"),
        BytesPair(b"p", b"i"),
    }
    assert untouched.isdisjoint(delta.keys())


def test_replace_symbol_two_occurrences_collide_into_new_pair():
    """'training', merge ('i','n') -> the two merged tokens end up
    adjacent to each other, creating a brand-new pair (in,in), and the
    pair (n,i) between them vanishes entirely even though it wasn't
    the merge target."""
    pt = make_pretoken("training")
    pair = BytesPair(b"i", b"n")

    delta, add_pairs, remove_pairs = pt.replace_symbol(pair)

    assert pt.symbols == [b"t", b"r", b"a", b"in", b"in", b"g"]

    assert dict(delta) == {
        BytesPair(b"a", b"i"): -1,
        BytesPair(b"i", b"n"): -2,
        BytesPair(b"n", b"i"): -1,
        BytesPair(b"n", b"g"): -1,
        BytesPair(b"a", b"in"): 1,
        BytesPair(b"in", b"in"): 1,
        BytesPair(b"in", b"g"): 1,
    }
    assert add_pairs == {
        BytesPair(b"a", b"in"),
        BytesPair(b"in", b"in"),
        BytesPair(b"in", b"g"),
    }
    assert remove_pairs == {
        BytesPair(b"a", b"i"),
        BytesPair(b"i", b"n"),
        BytesPair(b"n", b"i"),
        BytesPair(b"n", b"g"),
    }
    # (t,r) and (r,a) are untouched
    untouched = {BytesPair(b"t", b"r"), BytesPair(b"r", b"a")}
    assert untouched.isdisjoint(delta.keys())


def test_replace_symbol_simple_single_occurrence():
    """'newest', merge ('s','t') -> the textbook single-occurrence case
    from the assignment's worked example."""
    pt = make_pretoken("newest")
    pair = BytesPair(b"s", b"t")

    delta, add_pairs, remove_pairs = pt.replace_symbol(pair)

    assert pt.symbols == [b"n", b"e", b"w", b"e", b"st"]

    assert dict(delta) == {
        BytesPair(b"e", b"s"): -1,
        BytesPair(b"s", b"t"): -1,
        BytesPair(b"e", b"st"): 1,
    }
    assert add_pairs == {BytesPair(b"e", b"st")}
    assert remove_pairs == {BytesPair(b"e", b"s"), BytesPair(b"s", b"t")}


def test_replace_symbol_partial_count_survives_regression():
    """'abxab', merge ('b','x') -> regression test for the bug where a
    membership-based (rather than count-based) diff silently drops a
    pair whose count decreases but doesn't hit zero. Here (a,b) starts
    at count 2 and ends at count 1 (one occurrence untouched, the
    other's right neighbor gets absorbed into the merge) -- it must
    show up in delta as -1 and must NOT appear in remove_pairs, since
    it's still present afterward."""
    pt = make_pretoken("abxab")
    pair = BytesPair(b"b", b"x")

    delta, add_pairs, remove_pairs = pt.replace_symbol(pair)

    assert pt.symbols == [b"a", b"bx", b"a", b"b"]

    assert dict(delta) == {
        BytesPair(b"a", b"b"): -1,
        BytesPair(b"b", b"x"): -1,
        BytesPair(b"x", b"a"): -1,
        BytesPair(b"a", b"bx"): 1,
        BytesPair(b"bx", b"a"): 1,
    }
    assert add_pairs == {BytesPair(b"a", b"bx"), BytesPair(b"bx", b"a")}
    assert remove_pairs == {BytesPair(b"b", b"x"), BytesPair(b"x", b"a")}
    # the key assertion: (a,b) must NOT be in remove_pairs -- it survives
    assert BytesPair(b"a", b"b") not in remove_pairs


def test_replace_symbol_delta_is_per_occurrence_not_freq_weighted():
    """replace_symbol should return per-occurrence deltas; the caller
    (apply_one_merge) is responsible for multiplying by pretoken.freq.
    A pretoken with freq=6 should produce the exact same delta as
    freq=1 -- only the aggregation step downstream should scale it."""
    pt1 = make_pretoken("newest", freq=1)
    pt6 = make_pretoken("newest", freq=6)
    pair = BytesPair(b"s", b"t")

    delta1, _, _ = pt1.replace_symbol(pair)
    delta6, _, _ = pt6.replace_symbol(pair)

    assert dict(delta1) == dict(delta6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])