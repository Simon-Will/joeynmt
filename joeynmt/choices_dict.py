#!/usr/bin/env python3

from collections import defaultdict
import random

STRINGS = [
    'diet:vegan',
    'diet:vegetarian',
    'denomination',
    'diet:gluten_free',
    'highway',
    'historic',
    'name',
]


class Symbol:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'Symbol({!r})'.format(self.name)

    def __hash__(self):
        return hash(repr(self))


END_SYMBOL = Symbol('END')


def build_choices_dict(allowed_seqs, end_symbol=END_SYMBOL):
    choices = {}
    first_to_rests = defaultdict(set)
    for elm in allowed_seqs:
        if len(elm) > 0:
            first_to_rests[elm[0]].add(elm[1:])
        else:
            choices[end_symbol] = None

    for first, rests in first_to_rests.items():
        if rests is not None:
            choices[first] = build_choices_dict(rests, end_symbol=end_symbol)
        else:
            choices[first] = {}

    return choices


def generate_word(d):
    chars = []
    while True:
        c = random.choice(list(d))
        if c is END_SYMBOL:
            break
        chars.append(c)
        d = d[c]
    return ''.join(chars)


if __name__ == '__main__':
    D = build_choices_dict(STRINGS)
    print(D)
    print(generate_word(D))
