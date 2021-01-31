import pickle

import numpy as np
import torch
import torch.nn.functional as F

from joeynmt.choices_dict import build_choices_dict


def lin_escape(s):
    return (
        s.replace('(', 'BRACKETOPEN')
        .replace(')', 'BRACKETCLOSE')
        .replace("'", 'SAVEAPO')
        .replace(',', 'SAVECOMMA')
        .replace(' ', '€')
    )


class CharBasedTagRestricter:

    def __init__(self, vocab, tag_dict):
        self.vocab = vocab

        tag_dict_indices = {}
        for key, values in tag_dict.items():
            new_key = lin_escape(key) + '@0'
            new_key = tuple(vocab.sentence_to_array(new_key))
            if values is None:
                new_values = None
            else:
                new_values = {lin_escape(value) + '@s' for value in values}
                new_values.add('or@2')
                new_values.add('or@3')
                new_values = {tuple(vocab.sentence_to_array(value))
                              for value in new_values}
            tag_dict_indices[new_key] = new_values

        self.space_idx = vocab.stoi[' ']

        self.key_choices = build_choices_dict(list(tag_dict_indices),
                                              end_symbol=self.space_idx)
        self.value_choices = {
            key: (
                build_choices_dict(list(values), end_symbol=self.space_idx)
                if values else None
            )
            for key, values in tag_dict_indices.items()
        }


        self.keyval_seq = vocab.sentence_to_array('keyval@2')
        self.or2_seq = vocab.sentence_to_array('or@2')
        self.or3_seq = vocab.sentence_to_array('or@3')

    def get_last_n_tokens(self, seq, n, fill=True):
        tokens = []
        cur_token = []
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] == self.space_idx:
                cur_token.reverse()
                tokens.append(np.array(cur_token, dtype=np.int32))
                cur_token = []
                if len(tokens) == n:
                    break
            else:
                cur_token.append(int(seq[i]))
        if i == 0 and len(cur_token) > 0 and len(tokens) < n:
            cur_token.reverse()
            tokens.append(np.array(cur_token, dtype=np.int32))

        if fill:
            tokens.extend([None] * (n - len(tokens)))

        tokens.reverse()
        return tokens

    def get_next_choices(self, seq, choices):
        for elm in seq:
            if elm in choices:
                choices = choices[elm]
            else:
                return None
        return choices

    def key_seq_if_in_value_generation(self, last_six_tokens):
        eq = np.array_equal
        lst = last_six_tokens
        kv, or2, or3 = (self.keyval_seq, self.or2_seq, self.or3_seq)
        if eq(lst[-3], kv):
            return lst[-2]
        elif eq(lst[-4], kv) and (eq(lst[-2], or2) or eq(lst[-2], or3)):
            return lst[-3]
        elif eq(lst[-5], kv) and (eq(lst[-3], or2) or eq(lst[-3], or3)):
            return lst[-4]
        elif eq(lst[-6], kv) and eq(lst[-4], or3):
            return lst[-5]
        return None

    def __call__(self, alive_seq, logits):
        for i in range(alive_seq.shape[0]):
            last_six_tokens = self.get_last_n_tokens(alive_seq[i], n=6)
            ltt = [t for t in last_six_tokens if t is not None]
            ltt_s = ', '.join(
                ''.join(sentence)
                for sentence in self.vocab.arrays_to_sentences(ltt)
            )
            choices = None
            if np.array_equal(last_six_tokens[-2], self.keyval_seq):
                # Key generation in process. Restrict to allowed keys
                choices = self.get_next_choices(last_six_tokens[-1],
                                                self.key_choices)
            else:
                key_seq = self.key_seq_if_in_value_generation(last_six_tokens)
                if key_seq is not None:
                    # Value generation in process. Restrict to allowed values
                    choices_for_key = self.value_choices.get(tuple(key_seq))
                    if choices_for_key:
                        choices = self.get_next_choices(last_six_tokens[-1],
                                                        choices_for_key)

            if choices:
                forbidden_indices = torch.tensor(
                    [idx for idx in range(logits.shape[-1])
                     if idx not in choices],
                    dtype=torch.long,
                    device=logits.device
                )
                logits[i, 0].index_fill_(0, forbidden_indices, float('-inf'))
        return logits


def build_logit_adjuster(vocab, tag_dict_file):
    if tag_dict_file:
        with open(tag_dict_file, 'rb') as f:
            tag_dict = pickle.load(f)
            return CharBasedTagRestricter(vocab, tag_dict)
    return None
