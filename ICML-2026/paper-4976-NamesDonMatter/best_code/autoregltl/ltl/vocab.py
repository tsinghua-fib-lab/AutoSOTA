from dataclasses import dataclass, field
from typing import Dict, Union, Any, Optional, Tuple, List


EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'

@dataclass
class CharVocab():
    """
    Character vocabulary for either LTL formulas or traces.
    PAD_TOKEN is always the first.
    START and EOS tokens are added to the end.
    """
    ops: list
    aps: list
    specials: list = field(default_factory=list)
    consts: list = field(default_factory=lambda: ['0', '1'])
    start: bool = False

    def __post_init__(self):
        # Ordered token list
        self.token_list = [PAD_TOKEN] + self.aps + self.consts + self.ops + self.specials
        self.pad_id = 0
        if self.start:
            self.start_id = len(self.token_list)
            self.token_list.append(START_TOKEN)
        self.eos_id = len(self.token_list)
        self.token_list.append(EOS_TOKEN)
        self.token_to_id = {token: i for i, token in enumerate(self.token_list)}

    def encode(self, s, prepend_start_token=True):
        """
        Encode a string into a list of integers with end token.
        """
        if isinstance(s, str):
            s = s.rstrip()
        encoded = [] if (not prepend_start_token) or not self.start else [self.start_id]
        encoded += [self.token_to_id[c] for c in s]
        return encoded + [self.eos_id]

    def decode(self, ids):
        if self.start and ids[0] == self.start_id:
            ids = ids[1:]
        try:
            ids = ids[:ids.index(self.eos_id)]
        except ValueError:
            try:
                ids = ids[:ids.index(self.pad_id)]
            except ValueError:
                pass
        return ''.join([self.token_list[i] for i in ids])

    def size(self):
        return len(self.token_list)

    def force_eos(self, logits, i):
        logits[i, :-1] = -float('inf')

    def disallow_eos(self, logits, i):
        logits[i, self.eos_id] = -float('inf')
        logits[i, self.pad_id] = -float('inf')
        # This sometimes breaks it:
        # if self.start:
        #     logits[i, self.start_id] = -float('inf')


@dataclass
class EncDecVocab():
    """
    Structure that includes both vocabs for encoder-decoder models.
    """
    inp: CharVocab
    out: CharVocab

    @staticmethod
    def create_ltl_vocab(
        aps,
        consts: list = ['0', '1'],
        trace_ops: list = [],
        # ltl_ops: list = ['!', '&', '|', '<->', 'xor'],
        # Equiv   | <-> | =
        # XOR     | xor | ^
        ltl_ops: list = ['!', '&', '|', '=', '^'],
    ):
        raise NotImplementedError()
        # start setting: should it be reversed?
        inp = CharVocab(aps=aps, consts=consts, ops=ltl_ops, start=True)
        out = CharVocab(aps=aps, consts=consts, ops=trace_ops, start=False)
        return EncDecVocab(inp, out)
    
    def are_inputs_compatible(self, other):
        return self.inp == other.inp

    def _get_aps(self):
        return self.inp.aps

    aps = property(fget=_get_aps)

    def num_classes(self):
        return self.out.size()


@dataclass
class MergedLTLVocab():
    """
    Merged vocab for both LTL formulae and traces.
    Used by decoder-only models.
    First token is always EOS_TOKEN, which is also used as PAD_TOKEN.
    """
    aps: list
    consts: list = field(default_factory=lambda: ['0', '1'])
    trace_ops: list = field(default_factory=lambda: [])
    ltl_ops: list = field(default_factory=lambda: ['!', '&', '|', '=', '^'])
    use_start_token: bool = False
    use_pad_token: bool = False
    use_eos_token: bool = True

    def __post_init__(self):
        # Ordered token list
        self.token_list = []
        # Add special tokens in the given order, set attributes like eos_id
        for x in ['pad', 'eos', 'start']:
            if getattr(self, f'use_{x}_token'):
                setattr(self, f'{x}_id', len(self.token_list))
                self.token_list.append(f"<{x}>")
            else:
                setattr(self, f'{x}_id', None)
        self.special_token_count = len(self.token_list)

        ltl_ops = self.ltl_ops
        trace_ops = self.trace_ops
        # Determine common and unique ops
        common_ops = [x for x in ltl_ops if x in trace_ops]
        ltl_ops = [x for x in ltl_ops if x not in common_ops]
        trace_ops = [x for x in trace_ops if x not in common_ops]
        # Add to token list
        common = self._add_tokens(self.consts + common_ops)
        self.trace_tokens = common | self._add_tokens(trace_ops)
        self.ltl_tokens = common | self._add_tokens(ltl_ops)

        # Add all lowercase alphabet characters
        aps = self._add_tokens([chr(i) for i in range(ord('a'), ord('z')+1)])
        self.ltl_tokens |= aps
        self.trace_tokens |= aps
        
        # Decode fix for implies and xor operators
        self.token_list[self.ltl_tokens['=']] = "<->"
        self.token_list[self.ltl_tokens['^']] = "xor"
    
    def _add_tokens(self, tokens):
        start_id = len(self.token_list)
        self.token_list += tokens
        return {token: start_id + i for i, token in enumerate(tokens)}
    
    def are_inputs_compatible(self, other):
        """
        SIDE EFFECT: Modifies aps on self.
        """
        self.aps = other.aps
        return set(other.aps).issubset(set(self.aps)) \
            and set(other.consts).issubset(set(self.consts)) \
            and set(other.trace_ops).issubset(set(self.trace_ops))

    def encode(self, trace, ltl):
        """
        Encode trace and LTL formula with EOS token.
        """
        raise NotImplementedError()
        trace = [self.trace_tokens[c] for c in trace]
        ltl = [self.ltl_tokens[c] for c in ltl]
        return trace + ltl + [self.eos_id]
    
    def _encode(self, text, tokens, eos):
        text = text.replace("<->", '=').replace("xor", "^")
        out = [tokens[c] for c in text]
        if eos:
            out.append(self.eos_id)
        return out

    def encode_trace(self, trace, eos=False):
        return self._encode(trace, self.trace_tokens, eos)

    def encode_ltl(self, ltl, eos=False):
        return self._encode(ltl, self.ltl_tokens, eos)

    def decode(self, ids):
        if self.start_id is not None and ids[0] == self.start_id:
            ids = ids[1:]
        try:
            ids = ids[:ids.index(self.eos_id)]
        except ValueError:
            pass
        return ''.join([self.token_list[i] for i in ids])

    def decode_split(self, ids):
        string = self.decode(ids)
        # '}' marks the end of trace
        point = string.index('}') + 1
        return string[:point], string[point:]

    def size(self):
        return len(self.token_list)

    def num_classes(self):
        return len(self.token_list) - 26 + len(self.aps)

    def ltl_size(self):
        return len(self.ltl_tokens) + self.special_token_count

    def trace_size(self):
        return len(self.trace_tokens) + self.special_token_count

    def force_eos(self, logits, i):
        logits[i, self.eos_id+1:] = -float('inf')
        if self.eos_id > 0:
            logits[i, :self.eos_id] = -float('inf')
    def disallow_eos(self, logits, i):
        logits[i, self.eos_id] = -float('inf')