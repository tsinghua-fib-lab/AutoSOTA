import torch
from primitives_classes import Primitive


class ConstAttentionPrimitives:
    def only_bos(shape_right, tokenizer):
        matrix = torch.zeros(shape_right)
        matrix[tokenizer.bos_token_id] = 1.
        return matrix
    def op_only_bos():
        return "k==BOS"
    def only_eos(shape_right, tokenizer):
        matrix = torch.zeros(shape_right)
        matrix[tokenizer.eos_token_id] = 1.
        return matrix
    def op_only_eos():
        return "k==EOS"
    def only_sep(shape_right, tokenizer):
        matrix = torch.zeros(shape_right)
        matrix[tokenizer.sep_token_id] = 1.
        return matrix
    def op_only_sep():
        return "k==SEP"
    def gradually_smaller(shape_right, tokenizer):
        matrix = (torch.arange(shape_right) + 1) / shape_right
        return matrix
    def op_gradually_smaller():
        return "k is first"
    def gradually_bigger(shape_right, tokenizer):
        matrix = (torch.arange(shape_right, 0, -1)) / shape_right
        return matrix
    def op_gradually_bigger():
        return "k is last"
    def uniform_zeros(shape_right, tokenizer):
        matrix = torch.zeros(shape_right)
        return matrix
    def op_uniform_zeros():
        return "uniform selection"
    
    has_default_scalar = {
        "uniform_zeros": True,
        "only_bos":  True,
        "only_eos":  True,
        "only_sep":  True,
        "gradually_smaller": True,
        "gradually_bigger": True,
    }

    primitives = [
        ("uniform_zeros", False),
        ("only_bos", True),
        ("only_eos", True),
        ("only_sep", True),
        ("gradually_smaller", False),
        ("gradually_bigger", False),
    ]

class AttentionPrimitives:
    def diag(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        matrix[range(min(shape_left, shape_right)), range(min(shape_left, shape_right))] = 1.
        return matrix
    def op_diag():
        return "k==q"
    def second_diag(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        matrix[torch.arange(1, min(shape_left, shape_right)), torch.arange(0, min(shape_left, shape_right) - 1)] = 1.
        return matrix
    def op_second_diag():
        return "k==q-1"
    def third_diag(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        matrix[torch.arange(3, min(shape_left, shape_right)), torch.arange(0, min(shape_left, shape_right) - 3)] = 1.
        return matrix
    def op_third_diag():
        return "k==q-2"
    def every_second(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        for x in range(min(shape_left, shape_right) // 2 + (min(shape_left, shape_right) % 2 != 0)):
            matrix[torch.arange(x * 2, min(shape_left, shape_right)), torch.arange(0, min(shape_left, shape_right) - x * 2)] = 1.
        return matrix
    def op_every_second():
        return "k%2==q%2==0"
    def every_third(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        for x in range(min(shape_left, shape_right) // 3 + (min(shape_left, shape_right) % 3 != 0)):
            matrix[torch.arange(x * 3, min(shape_left, shape_right)), torch.arange(0, min(shape_left, shape_right) - x * 3)] = 1.
        return matrix
    def op_every_third():
        return "k%3==q%3==0"
    def uniform_zeros(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        return matrix
    def op_uniform_zeros():
        return "uniform selection"
    def to_bos(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        matrix[:, tokenizer.bos_token_id] = 1.
        return matrix
    def op_to_bos():
        return "k==BOS"
    def to_sep(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        matrix[:, tokenizer.sep_token_id] = 1.
        return matrix
    def op_to_sep():
        return "k==SEP"
    def gradually_smaller(shape_left, shape_right, tokenizer):
        matrix = (torch.arange(shape_right) + 1).unsqueeze(0).expand(shape_left, shape_right) / shape_right
        return matrix
    def op_gradually_smaller():
        return "k is first"
    def gradually_bigger(shape_left, shape_right, tokenizer):
        matrix = (torch.arange(shape_right, 0, -1)).unsqueeze(0).expand(shape_left, shape_right) / shape_right
        return matrix
    def op_gradually_bigger():
        return "k is last"
    
    has_default_scalar = {
        "uniform_zeros": True,
        "to_bos":  True,
        "to_sep":  True,
        "diag":  True,
        "second_diag":  True,
        "third_diag": True,
        "every_second":  True,
        "every_third":  True,
        "gradually_smaller": True,
        "gradually_bigger": True,
    }

    primitives = [
        ("uniform_zeros", False),
        ("to_bos", True),
        ("to_sep", True),
        ("diag", False),
        ("second_diag", False),
        ("third_diag", False),
        ("every_second", False),
        ("every_third", False),
        ("gradually_smaller", False),
        ("gradually_bigger", False)
    ]


class ConstLogitsPrimitives:
    def uniform_zeros(shape_right, tokenizer):
        matrix = torch.zeros(shape_right)
        return matrix
    def op_uniform_zeros():
        return "uniform selection"
    def to_bos(shape_right, tokenizer):
        matrix = torch.zeros(shape_right)
        matrix[tokenizer.bos_token_id] = 1.
        return matrix
    def op_to_bos():
        return "out==BOS"
    def to_eos(shape_right, tokenizer):
        matrix = torch.zeros(shape_right)
        matrix[tokenizer.eos_token_id] = 1.
        return matrix
    def op_to_eos():
        return "out==EOS"
    
    has_default_scalar = {
        "uniform_zeros": True,
        "to_eos":  True,
    }

    primitives = [
        ("uniform_zeros", False),
        ("to_eos", True),
    ]


class LogitsPrimitives:
    def diag(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        matrix[range(min(shape_left, shape_right)), range(min(shape_left, shape_right))] = 1.
        return matrix
    def op_diag():
        return "inp==out"
    def uniform_zeros(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        return matrix
    def op_uniform_zeros():
        return "uniform selection"
    def to_bos(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        matrix[:, tokenizer.bos_token_id] = 1.
        return matrix
    def op_to_bos():
        return "out==BOS"
    def to_eos(shape_left, shape_right, tokenizer):
        matrix = torch.zeros(shape_left, shape_right)
        matrix[:, tokenizer.eos_token_id] = 1.
        return matrix
    def op_to_eos():
        return "out==EOS"
    
    has_default_scalar = {
        "uniform_zeros": True,
        "to_eos": True,
        "diag":  True,
    }

    primitives = [
        ("uniform_zeros", False),
        ("to_eos", True),
        ("diag", False),
    ]

ATTENTION_CONST_ALL_PRIMITIVES = [
    Primitive(
        name=name,
        is_only_token=is_only_token,
        contruct_matrix=getattr(ConstAttentionPrimitives, name),
        has_default_scalar=ConstAttentionPrimitives.has_default_scalar[name],
        operation=(getattr(ConstAttentionPrimitives, "op_" + name) if hasattr(ConstAttentionPrimitives, "op_" + name) else None),
    )
    for name, is_only_token in ConstAttentionPrimitives.primitives
]

ATTENTION_ALL_PRIMITIVES = [
    Primitive(
        name=name,
        is_only_token=is_only_token,
        contruct_matrix=getattr(AttentionPrimitives, name),
        has_default_scalar=AttentionPrimitives.has_default_scalar[name],
        operation=(getattr(AttentionPrimitives, "op_" + name) if hasattr(AttentionPrimitives, "op_" + name) else None),
    )
    for name, is_only_token in AttentionPrimitives.primitives
]

LOGITS_CONST_ALL_PRIMITIVES = [
    Primitive(
        name=name,
        is_only_token=is_only_token,
        contruct_matrix=getattr(ConstLogitsPrimitives, name),
        has_default_scalar=ConstLogitsPrimitives.has_default_scalar[name],
        operation=(getattr(ConstLogitsPrimitives, "op_" + name) if hasattr(ConstLogitsPrimitives, "op_" + name) else None),
    )
    for name, is_only_token in ConstLogitsPrimitives.primitives
]

LOGITS_ALL_PRIMITIVES = [
    Primitive(
        name=name,
        is_only_token=is_only_token,
        contruct_matrix=getattr(LogitsPrimitives, name),
        has_default_scalar=LogitsPrimitives.has_default_scalar[name],
        operation=(getattr(LogitsPrimitives, "op_" + name) if hasattr(LogitsPrimitives, "op_" + name) else None),
    )
    for name, is_only_token in LogitsPrimitives.primitives
]