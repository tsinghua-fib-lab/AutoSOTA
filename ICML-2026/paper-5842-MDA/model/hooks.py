# model/hooks.py

from transformer_lens import HookedTransformer


# ---------------- QK cache ----------------

class QKActivationCache:
    def __init__(self):
        self.X = None  # [B, S, d_model]
        self.Q = None  # [B, S, n_heads, d_head]
        self.K = None  # [B, S, n_heads, d_head]

    def clear(self):
        self.X = None
        self.Q = None
        self.K = None


def setup_qk_hooks(model: HookedTransformer, layer: int, cache: QKActivationCache, verbose: bool = False) -> list:
    """
    Install hooks to capture layer norm output, Q and K activations.
    Returns hook handle list; call model.reset_hooks(hooks) to clean up.
    """
    def h_ln(act, hook): cache.X = act; return act
    def h_q(act, hook):  cache.Q = act; return act
    def h_k(act, hook):  cache.K = act; return act

    hooks = [
        model.add_hook(f"blocks.{layer}.ln1.hook_normalized", h_ln, dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_q",         h_q,  dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_k",         h_k,  dir="fwd"),
    ]

    if verbose:
        print(f"QK hooks installed at layer {layer}")

    return [h for h in hooks if h is not None]


# ---------------- QKVO cache ----------------

class QKVOActivationCache:
    def __init__(self):
        self.X      = None  # [B, S, d_model]
        self.Q      = None  # [B, S, n_heads, d_head]
        self.K      = None  # [B, S, n_heads, d_head]
        self.V      = None  # [B, S, n_heads, d_head]
        self.Z      = None  # [B, S, n_heads, d_head]
        self.result = None  # [B, S, n_heads, d_model]

    def clear(self):
        self.X = None; self.Q = None; self.K = None
        self.V = None; self.Z = None; self.result = None


def setup_qkvo_hooks(model: HookedTransformer, layer: int, cache: QKVOActivationCache, verbose: bool = False) -> list:
    """
    Install hooks to capture layer norm output, Q, K, V, Z and result activations.
    Returns hook handle list; call model.reset_hooks(hooks) to clean up.
    """
    def h_ln(act, hook):     cache.X      = act; return act
    def h_q(act, hook):      cache.Q      = act; return act
    def h_k(act, hook):      cache.K      = act; return act
    def h_v(act, hook):      cache.V      = act; return act
    def h_z(act, hook):      cache.Z      = act; return act
    def h_result(act, hook): cache.result = act; return act

    hooks = [
        model.add_hook(f"blocks.{layer}.ln1.hook_normalized", h_ln,     dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_q",         h_q,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_k",         h_k,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_v",         h_v,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_z",         h_z,      dir="fwd"),
        model.add_hook(f"blocks.{layer}.attn.hook_result",    h_result, dir="fwd"),
    ]

    if verbose:
        print(f"QKVO hooks installed at layer {layer}")

    return [h for h in hooks if h is not None]


# ---------------- Pattern cache ----------------

class PatternCache:
    def __init__(self):
        self.pattern = None  # [B, n_heads, S, S]

    def clear(self):
        self.pattern = None


def setup_pattern_hooks(model: HookedTransformer, layer: int, cache: PatternCache, verbose: bool = False) -> list:
    """
    Install hook to capture post-softmax attention pattern.
    Returns hook handle list; call model.reset_hooks(hooks) to clean up.
    """
    def h_pattern(act, hook): cache.pattern = act; return act

    hooks = [
        model.add_hook(f"blocks.{layer}.attn.hook_pattern", h_pattern, dir="fwd"),
    ]

    if verbose:
        print(f"Pattern hook installed at layer {layer}")

    return [h for h in hooks if h is not None]
