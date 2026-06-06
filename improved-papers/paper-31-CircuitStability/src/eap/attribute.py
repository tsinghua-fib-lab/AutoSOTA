from typing import Callable, List, Union, Optional, Literal
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from tqdm import tqdm
from einops import einsum

from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode


def make_hooks_and_matrices(
    model: HookedTransformer,
    graph: Graph,
    batch_size: int,
    n_pos: int,
    scores: torch.Tensor,
    detach=True,
):
    """Makes a matrix containing activation differences, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, i.e. corrupted activation - clean activation.
        The first set of hooks will add in the activations they are run on (run these on corrupted input), while the second set will subtract out the activations they are run on (run these on clean input).
        The third set of hooks will take in the gradients during the backwards pass and multiply it by the activation differences, adding this value in-place to the scores matrix that you passed in.
    """
    activation_difference = torch.zeros(
        (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
        dtype=model.cfg.dtype,
        device="cpu",
        pin_memory=True,
    )

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []

    def activation_hook(index, activations, hook, add: bool = True):
        """Hook to add/subtract activations to/from the activation difference matrix
        Args:
            index ([type]): forward index of the node
            activations ([type]): activations to add
            hook ([type]): hook (unused)
            add (bool, optional): whether to add or subtract. Defaults to True."""
        acts = activations.detach() if detach else activations
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts.to(activation_difference.device)
        except RuntimeError as e:
            print(
                "Activation Hook Error",
                hook.name,
                activation_difference[:, :, index].size(),
                acts.size(),
                index,
            )
            raise e

    def gradient_hook(
        prev_index: Union[slice, int],
        bwd_index: Union[slice, int],
        gradients: torch.Tensor,
        hook,
    ):
        """Hook to multiply the gradients by the activations and add them to the scores matrix

        Args:
            prev_index (Union[slice, int]): index before which all nodes contribute to the present node
            bwd_index (Union[slice, int]): backward pass index of the node
            gradients (torch.Tensor): gradients of the node
            hook ([type]): hook
        """
        grads = gradients.detach().cpu()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)

            s = einsum(
                activation_difference[:, :, :prev_index],
                grads.to(activation_difference.device),
                "batch pos forward hidden, batch pos backward hidden -> forward backward",
            )
            s = s.squeeze(1)  # .to(scores.device)
            scores[:prev_index, bwd_index] += s.to(scores.device)
        except RuntimeError as e:
            print(
                "Gradient Hook Error",
                hook.name,
                activation_difference.size(),
                grads.size(),
                prev_index,
                bwd_index,
            )
            raise e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        if not isinstance(node, LogitNode):
            fwd_index = graph.forward_index(node)
            fwd_hooks_corrupted.append(
                (node.out_hook, partial(activation_hook, fwd_index))
            )
            fwd_hooks_clean.append(
                (node.out_hook, partial(activation_hook, fwd_index, add=False))
            )
        if not isinstance(node, InputNode):
            prev_index = graph.prev_index(node)
            if isinstance(node, AttentionNode):
                for i, letter in enumerate("qkv"):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append(
                        (
                            node.qkv_inputs[i],
                            partial(gradient_hook, prev_index, bwd_index),
                        )
                    )
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append(
                    (node.in_hook, partial(gradient_hook, prev_index, bwd_index))
                )

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def tokenize_plus(model: HookedTransformer, inputs: List[str]):
    """
    Tokenizes the input strings using the provided model.

    Args:
        model (HookedTransformer): The model used for tokenization.
        inputs (List[str]): The list of input strings to be tokenized.

    Returns:
        tuple: A tuple containing the following elements:
            - tokens (torch.Tensor): The tokenized inputs.
            - attention_mask (torch.Tensor): The attention mask for the tokenized inputs.
            - input_lengths (torch.Tensor): The lengths of the tokenized inputs.
            - n_pos (int): The maximum sequence length of the tokenized inputs.
    """
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side="left")
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos


def get_scores_eap(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    quiet=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device="cuda", dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, n_pos = clean
        corrupted_tokens, _, _, _ = corrupted

        batch_size = len(clean_tokens)
        total_items += batch_size

        (
            fwd_hooks_corrupted,
            fwd_hooks_clean,
            bwd_hooks,
        ), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
        )

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, corrupted_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores


def get_scores_eap_ig(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps=30,
    quiet=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device="cpu", dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, n_pos = clean
        corrupted_tokens, _, _, _ = corrupted
        batch_size = len(clean_tokens)
        total_items += batch_size

        (
            fwd_hooks_corrupted,
            fwd_hooks_clean,
            bwd_hooks,
        ), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[
                :, :, graph.forward_index(graph.nodes["input"])
            ].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = (
                input_activations_corrupted
                - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]
            )

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (
                    input_activations_clean - input_activations_corrupted
                )
                new_input.requires_grad = True
                return new_input.to(activations.device)

            return hook_fn

        total_steps = 0
        for step in range(1, steps + 1):
            total_steps += 1
            with model.hooks(
                fwd_hooks=[
                    (graph.nodes["input"].out_hook, input_interpolation_hook(step))
                ],
                bwd_hooks=bwd_hooks,
            ):
                clean_tokens = clean_tokens.to(model.cfg.device)
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward(retain_graph=False)
                # Free memory immediately after backward pass
        del metric_value, logits, clean_tokens
        torch.cuda.empty_cache()

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_ig_partial_activations(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps=30,
    quiet=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device="cuda", dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, n_pos = clean
        corrupted_tokens, _, _, _ = corrupted

        batch_size = len(clean_tokens)
        total_items += batch_size

        (
            fwd_hooks_corrupted,
            fwd_hooks_clean,
            bwd_hooks,
        ), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        def output_interpolation_hook(k: int, difference: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                new_output = activations + (1 - k / steps) * difference
                return new_output

            return hook_fn

        total_steps = 0
        for step in range(1, steps + 1):
            total_steps += 1
            with model.hooks(
                fwd_hooks=[
                    (
                        node.out_hook,
                        output_interpolation_hook(
                            step, activation_difference[:, :, graph.forward_index(node)]
                        ),
                    )
                    for node in graph.nodes.values()
                    if not isinstance(node, LogitNode)
                ],
                bwd_hooks=bwd_hooks,
            ):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_ig_activations(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps=30,
    quiet=False,
    ablate_all_at_once=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device="cuda", dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, n_pos = clean
        corrupted_tokens, _, _, _ = corrupted

        batch_size = len(clean_tokens)
        total_items += batch_size

        detach = bool(ablate_all_at_once)
        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores, detach=detach
        )
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores, detach=detach
        )
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores, detach=detach
        )

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        activation_difference += (
            activations_corrupted.clone().detach() - activations_clean.clone().detach()
        )

        def output_interpolation_hook(
            k: int, clean: torch.Tensor, corrupted: torch.Tensor
        ):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k / steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output

            return hook_fn

        total_steps = 0

        nodeslist = [[graph.nodes["input"]]]
        for layer in range(graph.cfg["n_layers"]):
            nodeslist.append(
                [
                    graph.nodes[f"a{layer}.h{head}"]
                    for head in range(graph.cfg["n_heads"])
                ]
            )
            nodeslist.append([graph.nodes[f"m{layer}"]])

        if ablate_all_at_once:
            nodeslist = [
                node for node in graph.nodes.values() if not isinstance(node, LogitNode)
            ]

        for nodes in nodeslist:
            for step in range(1, steps + 1):
                total_steps += 1
                fwd_hooks = []
                for node in nodes:
                    clean_acts = activations_clean[:, :, graph.forward_index(node)]
                    corrupted_acts = activations_corrupted[
                        :, :, graph.forward_index(node)
                    ]
                    fwd_hooks.append(
                        (
                            node.out_hook,
                            output_interpolation_hook(step, clean_acts, corrupted_acts),
                        )
                    )

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_clean_corrupted(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    quiet=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device="cuda", dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:

        clean_tokens, attention_mask, input_lengths, n_pos = clean
        corrupted_tokens, _, _, _ = corrupted

        batch_size = len(clean_tokens)
        total_items += batch_size

        (
            fwd_hooks_corrupted,
            fwd_hooks_clean,
            bwd_hooks,
        ), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            # model.zero_grad()

            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            corrupted_metric_value = metric(
                corrupted_logits, clean_logits, input_lengths, label
            )
            corrupted_metric_value.backward()
            # model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores


allowed_aggregations = {"sum", "mean", "l2"}


def attribute(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    aggregation="sum",
    method: Union[
        Literal[
            "EAP",
            "EAP-IG",
            "EAP-IG-partial-activations",
            "EAP-IG-activations",
            "clean-corrupted",
        ]
    ] = "EAP-IG",
    ig_steps: Optional[int] = 5,
    quiet=False,
):
    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"aggregation must be in {allowed_aggregations}, but got {aggregation}"
        )

    if method != "EAP" and method != "clean-corrupted":
        assert ig_steps is not None, f"ig_steps must be set for method {method}"

    if method == "EAP":
        scores = get_scores_eap(model, graph, dataloader, metric, quiet=quiet)
    elif method == "EAP-IG":
        scores = get_scores_eap_ig(
            model, graph, dataloader, metric, steps=ig_steps, quiet=quiet
        )
    elif method == "EAP-IG-partial-activations":
        scores = get_scores_ig_partial_activations(
            model, graph, dataloader, metric, steps=ig_steps, quiet=quiet
        )
    elif method == "EAP-IG-activations":
        scores = get_scores_ig_activations(
            model, graph, dataloader, metric, steps=ig_steps, quiet=quiet
        )
    elif method == "clean-corrupted":
        scores = get_scores_clean_corrupted(
            model, graph, dataloader, metric, quiet=quiet
        )
    else:
        raise ValueError(
            f"integrated_gradients must be in ['EAP', 'EAP-IG', 'EAP-IG-partial-activations', 'EAP-IG-activations', 'clean-corrupted'], but got {method}"
        )

    if aggregation == "mean":
        scores /= model.cfg.d_model
    elif aggregation == "l2":
        scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)

    scores = scores.float().cpu().numpy()

    print(scores.shape)
    print(len(graph.edges.values()))

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[
            graph.forward_index(edge.parent, attn_slice=False),
            graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
        ]
