"""Notebook helpers used by ``notebooks/progressive_cramming_demo.ipynb``.

The gallery sections of the demo notebook need three small functions for the
embedding-to-text round trip + a coloured token-level diff. Keeping them here
(rather than as a notebook cell) means:

- The notebook stays nearly free of utility code -- only what the reader has to
  understand inline lives in cells.
- We can update visuals (e.g. tweak diff colours, switch to raw-id comparison)
  without re-shipping the notebook itself.

The helpers are deliberately minimal: they assume an already-loaded frozen
``model`` + ``tokenizer`` (cell 4 in the notebook) and a row from the demo
gallery dataset (the schema produced by ``scripts/build_demo_gallery.py``).
"""

from __future__ import annotations

import contextlib
import html
import io
import logging
import sys
import warnings

import torch
from IPython.display import HTML, display

from ._core import reconstruct_text


# Silence the "Ignoring clean_up_tokenization_spaces=True for BPE tokenizer" advisory
# that fires on every tokenizer call in transformers 5.x. We always pass
# ``clean_up_tokenization_spaces=False`` ourselves, so the message is non-actionable
# noise. We hit it from three angles because transformers 5.x can emit it via any
# of Python's ``logging``, ``warnings``, or a direct ``stderr`` print from the Rust
# tokenizers backend depending on which code path decoded the tokens.
class _SuppressBPECleanupWarning(logging.Filter):
    def filter(self, record):
        return "clean_up_tokenization_spaces" not in record.getMessage()


for _logger_name in (
    "transformers",
    "transformers.tokenization_utils_base",
    "transformers.tokenization_utils_fast",
):
    logging.getLogger(_logger_name).addFilter(_SuppressBPECleanupWarning())

# Blanket-lower the transformers logger to ERROR: keeps genuine errors, drops the
# ``[transformers] Ignoring ...`` advisory that ships via transformers.logging.
try:
    import transformers.utils.logging as _tf_logging
    _tf_logging.set_verbosity_error()
except Exception:  # noqa: BLE001 -- best-effort; transformers is a hard dep anyway.
    pass


class _StderrFilter(io.TextIOBase):
    """Redirect around a token'iser call: drop lines matching ``pattern``, forward the rest."""

    def __init__(self, real, pattern: str):
        self._real = real
        self._pattern = pattern
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if self._pattern not in line:
                self._real.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buf and self._pattern not in self._buf:
            self._real.write(self._buf)
        self._buf = ""
        self._real.flush()


@contextlib.contextmanager
def _quiet_tokenizer():
    """Suppress the BPE cleanup advisory across all three emission paths."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
        real_stderr = sys.stderr
        sys.stderr = _StderrFilter(real_stderr, "clean_up_tokenization_spaces")
        try:
            yield
        finally:
            try:
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass
            sys.stderr = real_stderr

# Token-diff colours -- foreground only, NOT background, so the diff stays
# readable on both light and dark Colab themes (background fills wash the
# foreground text in dark mode).
COLOR_MATCH = "#2ea043"     # GitHub-ish green; passes contrast on white and on #0d1117
COLOR_MISMATCH = "#d44a2c"  # HSE coral
COLOR_PAST_GT = "#888888"   # neutral grey for free continuation past the GT span


def emb_from_row(row) -> torch.Tensor:
    """Dataset row -> compression embedding tensor ``[n_cram, hidden]`` (float32).

    The embedding is stored in the Hub dataset as a 2D nested list of float32;
    ``torch.tensor`` rebuilds the correct shape without extra reshaping.
    """
    return torch.tensor(row["embedding"], dtype=torch.float32)


def _strip_bos(ids, tokenizer) -> list[int]:
    """Drop the BOS token from a generated/ground-truth id sequence if present.

    Llama-3.2 tokenizes with a leading BOS; SmolLM2 doesn't. ``strip_bos`` is a
    no-op for tokenizers without a BOS id, so callers can apply it unconditionally.
    """
    bos = tokenizer.bos_token_id
    ids = list(ids)
    if bos is not None and ids and ids[0] == bos:
        ids = ids[1:]
    return ids


def render_token_diff(gt_ids, gen_ids, title: str = "", *, tokenizer) -> str:
    """Build an HTML string colouring each generated token by whether it matches GT.

    Foreground-only (no background fills), so legible on dark Colab themes.
    Real newlines from the tokens are preserved verbatim and the container uses
    ``white-space: pre-wrap``, so multi-line code/poetry samples render across
    real lines instead of being smashed onto one line.
    """
    g = _strip_bos(gt_ids, tokenizer)
    h = list(gen_ids)
    spans = []
    for i in range(max(len(g), len(h))):
        a = g[i] if i < len(g) else None
        b = h[i] if i < len(h) else None
        piece = tokenizer.decode([b], clean_up_tokenization_spaces=False) if b is not None else ""
        if i >= len(g):
            color = COLOR_PAST_GT
        elif a is not None and b is not None and a == b:
            color = COLOR_MATCH
        else:
            color = COLOR_MISMATCH
        # html.escape but keep newlines so white-space:pre-wrap can break them.
        disp = html.escape(piece)
        spans.append(
            f'<span style="color:{color};font-weight:600">{disp}</span>'
        )
    body = "".join(spans)
    return (
        f'<div style="font-family:monospace;line-height:1.55;margin:6px 0;'
        f'white-space:pre-wrap;word-break:break-word">'
        f'<b style="color:inherit">{html.escape(title)}</b>\n{body}</div>'
    )


def reconstruct_and_show(
    row, label: str, *,
    model, tokenizer,
    extra_tokens: int = 0,
    stream: bool = False,
    delay_ms: int = 25,
) -> None:
    """Greedy-decode from a row's embedding and render the coloured diff inline.

    Generates ``row["num_tokens"] + extra_tokens`` tokens. The base ``num_tokens``
    is the span the embedding was trained on; tokens past it appear in grey as
    "free continuation".

    ``stream=True`` runs the greedy loop token-by-token and refreshes the diff
    HTML after every new token (with a ``delay_ms``-millisecond pause between
    frames so the eye can track it). ``stream=False`` (default) runs the whole
    generation in one call for the metrics-focused code paths.

    ``model`` + ``tokenizer`` are passed explicitly so the same helper works
    across §3 / §4 / §5.
    """
    # Clip GT to the actual trained span -- raw input_ids carry pad tokens out to
    # max_sequence_length, which would otherwise count as mismatches and turn
    # the "free continuation" tail red instead of grey.
    gt_ids = list(row["input_ids"][: row["num_tokens"]])
    total_new_tokens = row["num_tokens"] + extra_tokens

    if not stream:
        with _quiet_tokenizer():
            gen = reconstruct_text(
                model, tokenizer, emb_from_row(row),
                max_new_tokens=total_new_tokens,
            )
            gen_ids = tokenizer(gen, add_special_tokens=False)["input_ids"]
        display(HTML(render_token_diff(gt_ids, gen_ids, label, tokenizer=tokenizer)))
        return

    # Streaming path: custom greedy loop that yields after every token.
    import time

    import ipywidgets as widgets
    from IPython.display import clear_output

    emb = emb_from_row(row)
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)  # [1, num_mem, hidden]
    device = next(model.parameters()).device
    emb = emb.to(device)
    input_emb_layer = model.get_input_embeddings()
    torch_dtype = input_emb_layer.weight.dtype
    hidden_size = emb.size(-1)

    stream_out = widgets.Output()
    display(stream_out)
    gen_ids_tensor = torch.empty((1, 0), dtype=torch.long, device=device)

    with torch.no_grad(), _quiet_tokenizer():
        for _ in range(total_new_tokens):
            if gen_ids_tensor.size(1) == 0:
                gen_embs = torch.empty((1, 0, hidden_size), device=device, dtype=torch_dtype)
            else:
                gen_embs = input_emb_layer(gen_ids_tensor).to(torch_dtype)
            united = torch.cat([emb.to(torch_dtype), gen_embs], dim=1)
            mask = torch.ones(united.shape[:2], dtype=torch.long, device=device)
            out = model(inputs_embeds=united, attention_mask=mask)
            next_id = out.logits[:, -1, :].argmax(dim=-1)
            gen_ids_tensor = torch.cat([gen_ids_tensor, next_id.unsqueeze(-1)], dim=-1)

            gen_ids_partial = _strip_bos(gen_ids_tensor[0].cpu().tolist(), tokenizer)
            with stream_out:
                clear_output(wait=True)
                display(HTML(render_token_diff(
                    gt_ids, gen_ids_partial, label, tokenizer=tokenizer,
                )))
            time.sleep(delay_ms / 1000.0)


# ─────────────────────────────────────────────────────────────────────────────
# High-level widgets -- what notebook cells §3 and §4 ultimately call.
# ─────────────────────────────────────────────────────────────────────────────


def _card(*children):
    """Standard card box used by both §3 and §4."""
    import ipywidgets as widgets
    return widgets.VBox(
        children,
        layout=widgets.Layout(border="1px solid #555", padding="10px", margin="6px 0"),
    )


def _original_block(text: str) -> str:
    """Render the full original passage with newlines preserved and dark-mode-safe colours."""
    return (
        f'<div style="font-family:monospace;font-size:0.92em;color:#888;'
        f'white-space:pre-wrap;word-break:break-word;margin-top:4px">'
        f'{html.escape(text)}</div>'
    )


def display_gallery(rows, *, model, tokenizer) -> None:
    """Render the §3 gallery: one card per ``kind=="gallery"`` row, each with a
    ▶ Reconstruct button that greedy-decodes from the saved compression embedding.

    ``rows`` is a list of dataset rows (already filtered by ``kind == "gallery"``
    in the notebook -- keeps the dataset-loading logic visible to the reader).
    The frozen ``model`` / ``tokenizer`` must match every row's
    ``model_checkpoint`` (the canonical demo uses Llama-3.2-1B for all gallery
    rows).
    """
    import ipywidgets as widgets

    legend = (
        f'<span style="color:{COLOR_MATCH};font-weight:600">green = match</span>'
        f' &middot; '
        f'<span style="color:{COLOR_MISMATCH};font-weight:600">red = mismatch</span>'
        f' &middot; '
        f'<span style="color:{COLOR_PAST_GT};font-weight:600">grey = past the original span</span>'
    )

    def make_card(row):
        out = widgets.Output()
        btn = widgets.Button(description="▶ Reconstruct", button_style="primary")
        header = widgets.HTML(
            f"<b>{html.escape(row['domain'])}</b> &mdash; {html.escape(row['title'])}"
            f"{_original_block(row['text'])}"
        )

        def on_click(_):
            out.clear_output()
            with out:
                display(HTML(
                    f"<div style='font-size:0.9em;color:#888;margin-bottom:4px'>{legend}</div>"
                ))
                # Show ~10% extra tokens past the compression horizon so the reader
                # sees the model's free continuation past the trained span (in grey).
                extra = max(1, row["num_tokens"] // 5)
                reconstruct_and_show(
                    row, "Reconstruction",
                    model=model, tokenizer=tokenizer, extra_tokens=extra,
                    stream=True,
                )
                n_cram_label = (
                    "Single compression embedding"
                    if row["n_cram"] == 1
                    else f"{row['n_cram']} compression embeddings"
                )
                display(HTML(
                    f"<div style='margin-top:4px;font-size:0.9em;color:#888'>"
                    f"{n_cram_label} &rarr; {row['num_tokens']} reconstructed tokens</div>"
                ))

        btn.on_click(on_click)
        return _card(header, btn, out)

    display(widgets.VBox([make_card(r) for r in rows]))


def _passage_label(row) -> str:
    """Strip the trailing method suffix from a tc_pc row's title.

    Accepts both the new ``' — full cramming'`` and the legacy ``' — total cramming'``
    naming so datasets already published on the Hub keep working after the rename.
    """
    title = row["title"]
    for tail in (
        " — full cramming",
        " — total cramming",  # legacy, kept for backward-compat with published rows
        " — progressive cramming",
    ):
        if title.endswith(tail):
            return title[: -len(tail)]
    return title


def display_side_by_side(rows, *, models: dict) -> None:
    """Render the §4 side-by-side: one card per (model_checkpoint, sample) pair.

    Visually mirrors :func:`display_gallery` (same card layout, same legend, same
    colour scheme). Each card pairs the TC + PC rows for one ``model_checkpoint``
    and runs them through ``reconstruct_and_show`` on click.

    ``models`` is a ``{checkpoint: (model, tokenizer)}`` mapping built by the
    notebook -- one entry per frozen model the gallery references. The function
    NEVER loads a model itself; if a pair's checkpoint is missing from ``models``
    we render a placeholder card instead of crashing or printing diagnostics.
    """
    from collections import defaultdict

    import ipywidgets as widgets

    pairs_by_model: dict[str, dict] = defaultdict(dict)
    for r in rows:
        pairs_by_model[r["model_checkpoint"]][r["method"]] = r

    legend = (
        f'<span style="color:{COLOR_MATCH};font-weight:600">green = match</span>'
        f' &middot; '
        f'<span style="color:{COLOR_MISMATCH};font-weight:600">red = mismatch</span>'
        f' &middot; '
        f'<span style="color:{COLOR_PAST_GT};font-weight:600">grey = past the original span</span>'
    )

    def make_card(ckpt: str, pair: dict):
        tc_row = pair["full_cramming"]
        pc_row = pair["progressive_cramming"]
        short_ckpt = ckpt.split("/", 1)[-1]
        out = widgets.Output()

        # Header mirrors §3: domain + passage title, then short meta line, then
        # the full original text as a monospace pre-wrap block.
        header = widgets.HTML(
            f"<b>{html.escape(tc_row['domain'])}</b> &mdash; "
            f"{html.escape(_passage_label(tc_row))}"
            f"<div style='font-size:0.85em;color:#888;margin-top:2px'>"
            f"<code>{html.escape(ckpt)}</code> &middot; "
            f"{tc_row['num_tokens']} tokens &middot; "
            f"PC horizon <b>{pc_row['horizon']}/{pc_row['num_tokens']}</b></div>"
            f"{_original_block(tc_row['text'])}"
        )

        if ckpt not in models:
            placeholder = widgets.HTML(
                f"<div style='color:{COLOR_MISMATCH};font-size:0.9em;margin-top:6px'>"
                f"Model <code>{html.escape(ckpt)}</code> not loaded. "
                f"Add it to <code>models=</code> in the cell above to enable this pair."
                f"</div>"
            )
            return _card(header, placeholder)

        m, t = models[ckpt]
        btn = widgets.Button(
            description=f"▶ Run side-by-side ({short_ckpt})",
            button_style="primary",
        )

        def on_click(_):
            out.clear_output()
            with out:
                display(HTML(
                    f"<div style='font-size:0.9em;color:#888;margin-bottom:4px'>{legend}</div>"
                ))
                # ~10% extra tokens past the trained span -> visible grey continuation.
                extra = max(1, tc_row["num_tokens"] // 5)
                reconstruct_and_show(
                    tc_row, "Full cramming — whole span at once",
                    model=m, tokenizer=t, extra_tokens=extra,
                    stream=True,
                )
                reconstruct_and_show(
                    pc_row, "Progressive cramming — grown to the horizon",
                    model=m, tokenizer=t, extra_tokens=extra,
                    stream=True,
                )
                display(HTML(
                    f"<div style='margin-top:4px;font-size:0.9em;color:#888'>"
                    f"Full Cramming teacher-forced accuracy="
                    f"<b>{tc_row['final_convergence']:.3f}</b></div>"
                ))

        btn.on_click(on_click)
        return _card(header, btn, out)

    display(widgets.VBox([make_card(ckpt, p) for ckpt, p in pairs_by_model.items()]))


# ─────────────────────────────────────────────────────────────────────────────
# §5 -- interactive progressive cramming with optimisation curves (live) +
# accuracy landscape on PC1-PC2 (after run completes)
# ─────────────────────────────────────────────────────────────────────────────


def _stage_palette(n: int, plt):
    """Return an ``n``-colour palette matching the paper's ``rocket_r`` ramp.

    Tries seaborn first (which registers the ``rocket_r`` colormap in matplotlib);
    if seaborn is not installed / import fails, falls back to a hand-picked pink→
    purple ramp that reproduces the paper's look reasonably closely without any
    external dependency. As a last resort, uses matplotlib's built-in ``plasma``.
    """
    import numpy as np

    n = max(int(n), 1)
    try:
        import seaborn as _sns
        cols = np.array(_sns.color_palette("rocket_r", n_colors=n))
        if cols.shape[1] == 3:
            cols = np.concatenate([cols, np.ones((cols.shape[0], 1))], axis=1)
        return cols
    except Exception:
        pass
    try:
        from matplotlib.colors import LinearSegmentedColormap
        # Approximation of seaborn rocket_r (pink → deep purple).
        cmap = LinearSegmentedColormap.from_list(
            "rocket_r_approx",
            ["#f6d5c2", "#ee8f81", "#c6497b", "#7a1f68", "#2e0a48"],
        )
        return cmap(np.linspace(0.15, 0.9, n))
    except Exception:
        return plt.get_cmap("plasma")(np.linspace(0.08, 0.92, n))


class _ProgressiveLiveViz:
    """Live-updating optimisation dashboard for the §5 progressive cramming widget.

    Left panel  : loss + teacher-forced reconstruction curves.
    Right panel : PC1-PC2 accuracy landscape reveal, mirroring the paper's
        ``visual_abstract_trajectory_zoom_progressive`` composite. Each PC stage
        is scored *at the moment it converges*: we snapshot the frozen embedding
        space (PCA is fit once, on the first ``pca_freeze_after`` snapshots, and
        then held fixed), compute the accuracy grid using the just-completed
        stage's prefix, and add its ``accuracy > threshold`` region to the
        cached overlay. Trajectory line + current cursor overlay on top.

    Model, ``num_mem_tokens``, ``hidden_size``, and the full tokenised
    ``input_ids`` arrive via ``on_step`` payload -- no external configuration
    needed. If ``on_step`` never fires (i.e. progressive_cram_text exited early),
    the dashboard just draws the curves.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        redraw_every: int = 20,
        grid_size: int = 20,
        threshold: float = 0.9,
        pca_padding: float = 0.2,
        pca_freeze_after: int = 24,
        accuracy_batch_size: int = 8,
        region_seq_len_stride: int = 1,
        first_seq_len: int | None = None,
        min_region_seq_len: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        # For long spans, accuracy on very short prefixes (seq_len <= a few
        # tokens) is trivially high nearly everywhere in the PCA grid -- the
        # resulting region blankets the whole plane and buries the meaningful
        # islands. We simply skip stages whose seq_len is below this floor.
        self.min_region_seq_len = int(min_region_seq_len)
        self.redraw_every = redraw_every
        self.grid_size = grid_size
        self.threshold = threshold
        self.pca_padding = pca_padding
        self.pca_freeze_after = pca_freeze_after
        self.accuracy_batch_size = accuracy_batch_size
        # For paper-style "few anchors" rendering: skip regions whose seq_len
        # doesn't hit the stride. ``first_seq_len`` (the min_seq_len used by
        # progressive_cram_text) is always shown regardless of the stride, so
        # the initial stage doesn't get skipped when stride > 1.
        self.region_seq_len_stride = max(int(region_seq_len_stride), 1)
        self.first_seq_len = first_seq_len

        # Rolling per-step metrics.
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.convs: list[float] = []
        self.snaps: list = []
        self.snap_seqlens: list[int] = []
        self._since_redraw = 0

        # Payload snapshot -- captured from the first ``on_step`` call.
        self._input_ids: list[int] | None = None
        self._num_mem_tokens: int | None = None
        self._hidden_size: int | None = None

        # Frozen PCA landscape.
        self._pca = None
        self._pca_XX = None
        self._pca_YY = None
        self._pca_grid_t = None  # torch.Tensor of inverse-projected grid embeddings

        # Cached per-stage accuracy regions.
        self._cached: list[dict] = []  # each: {"Z": ndarray, "seq_len": int}
        self._last_stage_index: int = -1

    # ─── on_step callback ────────────────────────────────────────────────
    def __call__(self, info: dict) -> None:
        # Purely accumulative: we don't fit PCA or score accuracy during the run
        # (see class docstring for why this is honest).
        self.steps.append(info["global_step"])
        self.losses.append(info["loss"])
        self.convs.append(info["convergence"])
        emb_flat = info["embedding"].detach().reshape(-1).to(torch.float32).cpu().numpy()
        self.snaps.append(emb_flat)
        self.snap_seqlens.append(info["seq_len"])

        if self._input_ids is None:
            self._input_ids = list(info["input_ids"])
            self._num_mem_tokens = int(info["num_mem_tokens"])
            self._hidden_size = int(info["hidden_size"])

        self._last_stage_index = int(info["stage_index"])
        self._since_redraw += 1
        if self._since_redraw >= self.redraw_every:
            self._since_redraw = 0
            self.draw()

    # ─── explicit finalisers ─────────────────────────────────────────────
    def finalize(self) -> None:
        """Freeze PCA on the FULL trajectory + compute per-anchor local accuracy
        regions (paper style: ``visualize_landscale_2pca.py --radius``).

        For each stride-aligned stage (+ the horizon), we build a *small* PC1-PC2
        grid **centred on that stage's anchor** (the last snapshot whose seq_len
        was this stage). The grid radius is a fraction of the median inter-anchor
        distance -- exactly what makes the paper's islands compact rather than
        flooding the whole plane.
        """
        import numpy as np
        from sklearn.decomposition import PCA

        if len(self.snaps) < 3 or self._input_ids is None:
            return

        stacked = np.stack(self.snaps)
        self._pca = PCA(n_components=2).fit(stacked)
        coords_all = self._pca.transform(stacked)  # [N, 2]

        # Anchor = last snapshot of each seq_len (== convergence point of that
        # stage, right before the warm-start jump into the next).
        anchors_by_seq_len: dict[int, tuple[float, float, int]] = {}
        for i in range(len(self.snap_seqlens)):
            sl = int(self.snap_seqlens[i])
            anchors_by_seq_len[sl] = (float(coords_all[i, 0]), float(coords_all[i, 1]), i)
        current_seq_len = int(self.snap_seqlens[-1])

        # Decide which anchors to render.
        seq_lens_sorted = sorted(anchors_by_seq_len.keys())
        to_render = [sl for sl in seq_lens_sorted
                     if self._should_render_region(sl) and sl != current_seq_len]
        # Horizon is always drawn.
        if current_seq_len not in to_render:
            to_render.append(current_seq_len)

        # Local radius: 0.5 * median distance between consecutive rendered anchors.
        if len(to_render) >= 2:
            xy = np.array([anchors_by_seq_len[sl][:2] for sl in to_render])
            dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            local_radius = float(0.5 * np.median(dists[dists > 0])) if (dists > 0).any() else 1.0
        else:
            span_x = float(coords_all[:, 0].max() - coords_all[:, 0].min())
            local_radius = max(0.08 * span_x, 0.5)

        # Global extent for the animation viewport = union of all local grids
        # (plus trajectory), so nothing pops out mid-frame.
        min_x = float(coords_all[:, 0].min())
        max_x = float(coords_all[:, 0].max())
        min_y = float(coords_all[:, 1].min())
        max_y = float(coords_all[:, 1].max())
        device = next(self.model.parameters()).device

        for sl in to_render:
            ax_x, ax_y, _snap_idx = anchors_by_seq_len[sl]
            x_grid = np.linspace(ax_x - local_radius, ax_x + local_radius, self.grid_size)
            y_grid = np.linspace(ax_y - local_radius, ax_y + local_radius, self.grid_size)
            XX_local, YY_local = np.meshgrid(x_grid, y_grid)
            grid_xy = np.stack([XX_local.ravel(), YY_local.ravel()], axis=1)
            grid_embeds = self._pca.inverse_transform(grid_xy).astype(np.float32)
            grid_t = torch.tensor(grid_embeds, dtype=torch.float32)

            input_ids = torch.tensor([self._input_ids[:sl]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            text_embeds = self.model.get_input_embeddings()(input_ids)
            Z = _accuracy_batch(
                self.model,
                compression_flat=grid_t,
                mem_shape=(self._num_mem_tokens, self._hidden_size),
                input_ids=input_ids,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
                batch_size=self.accuracy_batch_size,
            ).reshape(XX_local.shape)

            self._cached.append({
                "Z": Z,
                "XX": XX_local, "YY": YY_local,
                "anchor_xy": (ax_x, ax_y),
                "seq_len": int(sl),
            })
            # Expand global extent to include this local grid.
            min_x = min(min_x, float(XX_local.min()))
            max_x = max(max_x, float(XX_local.max()))
            min_y = min(min_y, float(YY_local.min()))
            max_y = max(max_y, float(YY_local.max()))

        # Store viewport extent for animation.
        self._pca_XX = np.array([[min_x, max_x], [min_x, max_x]])
        self._pca_YY = np.array([[min_y, min_y], [max_y, max_y]])

    def _should_render_region(self, seq_len: int, *, force: bool = False) -> bool:
        """Whether we materialise a region for this ``seq_len``.

        Skips seq_lens that don't hit ``region_seq_len_stride`` so the paper's
        "few anchors" look is preserved for long spans.
        """
        if force:
            return True
        if seq_len < self.min_region_seq_len:
            return False
        return (seq_len % self.region_seq_len_stride) == 0

    def _coords_now(self):
        """Project all snapshots into the frozen PCA plane (if fit)."""
        import numpy as np

        if self._pca is None:
            return None
        return self._pca.transform(np.stack(self.snaps))

    # ─── rendering ──────────────────────────────────────────────────────
    def draw(self) -> None:
        """Live mode: render the growing compressed prefix.

        Tokens up to the current stage's ``seq_len`` are green (successfully
        compressed so far); the rest are grey (still to be compressed).
        The paper-style PC1-PC2 landscape is not drawn here -- it's rendered
        as a separate animated figure after :meth:`finalize` collects the
        accuracy regions on the finalised PCA plane.
        """
        from IPython.display import clear_output

        clear_output(wait=True)
        if self._input_ids is None or not self.snap_seqlens:
            display(HTML(
                "<div style='font-family:monospace;color:#888'>"
                "Waiting for first optimisation step...</div>"
            ))
            return

        current_seq_len = int(self.snap_seqlens[-1])
        n_total = len(self._input_ids)
        current_stage = max(0, int(self._last_stage_index)) + 1
        loss_val = self.losses[-1] if self.losses else float("nan")
        conv_val = self.convs[-1] if self.convs else float("nan")

        parts = []
        for i, tid in enumerate(self._input_ids):
            piece = self.tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
            disp = html.escape(piece)
            color = COLOR_MATCH if i < current_seq_len else COLOR_PAST_GT
            parts.append(f'<span style="color:{color};font-weight:600">{disp}</span>')
        body = "".join(parts)
        header = (
            f"<div style='font-size:0.95em;color:#888;margin-bottom:4px'>"
            f"<b>Stage:</b> {current_stage}"
            f" &middot; <b>Compressed</b> "
            f"<span style='color:{COLOR_MATCH};font-weight:600'>{current_seq_len}</span>"
            f" out of {n_total} tokens"
            f" &middot; <b>Accuracy:</b> {conv_val:.3f}"
            f" &middot; <b>Loss:</b> {loss_val:.3f}"
            f"</div>"
        )
        block = (
            f"<div style='font-family:monospace;line-height:1.55;white-space:pre-wrap;"
            f"word-break:break-word;margin:6px 0'>{body}</div>"
        )
        display(HTML(header + block))


def _render_landscape_panel(ax, *, XX, YY, cached, coords, threshold):
    """Draw the paper-style PC1-PC2 landscape into a matplotlib axis.

    Each cached stage contributes one filled region (accuracy > threshold),
    coloured from the ``rocket_r`` palette. Seq-len labels sit at each region's
    centre of mass; the optimisation path is a thin grey polyline with a red
    star at the current cursor.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    n_stages = max(1, len(cached))
    palette = _stage_palette(n_stages, plt)

    for i, region in enumerate(cached):
        Z = region["Z"]
        colour = palette[i]
        ax.contourf(XX, YY, Z, levels=[threshold, 1.001], colors=[colour], alpha=0.55)
        ax.contour(XX, YY, Z, levels=[threshold], colors=[colour], linewidths=0.9, alpha=0.9)
        mask = Z > threshold
        if mask.any():
            cx = float(XX[mask].mean())
            cy = float(YY[mask].mean())
            ax.scatter([cx], [cy], s=42, c="#222", marker="o",
                       edgecolors="white", linewidths=0.9, zorder=8)
            ax.text(
                cx, cy, str(region["seq_len"]),
                fontsize=10, color="#111", ha="center", va="center",
                fontweight="bold", zorder=9,
                path_effects=[withStroke(linewidth=2, foreground="white")],
            )

    ax.plot(coords[:, 0], coords[:, 1], color="#666", alpha=0.7, lw=1.1, zorder=6)
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=60, c="#333", marker="o",
               edgecolors="white", linewidths=0.8, zorder=10)
    ax.scatter([coords[-1, 0]], [coords[-1, 1]], s=180, marker="*",
               c=COLOR_MISMATCH, edgecolors="white", linewidths=0.8, zorder=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Progressive accuracy reveal (regions where accuracy > {threshold:.2f})")
    ax.set_aspect("equal", adjustable="datalim")


def _save_compressed_embedding(result) -> None:
    """Save the converged progressive-cramming embedding + minimal metadata as
    a ``.pt`` file in the current working directory, and offer a Colab download
    button if we're running inside Google Colab.

    The saved dict is intentionally self-contained -- it holds everything needed
    to reload the embedding and reconstruct with :func:`reconstruct_text`:

    ```python
    import torch
    from progressive_cramming.demo import load_frozen_model, reconstruct_text
    blob = torch.load("compressed_embedding.pt", weights_only=False)
    model, tokenizer = load_frozen_model(blob["model_checkpoint"], dtype="float16")
    text = reconstruct_text(model, tokenizer, blob["embedding"], max_new_tokens=blob["horizon"])
    ```
    """
    import os
    import time

    import torch

    payload = {
        "embedding": result.embedding.detach().cpu(),
        "input_ids": list(result.input_ids),
        "text": result.text,
        "horizon": int(result.horizon),
        "num_tokens": int(result.num_tokens),
        "num_mem_tokens": int(result.num_mem_tokens),
        "hidden_size": int(result.hidden_size),
        "model_checkpoint": result.model_checkpoint,
        "num_stages": len(result.stages),
        "total_steps": int(result.total_steps),
        "elapsed_s": float(result.elapsed_s),
    }

    filename = f"compressed_embedding_{int(time.time())}.pt"
    path = os.path.abspath(filename)
    torch.save(payload, path)

    size_kb = os.path.getsize(path) / 1024.0
    display(HTML(
        f"<div style='margin-top:6px;font-size:0.9em;color:#888'>"
        f"💾 Saved compressed embedding to <code>{html.escape(path)}</code> "
        f"({size_kb:.1f} KB)</div>"
    ))


def display_paper_style_animation(viz, *, target_frames: int = 60,
                                  interval_ms: int = 90) -> None:
    """Replay the cached progressive-reveal animation, mirroring the paper's
    ``animate_trajectory.py``.

    Frame-by-frame:
      * cursor moves along ``viz.snaps`` (down-sampled to ``target_frames``)
      * trail grows behind it
      * a stage's accuracy region + seq_len label pop in the first frame whose
        ``snap_seqlens`` reaches that stage's ``seq_len``. Labels sit on the
        stage's *anchor* in trajectory (the last snapshot before the stage
        transitioned), so labels spread along the path instead of stacking at
        the accuracy-mask centre of mass.

    Purely cache-driven -- no model forward passes here, all accuracy maps
    already sit in ``viz._cached`` from :meth:`_ProgressiveLiveViz.finalize`.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if viz._pca is None or not viz._cached:
        display(HTML(
            f"<div style='color:{COLOR_MISMATCH}'>No cached landscape to animate. "
            f"Call <code>viz.finalize()</code> first.</div>"
        ))
        return

    coords = viz._pca.transform(np.stack(viz.snaps))
    n_snaps = coords.shape[0]
    n_frames = int(min(target_frames, n_snaps))
    frame_idx = np.linspace(0, n_snaps - 1, n_frames).astype(int)

    # Anchors, palette: already stored per cached region (each entry has its own
    # ``XX``/``YY`` local grid + ``anchor_xy`` centre point).
    palette = _stage_palette(len(viz._cached), plt)
    anchors = [r["anchor_xy"] for r in viz._cached]

    # Viewport extent = union of all local grids (stored on viz by finalize).
    xmin = float(viz._pca_XX.min())
    xmax = float(viz._pca_XX.max())
    ymin = float(viz._pca_YY.min())
    ymax = float(viz._pca_YY.max())
    pad_x = 0.03 * (xmax - xmin)
    pad_y = 0.03 * (ymax - ymin)
    xlim = (xmin - pad_x, xmax + pad_x)
    ylim = (ymin - pad_y, ymax + pad_y)

    fig, ax = plt.subplots(figsize=(10, 7))
    # No colorbar: the paper's ``visual_abstract_trajectory_zoom_progressive`` also
    # doesn't have one. Every filled region is *bounded* by teacher-forced
    # reconstruction accuracy > 0.9 -- there is no continuous accuracy scale to
    # colour along. What the palette encodes is *which stage anchor* opened this
    # region (rocket_r ramp: light for early anchors, dark for late), and the
    # anchor's ``seq_len`` is written as a label on the anchor point itself.

    def render(k: int) -> None:
        ax.clear()
        snap_k = int(frame_idx[k])
        seqlen_k = int(viz.snap_seqlens[snap_k])

        for i, region in enumerate(viz._cached):
            if region["seq_len"] > seqlen_k:
                continue
            Z = region["Z"]
            colour = palette[i]
            # Per-anchor local grid -- this is what makes islands compact.
            XX_i = region["XX"]
            YY_i = region["YY"]
            ax.contourf(XX_i, YY_i, Z, levels=[viz.threshold, 1.001], colors=[colour], alpha=0.55)
            ax.contour(XX_i, YY_i, Z, levels=[viz.threshold], colors=[colour], linewidths=0.9, alpha=0.9)
            ax_anchor = anchors[i]
            ax.scatter([ax_anchor[0]], [ax_anchor[1]], s=48, c="#222",
                       marker="o", edgecolors="white", linewidths=0.9, zorder=8)
            ax.annotate(
                str(region["seq_len"]),
                xy=ax_anchor,
                xytext=(4, 4), textcoords="offset points",
                fontsize=9, color="#111", fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )

        # Trajectory trail up to current snapshot + init + cursor markers.
        ax.plot(coords[: snap_k + 1, 0], coords[: snap_k + 1, 1],
                color="#666", alpha=0.75, lw=1.1, zorder=6)
        ax.scatter([coords[0, 0]], [coords[0, 1]], s=90, c="#333", marker="o",
                   edgecolors="white", linewidths=1.0, zorder=10, label="init")
        ax.scatter([coords[snap_k, 0]], [coords[snap_k, 1]], s=220, marker="*",
                   c=COLOR_MISMATCH, edgecolors="white", linewidths=0.8,
                   zorder=11, label="cursor")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"Regions where teacher-forced reconstruction accuracy > 0.90  ·  "
            f"cursor at seq_len = {seqlen_k}  (frame {k + 1}/{n_frames})"
        )
        ax.set_aspect("equal", adjustable="box")

    anim = FuncAnimation(fig, render, frames=n_frames, interval=interval_ms, blit=False)
    html_str = anim.to_jshtml(default_mode="loop")
    plt.close(fig)
    display(HTML(html_str))


@torch.no_grad()
def _accuracy_batch(model, *, compression_flat, mem_shape, input_ids, text_embeds,
                    attention_mask, batch_size: int = 8):
    """Teacher-forced match-rate of a batch of candidate compression embeddings against ``input_ids``.

    Mirrors the ``_compute_accuracy_batch`` helper used by the paper's
    ``visualize_landscale_2pca.py``: cat compression + text embeddings, forward,
    argmax over logits at the continuation positions, count matches.
    """
    import numpy as np

    model.eval()
    device = next(model.parameters()).device
    num_grid = int(compression_flat.shape[0])
    mem_tokens, hidden = int(mem_shape[0]), int(mem_shape[1])
    denom = attention_mask.sum(dim=-1).clamp_min(1).float()

    accs: list = []
    for batch_start in range(0, num_grid, batch_size):
        batch_end = min(batch_start + batch_size, num_grid)
        batch = compression_flat[batch_start:batch_end]
        bs = int(batch.shape[0])
        comp = batch.reshape(bs, mem_tokens, hidden).to(device=device, dtype=text_embeds.dtype)
        text_bs = text_embeds.expand(bs, -1, -1)
        inputs_embeds = torch.cat([comp, text_bs], dim=1)
        comp_attn = torch.ones((bs, mem_tokens), device=device, dtype=attention_mask.dtype)
        attn_bs = attention_mask.expand(bs, -1)
        full_attn = torch.cat([comp_attn, attn_bs], dim=1)
        out = model(inputs_embeds=inputs_embeds, attention_mask=full_attn, use_cache=False)
        pred = out.logits[:, mem_tokens - 1 : -1].argmax(dim=-1)  # [bs, L]
        match = (pred == input_ids.expand(bs, -1)).to(torch.float32)
        match = match * attn_bs.to(torch.float32)
        accs.append((match.sum(dim=-1) / denom.expand(bs)).cpu().numpy())
    return np.concatenate(accs, axis=0)


def _compute_per_stage_landscape(result, model, *, grid_size: int = 28,
                                 padding: float = 0.4, batch_size: int = 8,
                                 threshold: float = 0.9):
    """For each PC stage, compute teacher-forced accuracy on a shared PC1-PC2
    grid fit on the whole optimisation trajectory -- so the "high-accuracy
    regions" of every stage live in the same plane and can be overlaid.

    Adapted from ``compression_horizon/scripts/paper/animate_trajectory.py`` +
    ``visualize_landscale_2pca.py``: cache-only per-anchor regions, then draw
    them together (see ``visual_abstract_trajectory_zoom_progressive``). We
    skip the mp4/zoom animation and render a single static composite.

    Returns a dict with ``XX``, ``YY``, ``per_stage_Z`` (list of accuracy maps
    per stage), ``coords`` (trajectory-snapshot projections), ``snap_seq_lens``,
    ``stage_seq_lens``, and ``threshold``. Returns ``None`` if the trajectory is
    too short to fit a PCA.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    if result.trajectory is None or len(result.trajectory) < 3:
        return None

    snapshots = result.trajectory.numpy()
    pca = PCA(n_components=2).fit(snapshots)
    coords = pca.transform(snapshots)  # [N_snaps, 2]
    span = (coords.max(axis=0) - coords.min(axis=0))
    pad = padding * np.maximum(span, 1e-3)
    x = np.linspace(coords[:, 0].min() - pad[0], coords[:, 0].max() + pad[0], grid_size)
    y = np.linspace(coords[:, 1].min() - pad[1], coords[:, 1].max() + pad[1], grid_size)
    XX, YY = np.meshgrid(x, y)
    grid_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)
    grid_embeds = pca.inverse_transform(grid_xy).astype(np.float32)
    grid_t = torch.tensor(grid_embeds, dtype=torch.float32)

    device = next(model.parameters()).device
    per_stage_Z: list = []
    stage_seq_lens = [s.seq_len for s in result.stages]
    for seq_len in stage_seq_lens:
        input_ids = torch.tensor([result.input_ids[:seq_len]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        text_embeds = model.get_input_embeddings()(input_ids)
        Z = _accuracy_batch(
            model,
            compression_flat=grid_t,
            mem_shape=(result.num_mem_tokens, result.hidden_size),
            input_ids=input_ids,
            text_embeds=text_embeds,
            attention_mask=attention_mask,
            batch_size=batch_size,
        ).reshape(XX.shape)
        per_stage_Z.append(Z)

    return dict(
        XX=XX, YY=YY,
        per_stage_Z=per_stage_Z,
        coords=coords,
        snap_seq_lens=list(result.trajectory_seq_len) if result.trajectory_seq_len else None,
        stage_seq_lens=stage_seq_lens,
        threshold=threshold,
    )


def _draw_per_stage_landscape(landscape) -> None:
    """Overlay one filled region per stage (``accuracy > threshold``) on a shared
    PC1-PC2 plane + trajectory polyline through the snapshots.

    Same semantics as ``visual_abstract_trajectory_zoom_progressive``'s final
    frame -- every stage gets its own colour, the trajectory line stitches the
    stages together, and the converged embedding is starred at the end.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    XX = landscape["XX"]
    YY = landscape["YY"]
    per_stage_Z = landscape["per_stage_Z"]
    coords = landscape["coords"]
    snap_seq_lens = landscape["snap_seq_lens"]
    stage_seq_lens = landscape["stage_seq_lens"]
    threshold = landscape["threshold"]

    n_stages = len(per_stage_Z)
    palette = _stage_palette(n_stages, plt)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.set_facecolor("#111111")

    handles = []
    for i, (Z, colour, seq_len) in enumerate(zip(per_stage_Z, palette, stage_seq_lens)):
        # Filled contour where accuracy exceeds the threshold.
        ax.contourf(XX, YY, Z, levels=[threshold, 1.01], colors=[colour], alpha=0.5)
        # Thin bright outline so overlapping regions stay legible.
        ax.contour(XX, YY, Z, levels=[threshold], colors=[colour], linewidths=1.0, alpha=0.9)
        handles.append(Patch(facecolor=colour, alpha=0.7, label=f"stage {i+1} — {seq_len} tok"))

    # Trajectory polyline + stage-coloured snapshots.
    ax.plot(coords[:, 0], coords[:, 1], color="white", alpha=0.85, lw=1.3)
    if snap_seq_lens is not None and len(snap_seq_lens) == coords.shape[0]:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=snap_seq_lens, cmap="plasma", s=22, edgecolor="black", linewidths=0.3,
        )
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=20, c="white", edgecolor="black", linewidths=0.3)

    # Anchors: init (black dot) + converged (red star).
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=150, c="white", marker="o",
               edgecolors="black", linewidths=1.2, zorder=15, label="init")
    ax.scatter([coords[-1, 0]], [coords[-1, 1]], s=280, marker="*", c=COLOR_MISMATCH,
               edgecolors="black", linewidths=1.0, zorder=16, label="converged")

    ax.set_xlabel("PC1", color="#ddd")
    ax.set_ylabel("PC2", color="#ddd")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#666")
    ax.set_title(
        f"Per-stage accuracy regions on PC1–PC2 (colour = stage; "
        f"filled where teacher-forced accuracy > {threshold:.2f})",
        color="#ddd",
    )
    ax.set_aspect("equal", adjustable="datalim")
    legend = ax.legend(handles=handles, loc="upper left", fontsize=8,
                       ncol=min(2, max(1, n_stages // 4)), framealpha=0.85)
    legend.get_frame().set_facecolor("#222")
    for text in legend.get_texts():
        text.set_color("#ddd")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def display_interactive_compress(
    models: dict,
    *,
    default: str | None = None,
    max_seq_len: int = 128,
    default_max_steps_per_token: int = 300,
) -> None:
    """Interactive §5 widget: cram a user-provided text via *progressive* cramming
    and watch the PCA trajectory + loss/reconstruction curves live.

    ``models`` is a ``{display_name: (model, tokenizer)}`` mapping built by the
    notebook from already-loaded frozen models (typically the Llama-3.2-1B and
    SmolLM2-360M instances used by §3/§4). The widget never loads a model itself
    -- this keeps T4 memory predictable and matches the "no GPU side-effects in
    plotting helpers" convention used in §3 and §4.

    Each ▶ Compress click runs :func:`progressive_cram_text` end to end with
    ``capture_every`` tuned to the step budget, then renders the reconstruction
    at the horizon via the same coloured diff used in §3 and §4.
    """
    import ipywidgets as widgets

    from ._core import progressive_cram_text

    if not models:
        display(HTML(
            f"<div style='color:{COLOR_MISMATCH}'>No models supplied. Pass a "
            f"<code>{{name: (model, tokenizer)}}</code> dict.</div>"
        ))
        return

    if default is None or default not in models:
        default = next(iter(models))

    # Ready-made examples the user can drop into the textarea in one click.
    # Each is short (~80-120 tokens on the Llama-3.2 tokenizer) so it fits under
    # the default 128-token slider without truncation.
    EXAMPLES = {
        "Literature": (
            "As Gregor Samsa awoke one morning from uneasy dreams he found himself "
            "transformed in his bed into a gigantic insect. He was lying on his hard, "
            "armour-plated back, and when he lifted his head a little he could see his "
            "dome-like brown belly divided into stiff arched segments, on top of which "
            "the bed quilt could hardly keep in position and was about to slide off completely."
        ),
        "Poetry": (
            "I shall be telling this with a sigh\n"
            "Somewhere ages and ages hence:\n"
            "Two roads diverged in a wood, and I—\n"
            "I took the one less traveled by,\n"
            "And that has made all the difference."
        ),
        "Python": (
            "def sieve(n):\n"
            "    is_prime = [True] * (n + 1)\n"
            "    is_prime[0] = is_prime[1] = False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if is_prime[i]:\n"
            "            for j in range(i * i, n + 1, i):\n"
            "                is_prime[j] = False\n"
            "    return [i for i, p in enumerate(is_prime) if p]\n"
            "\n"
            "print(sieve(50))\n"
        ),
        "Go": (
            "package main\n"
            "\n"
            "import \"fmt\"\n"
            "\n"
            "func fib(n int) int {\n"
            "    if n < 2 {\n"
            "        return n\n"
            "    }\n"
            "    return fib(n-1) + fib(n-2)\n"
            "}\n"
            "\n"
            "func main() {\n"
            "    for i := 0; i < 10; i++ {\n"
            "        fmt.Println(fib(i))\n"
            "    }\n"
            "}\n"
        ),
    }

    text_input = widgets.Textarea(
        value=EXAMPLES["Literature"],
        layout=widgets.Layout(width="100%", height="120px"),
    )

    example_buttons = []
    for name in EXAMPLES:
        b = widgets.Button(
            description=name,
            layout=widgets.Layout(width="auto"),
            tooltip=f"Load the {name} example",
        )
        b.on_click(lambda _b, key=name: setattr(text_input, "value", EXAMPLES[key]))
        example_buttons.append(b)
    examples_row = widgets.HBox(
        [widgets.HTML("<span style='color:#888;font-size:0.9em;margin-right:6px'>Examples:</span>")]
        + example_buttons,
        layout=widgets.Layout(margin="0 0 4px 0"),
    )
    model_dd = widgets.Dropdown(options=list(models), value=default, description="Model")
    len_slider = widgets.IntSlider(
        value=min(128, max_seq_len), min=8, max=max_seq_len, step=4, description="Max tokens",
    )
    steps_slider = widgets.IntSlider(
        value=default_max_steps_per_token, min=100, max=1000, step=50,
        description="Max steps / token",
    )
    btn = widgets.Button(description="▶ Compress", button_style="primary")
    out = widgets.Output()
    last_result: dict = {}

    legend = (
        f'<span style="color:{COLOR_MATCH};font-weight:600">green = match</span>'
        f' &middot; '
        f'<span style="color:{COLOR_MISMATCH};font-weight:600">red = mismatch</span>'
        f' &middot; '
        f'<span style="color:{COLOR_PAST_GT};font-weight:600">grey = past the original span</span>'
    )

    def on_click(_):
        out.clear_output()
        with out:
            m, t = models[model_dd.value]
            span_tokens = int(len_slider.value)
            # Paper's "few anchors" look: target ~8 regions on the landscape,
            # independent of the total span. For short texts (≤ 8 tokens) every
            # stage gets its own region; for longer texts we subsample.
            region_stride = max(1, span_tokens // 8)
            viz = _ProgressiveLiveViz(
                model=m,
                tokenizer=t,
                redraw_every=max(5, steps_slider.value // 30),
                region_seq_len_stride=region_stride,
                first_seq_len=1,
            )
            capture = max(2, steps_slider.value // 80)
            result = progressive_cram_text(
                m, t, text_input.value,
                max_seq_len=span_tokens,
                max_steps_per_token=int(steps_slider.value),
                # Canonical progressive cramming (paper Appendix A): one token
                # per stage. Total stages = span_tokens.
                step=1,
                capture_every=capture,
                on_step=viz,
            )

            # Build a row-shaped dict so reconstruct_and_show works without an adapter.
            row = {
                "embedding": result.embedding.numpy().tolist(),
                "input_ids": list(result.input_ids),
                "num_tokens": result.horizon,  # decode exactly the converged horizon
                "text": result.text,
                "final_convergence": 1.0 if result.horizon else 0.0,
            }
            extra = max(1, result.horizon // 5) if result.horizon else 0
            display(HTML(
                f"<div style='font-size:0.9em;color:#888;margin-top:6px;margin-bottom:4px'>{legend}</div>"
            ))
            reconstruct_and_show(
                row, "Reconstruction at horizon",
                model=m, tokenizer=t, extra_tokens=extra,
                stream=True,
            )
            display(HTML(
                f"<div style='margin-top:4px;font-size:0.9em;color:#888'>"
                f"horizon: <b>{result.horizon}/{result.num_tokens}</b> tokens &middot; "
                f"{result.total_steps} steps &middot; {result.elapsed_s:.1f}s &middot; "
                f"<b>{len(result.stages)}</b> stages &middot; "
                f"model: <code>{html.escape(result.model_checkpoint)}</code></div>"
            ))

            # Persist the converged embedding + minimal metadata so the user can
            # download the compression artefact from Colab (or reload it later).
            _save_compressed_embedding(result)
            last_result["result"] = result

    btn.on_click(on_click)
    display(widgets.VBox([
        examples_row,
        text_input,
        widgets.HBox([model_dd, len_slider, steps_slider]),
        btn,
        out,
    ]))
