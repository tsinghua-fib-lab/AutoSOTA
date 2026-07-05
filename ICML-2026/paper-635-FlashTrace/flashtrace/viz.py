from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import TraceResult


def _score_color(score: float, max_score: float) -> str:
    if max_score <= 0.0:
        return "rgba(245,245,245,0.75)"
    ratio = min(1.0, abs(float(score)) / (max_score + 1e-12))
    red = 255
    green = int(246 - 105 * ratio)
    blue = int(226 - 170 * ratio)
    alpha = 0.22 + 0.58 * ratio
    return f"rgba({red},{green},{blue},{alpha:.3f})"


def _render_token_row(tokens: list[str], scores: list[float]) -> str:
    max_score = max((abs(float(x)) for x in scores), default=0.0)
    spans = []
    for index, token in enumerate(tokens):
        score = float(scores[index]) if index < len(scores) else 0.0
        color = _score_color(score, max_score)
        spans.append(
            "<span class='tok' "
            f"title='idx={index} score={score:.6f}' "
            f"style='background:{color}'>{escape(token)}</span>"
        )
    return "".join(spans)


def render_trace_html(result: "TraceResult") -> str:
    top_rows = "\n".join(
        f"<tr><td>{item.index}</td><td><code>{escape(item.token)}</code></td><td>{item.score:.6f}</td></tr>"
        for item in result.topk_inputs(20)
    )
    hop_sections = []
    for hop_index, hop_scores in enumerate(result.per_hop_scores):
        hop_sections.append(
            f"<section><h2>Hop {hop_index}</h2><div class='tokens'>{_render_token_row(result.prompt_tokens, hop_scores)}</div></section>"
        )
    hop_html = "\n".join(hop_sections)
    metadata = escape(str(result.metadata))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FlashTrace</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #151515; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 24px 0; }}
    .tokens {{ line-height: 2.2; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .tok {{ display: inline-block; margin: 2px; padding: 2px 4px; border-radius: 4px; white-space: pre-wrap; }}
    table {{ border-collapse: collapse; margin-top: 12px; }}
    td, th {{ border-bottom: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
    .meta {{ color: #555; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>FlashTrace</h1>
  <p class="meta">method={escape(result.method)} output_span={escape(str(result.output_span))} reasoning_span={escape(str(result.reasoning_span))}</p>
  <section>
    <h2>Prompt Attribution</h2>
    <div class="tokens">{_render_token_row(result.prompt_tokens, result.scores)}</div>
  </section>
  {hop_html}
  <section>
    <h2>Top Input Tokens</h2>
    <table><thead><tr><th>Index</th><th>Token</th><th>Score</th></tr></thead><tbody>{top_rows}</tbody></table>
  </section>
  <section>
    <h2>Metadata</h2>
    <pre>{metadata}</pre>
  </section>
</body>
</html>
"""
