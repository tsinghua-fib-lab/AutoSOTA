from __future__ import annotations

import html
import re
import sys
import textwrap
import uuid
from collections import OrderedDict
from pathlib import Path

BLOCK_RE = re.compile(
    r'<pre\b[^>]*>\s*<code class="language-mermaid">(.*?)</code>\s*</pre>',
    re.DOTALL,
)
SCRIPT_RE = re.compile(
    r'\s*<style id="mermaid-postprocess-style">.*?</script>\s*',
    re.DOTALL,
)


class ParsedFlowchart:
    def __init__(self, direction: str, nodes: list[tuple[str, str]], edges: list[tuple[str, str]]):
        self.direction = direction
        self.nodes = nodes
        self.edges = edges


def _normalize_lines(src: str) -> list[str]:
    raw = html.unescape(src)
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('%%'):
            continue
        lines.append(line)
    return lines


def _parse_node_token(token: str) -> tuple[str, str | None]:
    token = token.strip().rstrip(';')
    m = re.fullmatch(r'([A-Za-z0-9_]+)\[(.+)\]', token)
    if m:
        return m.group(1), m.group(2).strip()
    return token, None


def parse_mermaid_flowchart(src: str) -> ParsedFlowchart | None:
    lines = _normalize_lines(src)
    if not lines:
        return None

    header = lines[0].split()
    if len(header) < 2 or header[0] not in {'flowchart', 'graph'}:
        return None
    direction = header[1].upper()
    if direction not in {'TD', 'TB', 'LR'}:
        return None

    nodes: OrderedDict[str, str] = OrderedDict()
    edges: list[tuple[str, str]] = []

    def ensure_node(token: str) -> str:
        node_id, label = _parse_node_token(token)
        if node_id not in nodes:
            nodes[node_id] = label or node_id
        elif label:
            nodes[node_id] = label
        return node_id

    for line in lines[1:]:
        if '-->' in line:
            parts = [part.strip() for part in line.split('-->')]
            ids = [ensure_node(part) for part in parts if part.strip()]
            for a, b in zip(ids, ids[1:], strict=False):
                edges.append((a, b))
            continue

        node_id, label = _parse_node_token(line)
        if label is not None:
            nodes[node_id] = label
        elif node_id and node_id not in nodes:
            nodes[node_id] = node_id

    if not nodes:
        return None
    return ParsedFlowchart(direction=direction, nodes=list(nodes.items()), edges=edges)


def _wrap_label(label: str, width: int = 18) -> list[str]:
    wrapped = textwrap.wrap(label, width=width) or [label]
    return wrapped[:3]


def render_svg(chart: ParsedFlowchart) -> str:
    node_specs = []
    max_width = 0
    for node_id, label in chart.nodes:
        lines = _wrap_label(label)
        width = max(180, max(len(line) for line in lines) * 9 + 34)
        height = 34 + (len(lines) - 1) * 16
        node_specs.append((node_id, label, lines, width, height))
        max_width = max(max_width, width)

    gap = 46
    pad = 24
    if chart.direction in {'TD', 'TB'}:
        total_height = pad * 2 + sum(h for *_, h in node_specs) + gap * (len(node_specs) - 1)
        total_width = max_width + pad * 2
    else:
        total_width = pad * 2 + sum(w for *_, w, _ in node_specs) + gap * (len(node_specs) - 1)
        total_height = max(h for *_, _, h in node_specs) + pad * 2

    marker_id = f"arrow-{uuid.uuid4().hex[:8]}"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_width} {total_height}" role="img" aria-label="Compiled Mermaid flowchart">',
        '<style>',
        '.box{fill:#fff9ef;stroke:#0f3d3e;stroke-width:2;rx:14;ry:14;}',
        '.edge{stroke:#8a4b08;stroke-width:2.4;fill:none;marker-end:url(#%s);}' % marker_id,
        '.label{fill:#102a2b;font-family:Aptos,Trebuchet MS,sans-serif;font-size:16px;font-weight:600;text-anchor:middle;dominant-baseline:middle;}',
        '</style>',
        '<defs>',
        f'<marker id="{marker_id}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="#8a4b08" />',
        '</marker>',
        '</defs>',
    ]

    centers: dict[str, tuple[float, float, float, float]] = {}
    if chart.direction in {'TD', 'TB'}:
        y = pad
        x = (total_width - max_width) / 2
        for node_id, _label, lines, _width, height in node_specs:
            svg.append(f'<rect class="box" x="{x}" y="{y}" width="{max_width}" height="{height}" rx="14" ry="14" />')
            line_y = y + height / 2 - (len(lines) - 1) * 9
            for idx, line in enumerate(lines):
                svg.append(f'<text class="label" x="{x + max_width / 2}" y="{line_y + idx * 18}">{html.escape(line)}</text>')
            centers[node_id] = (x + max_width / 2, y, x + max_width / 2, y + height)
            y += height + gap
        for a, b in chart.edges:
            x1, _y1t, _x1b, y1 = centers[a]
            x2, y2, _x2b, _y2b = centers[b]
            mid_y = (y1 + y2) / 2
            svg.append(f'<path class="edge" d="M {x1} {y1} L {x1} {mid_y} L {x2} {mid_y} L {x2} {y2 - 4}" />')
    else:
        x = pad
        max_height = max(h for *_, h in node_specs)
        for node_id, _label, lines, width, height in node_specs:
            y = (total_height - height) / 2
            svg.append(f'<rect class="box" x="{x}" y="{y}" width="{width}" height="{height}" rx="14" ry="14" />')
            line_y = y + height / 2 - (len(lines) - 1) * 9
            for idx, line in enumerate(lines):
                svg.append(f'<text class="label" x="{x + width / 2}" y="{line_y + idx * 18}">{html.escape(line)}</text>')
            centers[node_id] = (x, y + height / 2, x + width, y + height / 2)
            x += width + gap
        for a, b in chart.edges:
            x1, y1, x1r, _ = centers[a]
            x2, y2, _x2r, _ = centers[b]
            mid_x = (x1r + x2) / 2
            svg.append(f'<path class="edge" d="M {x1r} {y1} L {mid_x} {y1} L {mid_x} {y2} L {x2 - 4} {y2}" />')

    svg.append('</svg>')
    return ''.join(svg)


def replace_block(match: re.Match[str]) -> str:
    src = match.group(1)
    parsed = parse_mermaid_flowchart(src)
    if parsed is None:
        return match.group(0)
    return render_svg(parsed)


def main() -> int:
    if len(sys.argv) != 2:
        print('usage: postprocess_marp_mermaid.py <html-file>', file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    text = path.read_text(encoding='utf-8')
    text = SCRIPT_RE.sub('\n', text)
    text, count = BLOCK_RE.subn(replace_block, text)
    path.write_text(text, encoding='utf-8')
    print(f'compiled {count} mermaid block(s) in {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
