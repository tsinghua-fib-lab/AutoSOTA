from __future__ import annotations
from pathlib import Path
from typing import Any
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate, tabulate_formats

RUN_MODES = ["agentic", "normal"]
BENCH_SET = ["original", "extended", "algebraic"]

SHORT_NAME = {
    "us.anthropic.claude-3-7-sonnet-20250219-v1.0_deep-research": "N-Research\nClaude-3.7-Sonnet",
    "us.anthropic.claude-3-7-sonnet-20250219-v1.0_agentic": "A-Bitween\nClaude-3.7-Sonnet",
    "us.anthropic.claude-opus-4-20250514-v1.0_deep-research": "N-Research\nClaude-Opus-4",
    "us.anthropic.claude-opus-4-20250514-v1.0_agentic": "A-Bitween\nClaude-Opus-4",
    "openai.gpt-oss-120b-1.0_deep-research": "N-Research\nGPT-OSS-120B",
    "openai.gpt-oss-120b-1.0_agentic": "A-Bitween\nGPT-OSS-120B",
    "us.anthropic.claude-sonnet-4-20250514-v1.0_deep-research": "N-Research\nClaude-Sonnet-4",
    "us.anthropic.claude-sonnet-4-20250514-v1.0_agentic": "A-Bitween\nClaude-Sonnet-4",
    "us.anthropic.claude-opus-4-1-20250805-v1.0_deep-research": "N-Research\nClaude-Opus-4.1",
    "us.anthropic.claude-opus-4-1-20250805-v1.0_agentic": "A-Bitween\nClaude-Opus-4.1",
    "pysr": "V-Bitween\nPySR",
    "gplearn": "V-Bitween\nGPLearn",
    "milp": "V-Bitween\nMILP",
    "mreg": "V-Bitween\nLR",
}

COLUMN_INFO = {
    "results": "results",
    "rsr": "rsr",
    "verified": "verified",
    "unverified": "unverified",
    "time": "time",
    "tokens": "tokens",
    "empty": "empty",
    "tools": "tools",
}

TOOL_INFO = {
    "symbolic_verify_tool": "Bitween Verification Tool",
    "infer_property_tool": "Bitween Inference Tool",
    "sequentialthinking": "Sequential Thinking",
}


class SubTableRow:
    def __init__(
        self,
        id: str,
        name: str,
        time: float,
        tokens: int,
        verified: tuple[int, list[str]],
        unverified: tuple[int, list[str]],
        tools: dict,
    ):
        self.id = id
        self.name = name
        self.time = time
        self.tokens = tokens
        self.verified = verified
        self.unverified = unverified
        self.tools = tools

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "time": self.time,
            "tokens": self.tokens,
            "verified": self.verified,
            "unverified": self.unverified,
            "tools": self.tools,
            "rsr": "?",
        }

    @staticmethod
    def empty_row() -> SubTableRow:
        return SubTableRow("", "", 0, 0, (0, []), (0, []), {})

    def common_header(self) -> list[str]:
        return ["#", "name"]

    def header(
        self,
        columns: list[str] | None = None,
        full: bool = False,
    ) -> list[str]:
        columns = columns or list(COLUMN_INFO.keys())
        header = []

        for col in columns:
            if col not in COLUMN_INFO:
                print(f"No available column: {col}")
            else:
                header.append(COLUMN_INFO[col])

        if full:
            header = self.common_header() + header
        return header

    def body(
        self,
        columns: list[str] | None = None,
        full=False,
        properties=False,
    ) -> list[Any]:
        columns = columns or list(COLUMN_INFO.keys())

        def make_cell(pair):
            num, lines = pair
            if properties:
                return str(num) + "&&" + "&&".join(lines)
            else:
                return num

        rsr = make_cell(("?", self.verified[1]))
        verified = make_cell(self.verified)
        unverified = make_cell(self.unverified)

        col_value_dct = {
            "results": f"? / {self.verified[0]} | {self.unverified[0]}",
            "time": self.time,
            "tokens": self.tokens,
            "rsr": rsr,
            "verified": verified,
            "unverified": unverified,
            "tools": self.tools,
            "empty": "",
        }
        assert col_value_dct.keys() == COLUMN_INFO.keys(), col_value_dct

        row = []
        for col in columns:
            if col not in col_value_dct:
                print(f"No available column: {col}")
            else:
                row.append(col_value_dct[col])

        if full:
            row = [self.id, self.name] + row

        return row

    @staticmethod
    def from_file(file: Path) -> SubTableRow:
        if not file.suffix == ".txt":
            print(f"Not a .txt file: {file}")
            return SubTableRow.empty_row()

        bench_id, bench_name = file.with_suffix("").name.split("_", 1)

        # entry: (res_dct key, pattern with capture groups)
        pattern_info = [
            ("time", r"Took time: (.*)s", 0),
            ("tokens", r".*Tokens:.*, total=(\d+)", 0),
            ("verified", r"Verified \((\d+)\):\n(.*?)\n\n", re.DOTALL),
            ("unverified", r"Unverified \((\d+)\):\n(.*?)\n\n", re.DOTALL),
            *[
                ("tools", rf"({name}):\n.*?success=(\d+)", re.DOTALL)
                for name in TOOL_INFO.keys()
            ],
        ]

        res_dct = {
            "id": bench_id,
            "name": bench_name,
            "time": 0.0,
            "tokens": 0,
            "verified": (0, ""),
            "unverified": (0, ""),
            "tools": {},
        }

        def add_key_value(key: str, value: str):
            if key == "time":
                res_dct[key] = float(value)
            elif key == "tokens":
                res_dct[key] = int(value)
            elif key in [
                "verified",
                "unverified",
                "faulty",
                "unknown",
            ]:
                num, lines_str = value
                lines = [line.split(" | ")[0] for line in lines_str.split("\n")]
                res_dct[key] = (int(num), lines)
            elif key == "tools":
                tool_name, success_calls = value
                res_dct[key][tool_name] = int(success_calls)

        with file.open("rt") as fd:
            content = fd.read()

        for key, pat, flags in pattern_info:
            matches = re.findall(pat, content, flags)
            if len(matches) == 1:
                value = matches[0]
                add_key_value(key, value)

        return SubTableRow(**res_dct)


class SubTable:
    def __init__(self, model_name: str, rows: list[SubTableRow]):
        assert len(rows) > 0
        self.model_name = model_name
        self.rows = rows

    def common_header(self) -> list[str]:
        return self.rows[0].common_header()

    def header(
        self,
        columns: list[str] | None = None,
        full: bool = False,
    ) -> list[str]:
        row_header = self.rows[0].header(columns=columns, full=full)
        return [self.model_name] * len(row_header)

    def body(
        self,
        columns: list[str] | None = None,
        full: bool = False,
        properties: bool = False,
    ) -> list[list[Any]]:
        row_header = self.rows[0].header(columns=columns, full=full)
        rows = [row.body(columns, full, properties) for row in self.rows]

        return [row_header] + rows

    def column_values(self, col: str) -> list[Any]:
        vals = []
        for row in self.rows:
            row_dict = row.to_dict()
            if col not in row_dict:
                print(f"Column not found: {col}")
            else:
                vals.append(row_dict[col])
        return vals

    @staticmethod
    def name_key_from_dir(dir: Path) -> str:
        basename = dir.name.removesuffix("/")
        for bset in BENCH_SET:
            bset_pat = f"_{bset}"
            if bset_pat in basename:
                name_key = basename.split(bset_pat)[0]
                return name_key

        return "N/A"

    @staticmethod
    def from_dir(dir: Path) -> SubTable:
        name_key = SubTable.name_key_from_dir(dir)
        model_name = SHORT_NAME.get(name_key, name_key)

        if not dir.exists():
            print(f"Directory does not exist: {dir}")
            return SubTable(model_name, [])

        txt_files = list(dir.glob("*.txt"))
        txt_files.sort()

        rows = list(map(SubTableRow.from_file, txt_files))

        return SubTable(model_name, rows)


class Table:
    def __init__(self, dirnames: list[Path]):
        assert len(dirnames) > 0
        self.subtables = list(map(SubTable.from_dir, dirnames))
        self.common_header = self.subtables[0].common_header()

    def header(self, columns: list[str] | None = None) -> list[str]:
        header = [""] * len(self.common_header)

        for subtable in self.subtables:
            st_header = subtable.header(columns=columns, full=False)
            header += st_header

        return header

    def body(
        self,
        columns: list[str] | None = None,
        properties: bool = False,
    ) -> list[list[Any]]:
        body = self.subtables[0].body(
            columns=columns,
            full=True,
            properties=properties,
        )

        for subtable in self.subtables[1:]:
            st_body = subtable.body(
                columns=columns,
                full=False,
                properties=properties,
            )
            body = [row1 + row2 for row1, row2 in zip(body, st_body)]

        return body

    def tabulate(
        self,
        columns: list[str] | None = None,
        properties: bool = False,
        **tabulate_kwargs,
    ):
        return tabulate(
            self.body(columns, properties),
            headers=self.header(columns),
            **tabulate_kwargs,
        )

    def column_dict(self, col: str):
        dct = {}
        for tbl in self.subtables:
            dct.setdefault(tbl.model_name, [])
            dct[tbl.model_name].extend(tbl.column_values(col))
        return dct

    def plot_heatmap(
        self,
        col: str,
        title: str,
        colorbar_label: str,
        outfile: str,
        tool: str | None = None,
    ):
        dct = self.column_dict(col)

        if tool:
            assert col == "tools", col

            for model, dcts in dct.items():
                tool_calls = []
                for idct in dcts:
                    tool_calls.append(idct.get(tool, 0))
                dct[model] = tool_calls

        if not dct:
            print(f"No dictionary could be generated from {col}")
            return

        methods = list(dct.keys())
        data = np.array([dct[method] for method in methods])

        fig, ax = plt.subplots()
        cax = ax.imshow(data, aspect="auto", cmap="Oranges")

        cbar = fig.colorbar(
            cax,
            ax=ax,
            shrink=0.85,
        )
        cbar.set_label(label=colorbar_label, weight="bold")
        cbar.outline.set_visible(False)
        # ax.set_aspect(data.shape[1] / data.shape[0])
        for tick in cbar.ax.yaxis.get_major_ticks():
            tick.label2.set_fontweight("bold")
            xticks = [i for i in range(data.shape[1]) if (i + 1) % 5 == 0 or i == 0]

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i + 1) for i in xticks], fontsize=10, fontweight="bold")

        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods, fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Benchmarks", fontsize=12, fontweight="bold")
        # ax.set_ylabel("Methods")

        fig.tight_layout()
        fig.savefig(outfile, transparent=True)


def get_parser():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dirs",
        type=Path,
        required=True,
        nargs="+",
        help="space-separated directories that contain .txt output files",
    )

    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(COLUMN_INFO.keys()),
        help="space-separated columns to appear in the table",
    )

    parser.add_argument(
        "--plot_table",
        action="store_true",
        help="print the table",
    )

    parser.add_argument(
        "--table_out",
        type=str,
        default="table.txt",
        help="the output file of the table",
    )

    parser.add_argument(
        "--table_append",
        action="store_true",
        help="append the table to the output file (not overwrite)",
    )

    parser.add_argument(
        "--properties",
        action="store_true",
        help="include all properties in the table",
    )

    parser.add_argument(
        "--plot_time",
        action="store_true",
        help="plot the runtime heatmap based on input dirs",
    )

    parser.add_argument(
        "--time_out",
        type=str,
        default="time_heatmap.pdf",
        help="the output file of the time heatmap",
    )

    parser.add_argument(
        "--plot_tokens",
        action="store_true",
        help="plot the tokens heatmap based on input dirs",
    )

    parser.add_argument(
        "--tokens_out",
        type=str,
        default="tokens_heatmap.pdf",
        help="the output file of the tokens heatmap",
    )

    parser.add_argument(
        "--plot_tools",
        nargs="*",
        default=[],
        choices=TOOL_INFO.keys(),
        help="space-separated tools to plot their heatmap based on input dirs",
    )

    parser.add_argument(
        "--tools_out",
        nargs="*",
        default=[],
        help="space-separated output file for each tool in `--plot_tools`",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="rounded_grid",
        choices=tabulate_formats,
        help="format of the table according to tabulate",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    table = Table(args.dirs)
    table_str = table.tabulate(
        columns=args.columns,
        properties=args.properties,
        tablefmt=args.format,
    )

    if args.plot_table:
        if args.table_out:
            mode = "at" if args.table_append else "wt"
            with open(args.table_out, mode) as fp:
                fp.write(table_str + "\n\n")
        else:
            print(table_str)

    if args.plot_time:
        table.plot_heatmap(
            col="time",
            title="Benchmark Runtime (s)",
            colorbar_label="Runtime (s)",
            outfile=args.time_out,
        )

    if args.plot_tokens:
        table.plot_heatmap(
            col="tokens",
            title="Required Tokens",
            colorbar_label="Tokens",
            outfile=args.tokens_out,
        )

    if args.plot_tools:
        assert len(args.plot_tools) == len(args.tools_out), (
            "--plot_tools and --tools_out should have the same length"
        )

        for tool in args.plot_tools:
            table.plot_heatmap(
                col="tools",
                tool=tool,
                title=TOOL_INFO[tool],
                colorbar_label="Tool Calls",
                outfile=f"{tool}_calls_heatmap.pdf",
            )
