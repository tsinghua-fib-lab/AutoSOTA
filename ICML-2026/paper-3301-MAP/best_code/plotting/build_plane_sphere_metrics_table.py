#!/usr/bin/env python3
"""Build the integrated plane/sphere metrics table used in main.tex.

The per-task plotting scripts already emit LaTeX tables for the diffusion and
NF experiments. This script stitches those outputs into the combined table for
tab:PlaneSphereMetrics without changing the source evaluation logic.
"""

import argparse
import csv
import os


def _clean(cell):
    return cell.strip().rstrip("\\").strip()


def _w(handle, line):
    handle.write(line)
    handle.write("\n")


def _parse_sectioned_table(path):
    sections = {"extrinsic": {}, "intrinsic": {}}
    current = None
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if "Additional table" in line or "Num NaN/Inf" in line:
                current = None
                continue
            if "Method &" in line:
                if "Train" in line or "Sampling" in line:
                    current = "extrinsic"
                elif "COV" in line and ("JSD" in line or "TVD" in line):
                    current = "intrinsic"
                continue
            if "Extrinsic metrics" in line or "General metrics" in line or "Intrinsic metrics" in line:
                continue
            if current is None:
                continue
            if "&" not in line or "\\" not in line:
                continue
            if any(token in line for token in ("\\toprule", "\\midrule", "\\bottomrule", "\\cline")):
                continue
            cells = [_clean(part) for part in line.split("&")]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            if "multirow" in cells[0] or "rotatebox" in cells[0]:
                continue
            sections[current][cells[0]] = cells[1:]
    return sections

def _parse_single_table(path):
    rows = {}
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            # Stop parsing when an additional/summary table appears later in the file
            if "Additional table" in line or "Num NaN/Inf" in line or line.startswith("\\begin{table"):
                break
            if "&" not in line or "\\" not in line:
                continue
            if any(token in line for token in ("\\toprule", "\\midrule", "\\bottomrule", "Method &")):
                continue
            cells = [_clean(part) for part in line.split("&")]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            if "multirow" in cells[0] or "rotatebox" in cells[0]:
                continue
            rows[cells[0]] = cells[1:]
    return rows


def _normalize_label(label):
    return (
        label.replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace("\\", "")
        .replace("tilde", "")
        .replace(" ", "")
        .lower()
    )


def _row(rows, aliases):
    if isinstance(aliases, str):
        aliases = [aliases]
    for label in aliases:
        if label in rows:
            return rows[label]
    normalized = {_normalize_label(key): value for key, value in rows.items()}
    for label in aliases:
        value = normalized.get(_normalize_label(label))
        if value is not None:
            return value
    raise KeyError(f"Missing row matching any of {aliases!r}; available rows: {sorted(rows)}")


def _write_rows(handle, row_specs, rows):
    for label, prefix, aliases in row_specs:
        _w(handle, '{}{} & {} \\\\'.format(prefix, label, ' & '.join(_row(rows, aliases or label))))


def build_table(
    plane_diff,
    plane_nf,
    sphere_diff,
    sphere_nf,
    out_path,
    plane_diff_intrinsic=None,
    plane_nf_intrinsic=None,
    sphere_diff_intrinsic=None,
    sphere_nf_intrinsic=None,
):
    plane_diff_intrinsic = plane_diff_intrinsic or plane_diff
    plane_nf_intrinsic = plane_nf_intrinsic or plane_nf
    sphere_diff_intrinsic = sphere_diff_intrinsic or sphere_diff
    sphere_nf_intrinsic = sphere_nf_intrinsic or sphere_nf
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        _w(handle, "\\begin{table}[H]")
        _w(handle, "\\scriptsize")
        _w(handle, "\\centering")
        _w(handle, "\\setlength{\\tabcolsep}{3pt}")
        _w(handle, "\\renewcommand{\\arraystretch}{1.1}")
        _w(handle, "\\begin{tabular}{c  c  l  >{\\centering\\arraybackslash}p{0.8cm}  >{\\centering\\arraybackslash}p{1.0cm}  >{\\centering\\arraybackslash}p{0.8cm}  >{\\centering\\arraybackslash}p{0.8cm}  >{\\centering\\arraybackslash}p{0.8cm}}")
        _w(handle, "\\hline")
        _w(handle, ' & & Method & Train time & Sampling time & COV $\\uparrow$ & JSD $\\downarrow$ & TVD $\\downarrow$ \\\\')
        _w(handle, "\\hline")
        _w(handle, "% ===================== PLANE – Diffusion ===================== %")
        _w(handle, "\\multirow{7}{*}{\\rotatebox{90}{\\textbf{Plane}}}%")
        _w(handle, " & \\multirow{4}{*}{\\rotatebox{90}{\\textbf{DMs}}}")
        _write_rows(
            handle,
            [
                ("PDM", "&", None),
                ("PIDM", "&&", None),
                ("$\\tilde{p}_{\\sigma}$", "&&", ["$p_{\\sigma}$", "\\tilde{p}_{\\sigma}", "$\\tilde{p}_{\\sigma}$"]),
                ("DDPM", "&&", None),
                ("DDPM (proj.)", "&&", None),
                ("DDPM (proj., iso.)", "&&", None),
            ],
            plane_diff,
        )
        _w(handle, "\\cline{2-8}")
        _w(handle, "% ===================== PLANE – Normalizing Flows ============= %")
        _w(handle, " & \\multirow{6}{*}{\\rotatebox{90}{\\textbf{NFs}}}")
        _write_rows(
            handle,
            [
                ("Glow (iso.)", "&", None),
                ("Glow ($\\tilde{p}_{\\sigma}$, ours)", "&&", ["Glow ($p_{\\sigma}$, ours)", "Glow ($\\tilde{p}_{\\sigma}$, ours)"]),
                ("Glow (proj.)", "&&", None),
                ("Glow", "&&", None),
                ("RealNVP (iso.)", "&&", None),
                ("RealNVP ($\\tilde{p}_{\\sigma}$, ours)", "&&", ["RealNVP ($p_{\\sigma}$, ours)", "RealNVP ($\\tilde{p}_{\\sigma}$, ours)"]),
                ("RealNVP (proj.)", "&&", None),
                ("RealNVP", "&&", None),
            ],
            plane_nf,
        )
        _w(handle, "\\hline")
        _w(handle, "% ===================== SPHERE – Diffusion ==================== %")
        _w(handle, "\\multirow{8}{*}{\\rotatebox{90}{\\textbf{Sphere}}}%")
        _w(handle, " & \\multirow{4}{*}{\\rotatebox{90}{\\textbf{DMs}}}")
        _write_rows(
            handle,
            [
                ("PDM", "&", None),
                ("PIDM", "&&", None),
                ("$\\tilde{p}_{\\sigma}$", "&&", ["$p_{\\sigma}$", "\\tilde{p}_{\\sigma}", "$\\tilde{p}_{\\sigma}$"]),
                ("DDPM", "&&", None),
                ("DDPM (proj.)", "&&", None),
                ("DDPM (proj., iso.)", "&&", None),
            ],
            sphere_diff,
        )
        _w(handle, "\\cline{2-8}")
        _w(handle, "% ===================== SPHERE – Normalizing Flows ============ %")
        _w(handle, " & \\multirow{7}{*}{\\rotatebox{90}{\\textbf{NFs}}}")
        _write_rows(
            handle,
            [
                ("Glow", "&", None),
                ("Glow ($\\tilde{p}_{\\sigma}$, ours)", "&&", ["Glow ($p_{\\sigma}$, ours)", "Glow ($\\tilde{p}_{\\sigma}$, ours)"]),
                ("Glow (proj.)", "&&", None),
                ("Glow (iso.)", "&&", None),
                ("RealNVP", "&&", None),
                ("RealNVP ($\\tilde{p}_{\\sigma}$, ours)", "&&", ["RealNVP ($p_{\\sigma}$, ours)", "RealNVP ($\\tilde{p}_{\\sigma}$, ours)"]),
                ("RealNVP (proj.)", "&&", None),
                ("RealNVP (iso.)", "&&", None),
            ],
            sphere_nf,
        )
        _w(handle, "\\hline")
        _w(handle, "\\end{tabular}")
        _w(handle, "\\caption*{Extrinsic metrics.}")
        _w(handle, "")
        _w(handle, "\\begin{tabular}{c")
        _w(handle, "    c")
        _w(handle, "    l")
        _w(handle, "    >{\\centering\\arraybackslash}p{0.9cm}  % Intrinsic metric 1")
        _w(handle, "    >{\\centering\\arraybackslash}p{0.9cm}  % Intrinsic metric 2")
        _w(handle, "    >{\\centering\\arraybackslash}p{0.9cm}  % Intrinsic metric 3 (optional)")
        _w(handle, "}")
        _w(handle, "\\hline")
        _w(handle, ' & & Method & COV $\\uparrow$ & JSD $\\downarrow$ & TVD $\\downarrow$ \\\\')
        _w(handle, "\\hline")
        _w(handle, "% ===================== PLANE – Diffusion ===================== %")
        _w(handle, "\\multirow{7}{*}{\\rotatebox{90}{\\textbf{Plane}}}%")
        _w(handle, " & \\multirow{4}{*}{\\rotatebox{90}{\\textbf{DMs}}}")
        _write_rows(
            handle,
            [
                ("PDM", "&", None),
                ("$\\tilde{p}_{\\sigma}$", "&&", ["$p_{\\sigma}$", "\\tilde{p}_{\\sigma}", "$\\tilde{p}_{\\sigma}$"]),
                ("DDPM (proj.)", "&&", None),
                ("DDPM (proj., iso.)", "&&", None),
            ],
            plane_diff_intrinsic,
        )
        _w(handle, "\\cline{2-6}")
        _w(handle, "% ===================== PLANE – Normalizing Flows ============= %")
        _w(handle, " & \\multirow{6}{*}{\\rotatebox{90}{\\textbf{NFs}}}")
        _write_rows(
            handle,
            [
                ("Glow (iso.)", "&", None),
                ("Glow ($\\tilde{p}_{\\sigma}$, ours)", "&&", ["Glow ($p_{\\sigma}$, ours)", "Glow ($\\tilde{p}_{\\sigma}$, ours)"]),
                ("Glow (proj.)", "&&", None),
                ("RealNVP (iso.)", "&&", None),
                ("RealNVP ($\\tilde{p}_{\\sigma}$, ours)", "&&", ["RealNVP ($p_{\\sigma}$, ours)", "RealNVP ($\\tilde{p}_{\\sigma}$, ours)"]),
                ("RealNVP (proj.)", "&&", None),
            ],
            plane_nf_intrinsic,
        )
        _w(handle, "\\hline")
        _w(handle, "% ===================== SPHERE – Diffusion ==================== %")
        _w(handle, "\\multirow{8}{*}{\\rotatebox{90}{\\textbf{Sphere}}}%")
        _w(handle, " & \\multirow{4}{*}{\\rotatebox{90}{\\textbf{DMs}}}")
        _write_rows(
            handle,
            [
                ("PDM", "&", None),
                ("$\\tilde{p}_{\\sigma}$", "&&", ["$p_{\\sigma}$", "\\tilde{p}_{\\sigma}", "$\\tilde{p}_{\\sigma}$"]),
                ("DDPM (proj.)", "&&", None),
                ("DDPM (proj., iso.)", "&&", None),
            ],
            sphere_diff_intrinsic,
        )
        _w(handle, "\\cline{2-6}")
        _w(handle, "% ===================== SPHERE – Normalizing Flows ============ %")
        _w(handle, " & \\multirow{7}{*}{\\rotatebox{90}{\\textbf{NFs}}}")
        _write_rows(
            handle,
            [
                ("Glow ($\\tilde{p}_{\\sigma}$, ours)", "&", ["Glow ($p_{\\sigma}$, ours)", "Glow ($\\tilde{p}_{\\sigma}$, ours)"]),
                ("Glow (proj.)", "&&", None),
                ("Glow (iso.)", "&&", None),
                ("RealNVP ($\\tilde{p}_{\\sigma}$, ours)", "&&", ["RealNVP ($p_{\\sigma}$, ours)", "RealNVP ($\\tilde{p}_{\\sigma}$, ours)"]),
                ("RealNVP (proj.)", "&&", None),
                ("RealNVP (iso.)", "&&", None),
            ],
            sphere_nf_intrinsic,
        )
        _w(handle, "\\hline")
        _w(handle, "\\end{tabular}")
        _w(handle, "\\caption*{Intrinsic metrics (using 2-D representations).}")
        _w(handle, "")
        _w(handle, "\\caption{Metrics for plane and sphere tasks at $\\sigma = 0.05$. Among these experiments, learning $p_{\\sigma}$ \\emph{always} improves the sampled distribution compared to either learning $p_{0}$ or modifying the learning or sampling algorithms for the diffusion models. For NF approaches, we highlight the best-performing method within each architecture.}")
        _w(handle, "\\label{tab:PlaneSphereMetrics}")
        _w(handle, "\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Build the integrated plane/sphere metrics table from script outputs.")
    parser.add_argument("--plane-diffusion", default="results/smileyface_plane/metrics_table_seed_42.tex")
    parser.add_argument("--plane-nf-dir", default="results/smileyface_plane/nf_plane")
    parser.add_argument("--sphere-diffusion", default="results/smileyface_sphere/metrics_table.tex")
    parser.add_argument("--sphere-nf-dir", default="results/smileyface_sphere/nf_noise_0.05")
    parser.add_argument("--out", default="results/plane_sphere_metrics_table.tex")
    parser.add_argument("--csv-out", default="results/plane_sphere_metrics_table.csv")
    args = parser.parse_args()

    plane_diff_sections = _parse_sectioned_table(args.plane_diffusion)
    sphere_diff_sections = _parse_sectioned_table(args.sphere_diffusion)
    plane_diff = plane_diff_sections["extrinsic"]
    sphere_diff = sphere_diff_sections["extrinsic"]
    plane_nf = _parse_single_table(os.path.join(args.plane_nf_dir, "nf_extrinsic_metrics_table.tex"))
    sphere_nf = _parse_single_table(os.path.join(args.sphere_nf_dir, "nf_extrinsic_metrics_table.tex"))
    plane_nf_intrinsic = _parse_single_table(os.path.join(args.plane_nf_dir, "nf_intrinsic_metrics_table.tex"))
    sphere_nf_intrinsic = _parse_single_table(os.path.join(args.sphere_nf_dir, "nf_intrinsic_metrics_table.tex"))

    build_table(
        plane_diff,
        plane_nf,
        sphere_diff,
        sphere_nf,
        args.out,
        plane_diff_sections["intrinsic"],
        plane_nf_intrinsic,
        sphere_diff_sections["intrinsic"],
        sphere_nf_intrinsic,
    )

    with open(args.csv_out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "method", "c0", "c1", "c2", "c3", "c4"])
        for section, rows in [
            ("plane_diffusion", plane_diff),
            ("plane_nf", plane_nf),
            ("sphere_diffusion", sphere_diff),
            ("sphere_nf", sphere_nf),
        ]:
            for method, values in rows.items():
                writer.writerow([section, method] + values)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
