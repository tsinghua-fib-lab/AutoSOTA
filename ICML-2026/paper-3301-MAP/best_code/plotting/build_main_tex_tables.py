#!/usr/bin/env python3
"""Generate the LaTeX table snippets used by main.tex.

Dynamic metric tables are produced by the experiment plotting scripts and then
collected here. Static appendix tables are generated directly from data declared
in this file so every table in main.tex has a reproducible source.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Set


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting.build_plane_sphere_metrics_table import (  # noqa: E402
    _parse_sectioned_table,
    _parse_single_table,
    build_table,
)


DEFAULT_TABLES_DIR = ROOT / "Diffusion_with_Manifold_Constraints" / "tables"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n")
    print(f"Wrote {path.relative_to(ROOT)}")


def _copy_table(source: Path, target: Path, *, strict: bool) -> None:
    if not source.exists():
        message = f"Missing table source: {source.relative_to(ROOT)}"
        if strict:
            raise FileNotFoundError(message)
        print(f"Skipping {target.name}: {message}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"Wrote {target.relative_to(ROOT)} from {source.relative_to(ROOT)}")


def _selected(name: str, only: Set[str]) -> bool:
    return not only or "all" in only or name in only


def write_notation_table(tables_dir: Path) -> None:
    rows = [
        (r"\multicolumn{2}{l}{\textit{Spaces and Manifolds}}", None),
        (r"$\mathbb{R}^d$", r"Ambient space (dimension $d$)"),
        (r"$\mathcal{M}$", r"$m$-dimensional constraint manifold embedded in $\mathbb{R}^d$"),
        (r"$m$", r"Intrinsic dimension of $\mathcal{M}$"),
        (r"$k = d - m$", r"Codimension of $\mathcal{M}$"),
        (r"$T_x\mathcal{M}$", r"Tangent space to $\mathcal{M}$ at point $x$"),
        (r"$N_x\mathcal{M}$", r"Normal space to $\mathcal{M}$ at point $x$"),
        (r"$\text{reach}(\mathcal{M})$", r"Global reach of manifold $\mathcal{M}$"),
        (r"$\tau(z)$", r"Pointwise reach function at $z \in \mathcal{M}$"),
        (r"$r < \text{reach}(\mathcal{M})$", r"Radius of reach tube $T_r(\mathcal{M})$"),
        (r"\midrule", None),
        (r"\multicolumn{2}{l}{\textit{Constraints}}", None),
        (r"$h: \mathbb{R}^d \to \mathbb{R}^m$", r"Constraint function defining $\mathcal{M} = \{x : h(x) = 0\}$"),
        (r"$J_h(x)$", r"Jacobian of constraint function at $x$"),
        (r"\midrule", None),
        (r"\multicolumn{2}{l}{\textit{Distributions and Measures}}", None),
        (r"$p_0(x)$", r"Original data distribution on $\mathcal{M}$"),
        (r"$p_\sigma(x)$", r"Perturbed (lifted) distribution in $\mathbb{R}^d$"),
        (r"$\tilde{p}_\sigma(x)$", r"Projected distribution $\Pi_\# p_\sigma$ on $\mathcal{M}$"),
        (r"$p_T(x)$", r"Terminal/latent distribution (typically $\mathcal{N}(0, I)$)"),
        (r"$\mu$", r"Probability measure/law"),
        (r"$\lambda^d$", r"$d$-dimensional Lebesgue measure"),
        (r"$\mathcal{H}^m$", r"$m$-dimensional Hausdorff measure"),
        (r"\midrule", None),
        (r"\multicolumn{2}{l}{\textit{Random Variables}}", None),
        (r"$Z$", r"Random variable with law $p_0$ on $\mathcal{M}$"),
        (r"$N | Z=z$", r"Conditional Gaussian noise in $N_z\mathcal{M}$"),
        (r"$X = Z + N$", r"Lifted random variable with law $p_\sigma$"),
        (r"$Y = \Pi(X)$", r"Projected random variable with law $\tilde{p}_\sigma$"),
        (r"\midrule", None),
        (r"\multicolumn{2}{l}{\textit{Operators and Functions}}", None),
        (r"$\Pi: \mathbb{R}^d \to \mathcal{M}$", r"Nearest-point projection onto $\mathcal{M}$"),
        (r"\midrule", None),
        (r"\multicolumn{2}{l}{\textit{Hyperparameters}}", None),
        (r"$\sigma$", r"Noise scale for perturbation"),
        (r"$t \in [0, T]$", r"Virtual time for diffusion process"),
        (r"$T$", r"Terminal time (typically $T=250$ in experiments)"),
        (r"\midrule", None),
        (r"\multicolumn{2}{l}{\textit{Metrics}}", None),
        (r"$\text{TV}(p, q)$", r"Total variation distance: $\frac{1}{2}\int_{\mathcal{M}} |p(x) - q(x)| d\mathcal{H}^m(x)$"),
        (r"$\text{JSD}(p \| q)$", r"Jensen-Shannon divergence"),
        (r"$\text{COV}$", r"Coverage metric"),
        (r"$\text{FID}$", r"Fr\'echet Inception Distance."),
        (r"$\text{RMSD}$", r"Root mean square deviation"),
        (r"$\text{MMD}$", r"Maximum Mean Discrepancy"),
    ]
    body = []
    for left, right in rows:
        if right is None and left == r"\midrule":
            body.append(left)
        elif right is None:
            body.append(left + r" \\")
        else:
            body.append(f"{left} & {right} " + r"\\")
    content = "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\label{tab:notation}",
        r"\begin{tabular}{cl}",
        r"\toprule",
        r"\textbf{Symbol} & \textbf{Description} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Summary of notation used throughout this paper.}",
        r"\end{table}",
    ])
    _write(tables_dir / "notation_glossary.tex", content)


def write_diffusion_config_table(tables_dir: Path) -> None:
    rows = [
        ("Time embedding", "Normalized", "Normalized", "Normalized", "MLP", "MLP"),
        (r"Time emb.\ dim.", "1", "1", "1", "64", "16"),
        (r"\# Training samples", r"100{,}000", r"100{,}000", r"100{,}000", r"10{,}000", r"20{,}000"),
        ("Batch size", "64", "64", "64", "32", "32"),
        ("Epochs", "200", "200", "200", "1000", "1000"),
        ("Hidden dimension", "64", "64", "128", "1024", "1024"),
    ]
    row_lines = [" & ".join(row) + r" \\" for row in rows]
    content = "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& \textbf{Plane} & \textbf{Sphere} & \textbf{Mesh} & \textbf{Images} & \textbf{Protein} \\",
        r"\midrule",
        *row_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Combined diffusion model training configurations for all tasks.}",
        r"\end{table}",
    ])
    _write(tables_dir / "diffusion_training_config.tex", content)


def write_parameter_count_table(tables_dir: Path) -> None:
    rows = [
        ("RealNVP", r"42{,}738", r"42{,}738", "--", "--", "--"),
        ("Glow", r"53{,}010", r"53{,}010", "--", "--", "--"),
        ("DDPM", r"4{,}675", r"4{,}675", r"17{,}539", r"1{,}088{,}769", r"112{,}353"),
    ]
    row_lines = [" & ".join(row) + r" \\" for row in rows]
    content = "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{\# Parameters} & \textbf{Plane} & \textbf{Sphere} & \textbf{Mesh} & \textbf{Images} & \textbf{Protein} \\",
        r"\midrule",
        *row_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Total number of trainable parameters for each model across all tasks.}",
        r"\end{table}",
    ])
    _write(tables_dir / "model_parameter_counts.tex", content)


def write_static_tables(tables_dir: Path) -> None:
    write_notation_table(tables_dir)
    write_diffusion_config_table(tables_dir)
    write_parameter_count_table(tables_dir)


def write_plane_sphere_table(args: argparse.Namespace, tables_dir: Path, *, strict: bool) -> None:
    sources = [
        args.plane_diffusion,
        args.sphere_diffusion,
        args.plane_nf_dir / "nf_extrinsic_metrics_table.tex",
        args.plane_nf_dir / "nf_intrinsic_metrics_table.tex",
        args.sphere_nf_dir / "nf_extrinsic_metrics_table.tex",
        args.sphere_nf_dir / "nf_intrinsic_metrics_table.tex",
    ]
    missing = [path for path in sources if not path.exists()]
    if missing:
        message = "Missing plane/sphere table inputs: " + ", ".join(str(path.relative_to(ROOT)) for path in missing)
        if strict:
            raise FileNotFoundError(message)
        print(f"Skipping plane_sphere_metrics.tex: {message}")
        return

    plane_diff_sections = _parse_sectioned_table(str(args.plane_diffusion))
    sphere_diff_sections = _parse_sectioned_table(str(args.sphere_diffusion))
    plane_nf = _parse_single_table(str(args.plane_nf_dir / "nf_extrinsic_metrics_table.tex"))
    plane_nf_intrinsic = _parse_single_table(str(args.plane_nf_dir / "nf_intrinsic_metrics_table.tex"))
    sphere_nf = _parse_single_table(str(args.sphere_nf_dir / "nf_extrinsic_metrics_table.tex"))
    sphere_nf_intrinsic = _parse_single_table(str(args.sphere_nf_dir / "nf_intrinsic_metrics_table.tex"))

    target = tables_dir / "plane_sphere_metrics.tex"
    build_table(
        plane_diff_sections["extrinsic"],
        plane_nf,
        sphere_diff_sections["extrinsic"],
        sphere_nf,
        str(target),
        plane_diff_sections["intrinsic"],
        plane_nf_intrinsic,
        sphere_diff_sections["intrinsic"],
        sphere_nf_intrinsic,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table snippets used by main.tex.")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--only", default="all", help="Comma-separated subset: all,static,plane_sphere,mesh,mnist,protein")
    parser.add_argument("--strict", action="store_true", help="Fail if a requested dynamic table input is missing.")
    parser.add_argument("--plane-diffusion", type=Path, default=ROOT / "results/smileyface_plane/metrics_table_seed_42.tex")
    parser.add_argument("--plane-nf-dir", type=Path, default=ROOT / "results/smileyface_plane/nf_plane")
    parser.add_argument("--sphere-diffusion", type=Path, default=ROOT / "results/smileyface_sphere/metrics_table.tex")
    parser.add_argument("--sphere-nf-dir", type=Path, default=ROOT / "results/smileyface_sphere/nf_noise_0.05")
    parser.add_argument("--mesh-source", type=Path, default=ROOT / "results/bunny/metrics_table.tex")
    parser.add_argument("--mnist-source", type=Path, default=ROOT / "results/mnist/metrics_table.tex")
    parser.add_argument("--protein-source", type=Path, default=ROOT / "results/protein/metrics_table.tex")
    args = parser.parse_args()

    only = {item.strip() for item in args.only.split(",") if item.strip()}
    tables_dir = args.tables_dir

    if _selected("static", only):
        write_static_tables(tables_dir)
    if _selected("plane_sphere", only):
        write_plane_sphere_table(args, tables_dir, strict=args.strict)
    if _selected("mesh", only):
        _copy_table(args.mesh_source, tables_dir / "mesh_metrics.tex", strict=args.strict)
    if _selected("mnist", only):
        _copy_table(args.mnist_source, tables_dir / "mnist_metrics.tex", strict=args.strict)
    if _selected("protein", only):
        _copy_table(args.protein_source, tables_dir / "protein_metrics.tex", strict=args.strict)


if __name__ == "__main__":
    main()
