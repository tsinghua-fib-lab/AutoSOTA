import csv
import os

import numpy as np


def _is_finite(value):
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def _fmt_fixed(value, digits=4):
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_fixed_bold(value, digits=4):
    if not _is_finite(value):
        return "n/a"
    return rf"\textbf{{{float(value):.{digits}f}}}"


def _fmt_sci(value, digits=3):
    if not _is_finite(value):
        return "n/a"
    mantissa, exponent = f"{float(value):.{digits}e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def _fmt_sci_math(value, digits=3, bold=False):
    if not _is_finite(value):
        return "n/a"
    body = _fmt_sci(value, digits=digits)
    if bold:
        return rf"$\mathbf{{{body}}}$"
    return rf"${body}$"


def _write_csv(out_tex_path, rows, columns):
    csv_path = os.path.splitext(out_tex_path)[0] + ".csv"
    os.makedirs(os.path.dirname(out_tex_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method"] + columns)
        for row in rows:
            writer.writerow([row["method"]] + [row.get(column, "") for column in columns])


def write_mesh_metrics_table(rows, out_tex_path, caption, label):
    _write_csv(out_tex_path, rows, ["Train time (s/epoch)", "Sampling time (s)", "COV", "JSD", "TVD"])

    cov_best = max((row["COV"] for row in rows if _is_finite(row.get("COV"))), default=None)
    jsd_best = min((row["JSD"] for row in rows if _is_finite(row.get("JSD"))), default=None)
    tvd_best = min((row["TVD"] for row in rows if _is_finite(row.get("TVD"))), default=None)

    os.makedirs(os.path.dirname(out_tex_path) or ".", exist_ok=True)
    with open(out_tex_path, "w") as handle:
        handle.write("\\begin{table}[h]\n")
        handle.write("\\scriptsize\n")
        handle.write("\\centering\n")
        handle.write("\\begin{tabular}{l     >{\\centering\\arraybackslash}p{0.8cm}\n")
        handle.write("    >{\\centering\\arraybackslash}p{0.8cm}\n")
        handle.write("    >{\\centering\\arraybackslash}p{0.8cm}\n")
        handle.write("    >{\\centering\\arraybackslash}p{0.8cm}\n")
        handle.write("    >{\\centering\\arraybackslash}p{0.8cm}}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Train time   & Sampling time   & COV $\\uparrow$ & JSD $\\downarrow$ & TVD $\\downarrow$\\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            cov = _fmt_fixed_bold(row["COV"], 4) if _is_finite(cov_best) and float(row["COV"]) == float(cov_best) else _fmt_fixed(row["COV"], 4)
            jsd = _fmt_fixed_bold(row["JSD"], 4) if _is_finite(jsd_best) and float(row["JSD"]) == float(jsd_best) else _fmt_fixed(row["JSD"], 4)
            tvd = _fmt_fixed_bold(row["TVD"], 4) if _is_finite(tvd_best) and float(row["TVD"]) == float(tvd_best) else _fmt_fixed(row["TVD"], 4)
            handle.write(
                f"{row['method']} & {_fmt_fixed(row['Train time (s/epoch)'], 4)} & {_fmt_fixed(row['Sampling time (s)'], 4)} & {cov} & {jsd} & {tvd} \\\\ \n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write(f"\\caption{{{caption}}}\n")
        if label:
            handle.write(f"\\label{{{label}}}\n")
        handle.write("\\end{table}\n")


def write_mnist_metrics_table(rows, out_tex_path, caption):
    _write_csv(out_tex_path, rows, ["Train time (s/epoch)", "Sampling time (s)", "COV", "FID", "Class JSD"])

    cov_best = max((row["COV"] for row in rows if _is_finite(row.get("COV"))), default=None)
    fid_best = min((row["FID"] for row in rows if _is_finite(row.get("FID"))), default=None)
    jsd_best = min((row["Class JSD"] for row in rows if _is_finite(row.get("Class JSD"))), default=None)

    os.makedirs(os.path.dirname(out_tex_path) or ".", exist_ok=True)
    with open(out_tex_path, "w") as handle:
        handle.write("\\begin{table}[h]\n")
        handle.write("\\scriptsize\n")
        handle.write("\\centering\n")
        handle.write("\\begin{tabular}{l     >{\\centering\\arraybackslash}p{0.4cm}     >{\\centering\\arraybackslash}p{0.8cm}     >{\\centering\\arraybackslash}p{0.8cm}     >{\\centering\\arraybackslash}p{1.5cm} >{\\centering\\arraybackslash}p{0.8cm}}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Train time   & Sampling time   & COV $\\uparrow$ & FID $\\downarrow$ & Class JSD $\\downarrow$ \\\\ \n")
        handle.write("\\midrule\n")
        for row in rows:
            cov = _fmt_fixed_bold(row["COV"], 4) if _is_finite(cov_best) and float(row["COV"]) == float(cov_best) else _fmt_fixed(row["COV"], 4)
            fid = _fmt_sci_math(row["FID"], 3, bold=_is_finite(fid_best) and float(row["FID"]) == float(fid_best))
            jsd = _fmt_fixed_bold(row["Class JSD"], 4) if _is_finite(jsd_best) and float(row["Class JSD"]) == float(jsd_best) else _fmt_fixed(row["Class JSD"], 4)
            handle.write(
                f"{row['method']} & {_fmt_fixed(row['Train time (s/epoch)'], 4)} & {_fmt_fixed(row['Sampling time (s)'], 4)} & {cov} & {fid} & {jsd} \\\\ \n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write(f"\\caption{{{caption}}}\n")
        handle.write("\\end{table}\n")


def write_protein_metrics_table(rows, out_tex_path, caption, label):
    _write_csv(out_tex_path, rows, ["Train time (s/epoch)", "Sampling time (s)", "COV", "Pairwise RMSD", "MMD"])

    cov_best = max((row["COV"] for row in rows if _is_finite(row.get("COV"))), default=None)
    rmsd_best = max((row["Pairwise RMSD"] for row in rows if _is_finite(row.get("Pairwise RMSD"))), default=None)
    mmd_best = min((row["MMD"] for row in rows if _is_finite(row.get("MMD"))), default=None)

    os.makedirs(os.path.dirname(out_tex_path) or ".", exist_ok=True)
    with open(out_tex_path, "w") as handle:
        handle.write("\\begin{table}[h]\n")
        handle.write("\\scriptsize\n")
        handle.write("\\centering\n")
        handle.write("\\begin{tabular}{l     >{\\centering\\arraybackslash}p{0.8cm}     >{\\centering\\arraybackslash}p{0.8cm}     >{\\centering\\arraybackslash}p{0.4cm}     >{\\centering\\arraybackslash}p{0.4cm}     >{\\centering\\arraybackslash}p{1.8cm}}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Train time & Sampling time & COV & Pairwise RMSD & MMD \\\\ \n")
        handle.write("\\midrule\n")
        for row in rows:
            cov = _fmt_fixed_bold(row["COV"], 4) if _is_finite(cov_best) and float(row["COV"]) == float(cov_best) else _fmt_fixed(row["COV"], 4)
            rmsd = _fmt_fixed_bold(row["Pairwise RMSD"], 4) if _is_finite(rmsd_best) and float(row["Pairwise RMSD"]) == float(rmsd_best) else _fmt_fixed(row["Pairwise RMSD"], 4)
            mmd = _fmt_sci_math(row["MMD"], 2, bold=_is_finite(mmd_best) and float(row["MMD"]) == float(mmd_best))
            handle.write(
                f"{row['method']} & {_fmt_fixed(row['Train time (s/epoch)'], 4)} & {_fmt_fixed(row['Sampling time (s)'], 4)} & {cov} & {rmsd} & {mmd} \\\\ \n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write(f"\\caption{{{caption}}}\n")
        if label:
            handle.write(f"\\label{{{label}}}\n")
        handle.write("\\end{table}\n")
