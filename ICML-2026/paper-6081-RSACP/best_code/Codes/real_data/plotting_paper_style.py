from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RAW_DIR = OUT / "raw"
SUMMARY_DIR = OUT / "summary"
FIGURE_DIR = OUT / "figures"

METHOD_ORDER = ["SCP", "RSA-CP (OT) (Ours)", "SPI", "Synthetic-only"]
METHOD_LABEL = {
    "SCP": "SCP",
    "RSA-CP (OT) (Ours)": "RSA-CP",
    "SPI": "SPI",
    "Synthetic-only": "Syn-data-only",
}
COLORS = {
    "SCP": (31, 119, 180),
    "RSA-CP (OT) (Ours)": (255, 127, 14),
    "SPI": (44, 160, 44),
    "Synthetic-only": (214, 39, 40),
}


def font(size, bold=False, italic=False):
    if italic:
        candidates = ["timesi.ttf", "georgiai.ttf", "DejaVuSerif-Italic.ttf"]
    elif bold:
        candidates = ["arialbd.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf"]
    else:
        candidates = ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf"]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_TITLE = font(22)
FONT_LABEL = font(18)
FONT_SMALL = font(15)
FONT_TINY = font(12)
FONT_CAPTION = font(24, italic=True)


def save_both(img, base):
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    img.save(png)
    img.save(pdf, "PDF", resolution=300.0)
    return png, pdf


def read_csv_any(name: str, kind: str):
    dirs = [RAW_DIR if kind == "raw" else SUMMARY_DIR, OUT]
    for directory in dirs:
        path = directory / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(name)


def blend(color, frac=0.84):
    return tuple(int(c + (255 - c) * frac) for c in color)


def ypix(y, plot, ymin, ymax):
    left, top, right, bottom = plot
    return bottom - (float(y) - ymin) / (ymax - ymin) * (bottom - top)


def draw_rotated_label(img, text, x, y, size=(170, 26)):
    label = Image.new("RGBA", size, (255, 255, 255, 0))
    d = ImageDraw.Draw(label)
    d.text((0, 1), text, font=FONT_SMALL, fill=(20, 20, 20))
    label = label.rotate(90, expand=True)
    img.alpha_composite(label, (int(x), int(y)))


def draw_axes(draw, plot, ymin, ymax, xticks, xtick_labels, ylabel, xlabel=None, target=None, yticks=None):
    left, top, right, bottom = plot
    draw.rectangle([left, top, right, bottom], outline=(20, 20, 20), width=2)
    if yticks is None:
        yticks = np.linspace(ymin, ymax, 4)
    for y in yticks:
        yy = ypix(y, plot, ymin, ymax)
        draw.line([left, yy, right, yy], fill=(232, 232, 232), width=1)
        draw.line([left - 5, yy, left, yy], fill=(20, 20, 20), width=1)
        txt = f"{y:.2f}" if ymax <= 1.05 else f"{y:.1f}"
        tw = draw.textlength(txt, font=FONT_TINY)
        draw.text((left - 9 - tw, yy - 8), txt, fill=(20, 20, 20), font=FONT_TINY)
    if target is not None:
        yy = ypix(target, plot, ymin, ymax)
        x = left
        while x < right:
            draw.line([x, yy, min(x + 8, right), yy], fill=(25, 25, 25), width=2)
            x += 16
    for x, lab in zip(xticks, xtick_labels):
        draw.line([x, bottom, x, bottom + 5], fill=(20, 20, 20), width=1)
        tw = draw.textlength(str(lab), font=FONT_TINY)
        draw.text((x - tw / 2, bottom + 9), str(lab), fill=(20, 20, 20), font=FONT_TINY)
    if xlabel:
        tw = draw.textlength(xlabel, font=FONT_LABEL)
        draw.text(((left + right - tw) / 2, bottom + 34), xlabel, fill=(20, 20, 20), font=FONT_LABEL)
    # Keep y tick labels outside the panel. Long y labels are intentionally
    # omitted here because the panel title already names the metric.
    _ = ylabel


def line_summary(raw, x_col):
    return (
        raw.groupby(["Method", "alpha", x_col], dropna=False)
        .agg(
            Coverage_mean=("Coverage", "mean"),
            Coverage_sd=("Coverage", "std"),
            Coverage_n=("Coverage", "count"),
            Length_mean=("Length", "mean"),
            Length_sd=("Length", "std"),
            Length_n=("Length", "count"),
        )
        .reset_index()
    )


def finite_limits(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if np.isclose(lo, hi):
        hi = lo + 1.0
    pad = 0.10 * (hi - lo)
    return max(0.0, lo - pad), hi + pad


def draw_line_panel(draw, box, summary, x_col, y_col, sd_col, n_col, title, ylabel, xlabel, target=None, coverage=False):
    left, top, right, bottom = box
    plot = (left + 70, top + 38, right - 18, bottom - 58)
    xs = sorted(summary[x_col].unique())
    xmin, xmax = float(min(xs)), float(max(xs))

    def sx(v):
        if np.isclose(xmin, xmax):
            return (plot[0] + plot[2]) / 2
        return plot[0] + (float(v) - xmin) / (xmax - xmin) * (plot[2] - plot[0])

    if coverage:
        ymin, ymax = 0.70, 1.00
        yticks = [0.70, 0.80, 0.90, 1.00]
    else:
        ymeans = summary[y_col].to_numpy(dtype=float)
        ci = 1.96 * summary[sd_col].fillna(0).to_numpy(dtype=float) / np.sqrt(summary[n_col].to_numpy(dtype=float))
        ymin, ymax = finite_limits(np.concatenate([ymeans - ci, ymeans + ci]))
        yticks = np.linspace(ymin, ymax, 4)

    draw.text((left + 5, top), title, font=FONT_TITLE, fill=(20, 20, 20))
    draw_axes(draw, plot, ymin, ymax, [sx(x) for x in xs], [f"{x:g}" for x in xs], ylabel, xlabel=xlabel, target=target, yticks=yticks)

    series = []
    for method in METHOD_ORDER:
        sub = summary[summary["Method"] == method].sort_values(x_col)
        if sub.empty:
            continue
        pts = []
        upper = []
        lower = []
        color = COLORS[method]
        for _, row in sub.iterrows():
            x = sx(row[x_col])
            mean = float(row[y_col])
            se = float(row[sd_col]) / np.sqrt(float(row[n_col])) if row[n_col] else 0.0
            err = 1.96 * se
            lo = max(ymin, mean - err)
            hi = min(ymax, mean + err)
            yy = ypix(mean, plot, ymin, ymax)
            ylo = ypix(lo, plot, ymin, ymax)
            yhi = ypix(hi, plot, ymin, ymax)
            pts.append((x, yy))
            upper.append((x, yhi))
            lower.append((x, ylo))
        series.append((method, color, pts, upper, lower))

    # Draw all uncertainty ribbons first, then redraw every mean line on top.
    # This keeps later ribbons from washing out earlier method lines.
    for _, color, pts, upper, lower in series:
        if len(pts) >= 2:
            draw.polygon(upper + lower[::-1], fill=blend(color, 0.88))

    for _, color, pts, _, _ in series:
        draw.line(pts, fill=color, width=3)
        for x, yy in pts:
            draw.ellipse([x - 5, yy - 5, x + 5, yy + 5], fill=color, outline="white", width=1)


def draw_legend(draw, x, y):
    cursor = x
    for method in METHOD_ORDER:
        color = COLORS[method]
        draw.line([cursor, y + 8, cursor + 30, y + 8], fill=color, width=3)
        draw.ellipse([cursor + 11, y + 3, cursor + 21, y + 13], fill=color, outline="white")
        cursor += 40
        label = METHOD_LABEL[method]
        draw.text((cursor, y), label, fill=(20, 20, 20), font=FONT_LABEL)
        cursor += int(draw.textlength(label, font=FONT_LABEL)) + 34


def draw_sensitivity(raw_file, x_col, xlabel, base_name):
    raw = read_csv_any(raw_file, "raw")
    summary = line_summary(raw, x_col)
    summary.to_csv(SUMMARY_DIR / f"{base_name}_line_ribbon_summary.csv", index=False)
    img = Image.new("RGB", (1320, 820), "white")
    draw = ImageDraw.Draw(img)
    alphas = [0.05, 0.10]
    for r, alpha in enumerate(alphas):
        sub = summary[np.isclose(summary["alpha"], alpha)]
        y0 = 25 + r * 360
        draw_line_panel(
            draw,
            (30, y0, 640, y0 + 300),
            sub,
            x_col,
            "Coverage_mean",
            "Coverage_sd",
            "Coverage_n",
            f"Coverage (α={alpha:g})",
            "Coverage",
            xlabel,
            target=1.0 - alpha,
            coverage=True,
        )
        draw_line_panel(
            draw,
            (720, y0, 1290, y0 + 300),
            sub,
            x_col,
            "Length_mean",
            "Length_sd",
            "Length_n",
            f"Prediction set width (α={alpha:g})",
            "Prediction set width",
            xlabel,
            coverage=False,
        )
    draw_legend(draw, 430, 760)
    return save_both(img, FIGURE_DIR / base_name)


def draw_box(draw, x, width, values, plot, ymin, ymax, color):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        yy = plot[1] + 10
        draw.line([x - width / 2, yy, x + width / 2, yy], fill=color, width=2)
        draw.text((x - 10, yy + 5), "Inf", fill=color, font=FONT_TINY)
        return
    q1, med, q3 = np.percentile(finite, [25, 50, 75])
    lo, hi = np.percentile(finite, [5, 95])
    yq1, ymed, yq3 = [ypix(v, plot, ymin, ymax) for v in [q1, med, q3]]
    ylo, yhi = [ypix(v, plot, ymin, ymax) for v in [lo, hi]]
    draw.line([x, yhi, x, ylo], fill=color, width=2)
    draw.line([x - width * 0.28, yhi, x + width * 0.28, yhi], fill=color, width=2)
    draw.line([x - width * 0.28, ylo, x + width * 0.28, ylo], fill=color, width=2)
    draw.rectangle([x - width / 2, yq3, x + width / 2, yq1], fill=blend(color), outline=color, width=2)
    draw.line([x - width / 2, ymed, x + width / 2, ymed], fill=color, width=3)


def draw_box_axes(draw, plot, ymin, ymax, target=None, coverage=False):
    yticks = [0.70, 0.80, 0.90, 1.00] if coverage else np.linspace(ymin, ymax, 3)
    draw_axes(draw, plot, ymin, ymax, [], [], "", target=target, yticks=yticks)


def draw_age_cell(draw, cell, raw, age, alpha, caption):
    x0, y0, x1, y1 = cell
    draw.text((x0 + 6, y0 + 4), f"MEPS {age}, α={alpha:g}", font=FONT_SMALL, fill=(20, 20, 20))
    sub = raw[(raw["age_group"] == age) & np.isclose(raw["alpha"], alpha)]
    cov_plot = (x0 + 48, y0 + 38, x0 + 350, y0 + 210)
    len_plot = (x0 + 430, y0 + 38, x1 - 18, y0 + 210)
    draw.text((cov_plot[0], y0 + 22), "Coverage", font=FONT_TINY, fill=(20, 20, 20))
    draw.text((len_plot[0], y0 + 22), "Interval length", font=FONT_TINY, fill=(20, 20, 20))
    draw_box_axes(draw, cov_plot, 0.70, 1.00, target=1.0 - alpha, coverage=True)

    finite_len = sub["Length"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    ymin, ymax = finite_limits(finite_len)
    draw_box_axes(draw, len_plot, ymin, ymax, coverage=False)

    for plot, val_col, y_min, y_max in [(cov_plot, "Coverage", 0.70, 1.00), (len_plot, "Length", ymin, ymax)]:
        gap = (plot[2] - plot[0]) / (len(METHOD_ORDER) + 0.5)
        positions = [plot[0] + gap * (i + 0.75) for i in range(len(METHOD_ORDER))]
        for x, method in zip(positions, METHOD_ORDER):
            vals = sub.loc[sub["Method"] == method, val_col].to_numpy(dtype=float)
            draw_box(draw, x, min(34, gap * 0.52), vals, plot, y_min, y_max, COLORS[method])
        for x, method in zip(positions, METHOD_ORDER):
            lab = {
                "SCP": "SCP",
                "RSA-CP (OT) (Ours)": "RSA",
                "SPI": "SPI",
                "Synthetic-only": "Syn",
            }[method]
            tw = draw.textlength(lab, font=FONT_TINY)
            draw.text((x - tw / 2, plot[3] + 6), lab, fill=(20, 20, 20), font=FONT_TINY)

    tw = draw.textlength(caption, font=FONT_CAPTION)
    draw.text(((x0 + x1 - tw) / 2, y1 - 35), caption, font=FONT_CAPTION, fill=(20, 20, 20))


def draw_figure9_row4_col2():
    raw = read_csv_any("figure9_meps_age_groups_raw.csv", "raw")
    ages = ["0-20", "20-40", "40-60", "60-100"]
    alphas = [0.05, 0.10]
    captions = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    img = Image.new("RGB", (1600, 1900), "white")
    draw = ImageDraw.Draw(img)
    idx = 0
    for r, age in enumerate(ages):
        for c, alpha in enumerate(alphas):
            x0 = 50 + c * 780
            y0 = 35 + r * 445
            cell = (x0, y0, x0 + 700, y0 + 390)
            cap = f"{captions[idx]} {age.replace('60-100', '60+')}, α = {alpha:.2f}"
            draw_age_cell(draw, cell, raw, age, alpha, cap)
            idx += 1
    draw_legend(draw, 490, 1838)
    return save_both(img, FIGURE_DIR / "figure9_meps_age_groups_boxplot")


def main():
    p5 = draw_sensitivity(
        "figure5_imagenet_ncal_raw.csv",
        "n_cal",
        "Calibration size (n_cal)",
        "figure5_imagenet_ncal",
    )
    p8 = draw_sensitivity(
        "figure8_imagenet_nsyn_raw.csv",
        "n_ref",
        "Synthetic data size (n_cal_maj)",
        "figure8_imagenet_nsyn",
    )
    p9 = draw_figure9_row4_col2()
    print("Figure 5:", p5)
    print("Figure 8:", p8)
    print("Figure 9:", p9)


if __name__ == "__main__":
    main()
