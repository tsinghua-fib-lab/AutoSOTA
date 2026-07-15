from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RAW_DIR = OUT / "raw"
SUMMARY_DIR = OUT / "summary"
FIGURE_DIR = OUT / "figures"

METHOD_ORDER = ["SCP", "Synthetic-only", "SPI", "RSA-CP (OT) (Ours)"]
METHOD_SHORT = {
    "SCP": "SCP",
    "Synthetic-only": "Synthetic-only",
    "SPI": "SPI",
    "RSA-CP (OT) (Ours)": "RSA-CP (OT)",
}
METHOD_COLORS = {
    "SCP": (49, 93, 156),
    "Synthetic-only": (224, 126, 37),
    "SPI": (42, 143, 88),
    "RSA-CP (OT) (Ours)": (198, 57, 52),
}


def font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT = font(19)
FONT_SMALL = font(15)
FONT_TINY = font(13)
FONT_BOLD = font(22, True)


def blend(color, frac=0.82):
    return tuple(int(c + (255 - c) * frac) for c in color)


def save_both(img, base: Path):
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


def ypix(y, plot, ymin, ymax):
    left, top, right, bottom = plot
    y = float(y)
    return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)


def draw_y_axis(draw, plot, ymin, ymax, ticks, label):
    left, top, right, bottom = plot
    draw.rectangle([left, top, right, bottom], outline=(35, 35, 35), width=2)
    for t in ticks:
        yy = ypix(t, plot, ymin, ymax)
        draw.line([left, yy, right, yy], fill=(225, 225, 225), width=1)
        draw.line([left - 5, yy, left, yy], fill=(35, 35, 35), width=1)
        txt = f"{t:.2f}" if ymax <= 1.05 else f"{t:.1f}"
        draw.text((left - 48, yy - 8), txt, fill=(45, 45, 45), font=FONT_TINY)
    # y-axis label
    label_img = Image.new("RGBA", (180, 24), (255, 255, 255, 0))
    ld = ImageDraw.Draw(label_img)
    ld.text((0, 0), label, fill=(30, 30, 30), font=FONT_SMALL)
    label_img = label_img.rotate(90, expand=True)
    draw.bitmap((left - 80, top + (bottom - top - label_img.height) / 2), label_img, fill=(30, 30, 30))


def draw_x_labels(draw, plot, labels, positions, x_label=True):
    left, top, right, bottom = plot
    for x, label in zip(positions, labels):
        draw.line([x, bottom, x, bottom + 5], fill=(35, 35, 35), width=1)
        tw = draw.textlength(label, font=FONT_TINY)
        draw.text((x - tw / 2, bottom + 11), label, fill=(35, 35, 35), font=FONT_TINY)
    if x_label:
        txt = "Method"
        tw = draw.textlength(txt, font=FONT_SMALL)
        draw.text(((left + right - tw) / 2, bottom + 36), txt, fill=(30, 30, 30), font=FONT_SMALL)


def finite_limits(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(vals.min())
    hi = float(vals.max())
    if np.isclose(lo, hi):
        hi = lo + 1.0
    pad = 0.08 * (hi - lo)
    return max(0.0, lo - pad), hi + pad


def draw_box(draw, x, width, values, plot, ymin, ymax, color):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        yy = plot[1] + 14
        draw.line([x - width / 2, yy, x + width / 2, yy], fill=color, width=3)
        draw.text((x - 11, yy + 5), "Inf", fill=color, font=FONT_TINY)
        return
    q1, med, q3 = np.percentile(finite, [25, 50, 75])
    lo, hi = np.percentile(finite, [5, 95])
    yq1 = ypix(q1, plot, ymin, ymax)
    ymed = ypix(med, plot, ymin, ymax)
    yq3 = ypix(q3, plot, ymin, ymax)
    ylo = ypix(lo, plot, ymin, ymax)
    yhi = ypix(hi, plot, ymin, ymax)
    lw = 3 if color == METHOD_COLORS["RSA-CP (OT) (Ours)"] else 2
    draw.line([x, yhi, x, ylo], fill=color, width=lw)
    draw.line([x - width * 0.30, yhi, x + width * 0.30, yhi], fill=color, width=lw)
    draw.line([x - width * 0.30, ylo, x + width * 0.30, ylo], fill=color, width=lw)
    draw.rectangle([x - width / 2, yq3, x + width / 2, yq1], fill=blend(color), outline=color, width=lw)
    draw.line([x - width / 2, ymed, x + width / 2, ymed], fill=color, width=lw + 1)


def draw_box_panel(draw, box, df, value_col, title, ylabel, coverage=False, target=None, show_x_labels=True):
    left, top, right, bottom = box
    plot = (left + 72, top + 36, right - 20, bottom - 60)
    if coverage:
        ymin, ymax = 0.70, 1.00
        ticks = [0.70, 0.80, 0.90, 1.00]
    else:
        ymin, ymax = finite_limits(df[value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
        ticks = np.linspace(ymin, ymax, 4)
    draw.text((left + 5, top), title, fill=(25, 25, 25), font=FONT_BOLD)
    draw_y_axis(draw, plot, ymin, ymax, ticks, ylabel)
    if target is not None:
        yy = ypix(target, plot, ymin, ymax)
        x = plot[0]
        while x < plot[2]:
            draw.line([x, yy, min(x + 9, plot[2]), yy], fill=(70, 70, 70), width=2)
            x += 18
    gap = (plot[2] - plot[0]) / (len(METHOD_ORDER) + 0.5)
    positions = [plot[0] + gap * (i + 0.75) for i in range(len(METHOD_ORDER))]
    labels = [METHOD_SHORT[m] for m in METHOD_ORDER]
    box_w = min(48, gap * 0.55)
    for x, method in zip(positions, METHOD_ORDER):
        vals = df.loc[df["Method"] == method, value_col].to_numpy(dtype=float)
        draw_box(draw, x, box_w, vals, plot, ymin, ymax, METHOD_COLORS[method])
    if show_x_labels:
        draw_x_labels(draw, plot, labels, positions, x_label=True)


def draw_legend(draw, x, y):
    cursor = x
    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        draw.rectangle([cursor, y + 2, cursor + 28, y + 17], fill=blend(color), outline=color, width=2)
        cursor += 38
        draw.text((cursor, y), method, fill=(25, 25, 25), font=FONT_SMALL)
        cursor += int(draw.textlength(method, font=FONT_SMALL)) + 30


def restyle_figure4():
    raw = read_csv_any("figure4_imagenet_main_raw.csv", "raw")
    img = Image.new("RGB", (1640, 1040), "white")
    draw = ImageDraw.Draw(img)
    alphas = [0.05, 0.10]
    for col, alpha in enumerate(alphas):
        x0 = 50 + col * 790
        sub = raw[np.isclose(raw["alpha"], alpha)]
        draw_box_panel(
            draw,
            (x0, 35, x0 + 720, 455),
            sub,
            "Coverage",
            f"α = {alpha:g}",
            "Coverage",
            coverage=True,
            target=1.0 - alpha,
            show_x_labels=False,
        )
        draw_box_panel(
            draw,
            (x0, 510, x0 + 720, 930),
            sub,
            "Length",
            f"α = {alpha:g}",
            "Prediction set size",
            show_x_labels=True,
        )
    draw_legend(draw, 360, 985)
    return save_both(img, FIGURE_DIR / "figure4_imagenet_main_boxplot")


def restyle_figure9():
    raw = read_csv_any("figure9_meps_age_groups_raw.csv", "raw")
    ages = ["0-20", "20-40", "40-60", "60-100"]
    alphas = [0.05, 0.10]
    panels = [(a, al) for a in ages for al in alphas]
    img = Image.new("RGB", (3260, 1220), "white")
    draw = ImageDraw.Draw(img)
    for j, (age, alpha) in enumerate(panels):
        x0 = 35 + j * 400
        sub = raw[(raw["age_group"] == age) & np.isclose(raw["alpha"], alpha)]
        title = f"Age {age}, α={alpha:g}"
        draw_box_panel(
            draw,
            (x0, 35, x0 + 370, 500),
            sub,
            "Coverage",
            title,
            "Coverage",
            coverage=True,
            target=1.0 - alpha,
            show_x_labels=False,
        )
        draw_box_panel(
            draw,
            (x0, 560, x0 + 370, 1030),
            sub,
            "Length",
            title,
            "Interval width",
            show_x_labels=False,
        )
    draw.text((45, 1070), "Coverage panels use y-axis range [0.70, 1.00]; dashed lines are nominal coverage.", fill=(70, 70, 70), font=FONT_SMALL)
    draw_legend(draw, 955, 1145)
    return save_both(img, FIGURE_DIR / "figure9_meps_age_groups_boxplot_aesthetic")


def draw_line_axis(draw, plot, ymin, ymax, ticks, ylabel):
    draw_y_axis(draw, plot, ymin, ymax, ticks, ylabel)


def draw_line_panel(draw, box, df, x_col, value_col, title, ylabel, coverage=False, target=None):
    left, top, right, bottom = box
    plot = (left + 72, top + 36, right - 22, bottom - 62)
    if coverage:
        ymin, ymax = 0.70, 1.00
        ticks = [0.70, 0.80, 0.90, 1.00]
    else:
        ymin, ymax = finite_limits(df[value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
        ticks = np.linspace(ymin, ymax, 4)

    draw.text((left + 5, top), title, fill=(25, 25, 25), font=FONT_BOLD)
    draw_line_axis(draw, plot, ymin, ymax, ticks, ylabel)
    if target is not None:
        yy = ypix(target, plot, ymin, ymax)
        x = plot[0]
        while x < plot[2]:
            draw.line([x, yy, min(x + 9, plot[2]), yy], fill=(70, 70, 70), width=2)
            x += 18

    xs = sorted(df[x_col].dropna().unique())
    xmin, xmax = float(min(xs)), float(max(xs))

    def sx(v):
        if np.isclose(xmin, xmax):
            return (plot[0] + plot[2]) / 2
        return plot[0] + (float(v) - xmin) / (xmax - xmin) * (plot[2] - plot[0])

    for xval in xs:
        xx = sx(xval)
        draw.line([xx, plot[3], xx, plot[3] + 5], fill=(35, 35, 35), width=1)
        label = f"{xval:g}"
        tw = draw.textlength(label, font=FONT_TINY)
        draw.text((xx - tw / 2, plot[3] + 11), label, fill=(35, 35, 35), font=FONT_TINY)

    xlabel = "n_cal" if x_col == "n_cal" else "N"
    tw = draw.textlength(xlabel, font=FONT_SMALL)
    draw.text(((plot[0] + plot[2] - tw) / 2, plot[3] + 36), xlabel, fill=(30, 30, 30), font=FONT_SMALL)

    for method in METHOD_ORDER:
        sub = df[df["Method"] == method].sort_values(x_col)
        if sub.empty:
            continue
        pts = []
        for _, row in sub.iterrows():
            y = float(row[value_col])
            if not np.isfinite(y):
                y = ymax
            pts.append((sx(row[x_col]), ypix(y, plot, ymin, ymax)))
        color = METHOD_COLORS[method]
        lw = 4 if method == "RSA-CP (OT) (Ours)" else 3
        if len(pts) > 1:
            draw.line(pts, fill=color, width=lw)
        for px, py in pts:
            r = 5 if method == "RSA-CP (OT) (Ours)" else 4
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=1)


def restyle_sensitivity(summary_name, out_base, x_col):
    summary = read_csv_any(summary_name, "summary")
    img = Image.new("RGB", (1640, 1040), "white")
    draw = ImageDraw.Draw(img)
    alphas = [0.05, 0.10]
    for row, alpha in enumerate(alphas):
        y0 = 35 + row * 490
        sub = summary[np.isclose(summary["alpha"], alpha)]
        draw_line_panel(
            draw,
            (55, y0, 800, y0 + 420),
            sub,
            x_col,
            "Coverage_mean",
            f"α = {alpha:g}",
            "Coverage",
            coverage=True,
            target=1.0 - alpha,
        )
        draw_line_panel(
            draw,
            (860, y0, 1605, y0 + 420),
            sub,
            x_col,
            "Length_mean",
            f"α = {alpha:g}",
            "Prediction set size",
        )
    draw_legend(draw, 365, 985)
    return save_both(img, FIGURE_DIR / out_base)


def draw_compact_box_panel(draw, box, df, value_col, ylabel, coverage=False, target=None, show_method_labels=False):
    left, top, right, bottom = box
    plot = (left + 56, top + 14, right - 12, bottom - (46 if show_method_labels else 18))
    if coverage:
        ymin, ymax = 0.70, 1.00
        ticks = [0.70, 0.80, 0.90, 1.00]
    else:
        ymin, ymax = finite_limits(df[value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
        ticks = np.linspace(ymin, ymax, 3)
    draw_y_axis(draw, plot, ymin, ymax, ticks, ylabel)
    if target is not None:
        yy = ypix(target, plot, ymin, ymax)
        x = plot[0]
        while x < plot[2]:
            draw.line([x, yy, min(x + 8, plot[2]), yy], fill=(70, 70, 70), width=2)
            x += 16

    gap = (plot[2] - plot[0]) / (len(METHOD_ORDER) + 0.45)
    positions = [plot[0] + gap * (i + 0.72) for i in range(len(METHOD_ORDER))]
    box_w = min(34, gap * 0.52)
    for x, method in zip(positions, METHOD_ORDER):
        vals = df.loc[df["Method"] == method, value_col].to_numpy(dtype=float)
        draw_box(draw, x, box_w, vals, plot, ymin, ymax, METHOD_COLORS[method])
    if show_method_labels:
        labels = [METHOD_SHORT[m].replace("Synthetic-only", "Synth") for m in METHOD_ORDER]
        draw_x_labels(draw, plot, labels, positions, x_label=False)


def restyle_figure9_two_per_row():
    raw = read_csv_any("figure9_meps_age_groups_raw.csv", "raw")
    ages = ["0-20", "20-40", "40-60", "60-100"]
    alphas = [0.05, 0.10]
    panels = [(a, al) for a in ages for al in alphas]
    img = Image.new("RGB", (1760, 2460), "white")
    draw = ImageDraw.Draw(img)
    cell_w, cell_h = 820, 535
    x_starts = [55, 895]
    y_start = 35
    for idx, (age, alpha) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x0 = x_starts[col]
        y0 = y_start + row * 580
        sub = raw[(raw["age_group"] == age) & np.isclose(raw["alpha"], alpha)]
        draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(55, 55, 55), width=2)
        draw.text((x0 + 18, y0 + 14), f"Age {age}, α={alpha:g}", fill=(25, 25, 25), font=FONT_BOLD)
        draw_compact_box_panel(
            draw,
            (x0 + 12, y0 + 55, x0 + cell_w - 12, y0 + 285),
            sub,
            "Coverage",
            "Coverage",
            coverage=True,
            target=1.0 - alpha,
            show_method_labels=False,
        )
        draw_compact_box_panel(
            draw,
            (x0 + 12, y0 + 305, x0 + cell_w - 12, y0 + cell_h - 18),
            sub,
            "Length",
            "Interval width",
            coverage=False,
            show_method_labels=True,
        )
    draw.text((60, 2375), "Coverage y-axis is fixed to [0.70, 1.00]; dashed lines are nominal coverage.", fill=(70, 70, 70), font=FONT_SMALL)
    draw_legend(draw, 520, 2415)
    return save_both(img, FIGURE_DIR / "figure9_meps_age_groups_boxplot_aesthetic_2col")


def main():
    p4 = restyle_figure4()
    p9 = restyle_figure9()
    p5 = restyle_sensitivity("figure5_imagenet_ncal_summary.csv", "figure5_imagenet_ncal_aesthetic", "n_cal")
    p8 = restyle_sensitivity("figure8_imagenet_nsyn_summary.csv", "figure8_imagenet_nsyn_aesthetic", "n_ref")
    p9_2col = restyle_figure9_two_per_row()
    print("Figure 4 aesthetic:", p4)
    print("Figure 9 aesthetic:", p9)
    print("Figure 5 aesthetic:", p5)
    print("Figure 8 aesthetic:", p8)
    print("Figure 9 two-column aesthetic:", p9_2col)


if __name__ == "__main__":
    main()
