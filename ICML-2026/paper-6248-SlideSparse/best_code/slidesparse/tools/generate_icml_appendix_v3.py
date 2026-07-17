#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
ICML Appendix PDF 生成器 v3

特性:
1. 表格连续紧密排列，自动跨页续接
2. 自动裁剪所有页面空白（内置 pdfcrop 功能）
3. 最后一页动态高度

Usage:
    python3 generate_icml_appendix_v3.py
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# 页面配置
# =============================================================================

PAGE_WIDTH_INCH = 8.5
PAGE_HEIGHT_INCH = 11.0

# 表格样式
HEADER_BG_COLOR_L1 = '#1E3A5F'
HEADER_BG_COLOR_L2 = '#2E4A6F'
HEADER_TEXT_COLOR = 'white'
ROW_EVEN_COLOR = '#F8F9FA'
ROW_ODD_COLOR = 'white'
SPEEDUP_GOOD_COLOR = '#D4EDDA'
SPEEDUP_GREAT_COLOR = '#B8E6C1'
GRID_COLOR = '#DEE2E6'

# 字体设置
HEADER_FONT_SIZE = 6
CELL_FONT_SIZE = 5.5
TITLE_FONT_SIZE = 8

# 尺寸设置 (inch)
ROW_HEIGHT = 0.13
HEADER_HEIGHT = 0.28
TABLE_GAP = 0.2
TITLE_HEIGHT = 0.18

# 目录
SCRIPT_DIR = Path(__file__).parent
INPUT_BASE_DIR = SCRIPT_DIR / "all_results_table"
OUTPUT_DIR = SCRIPT_DIR / "all_results_table" / "icml_appendix_v3"


# =============================================================================
# PDF 裁剪功能 (纯 Python 实现)
# =============================================================================

def crop_pdf_with_pymupdf(input_path: Path, output_path: Path, margin: float = 0):
    """
    使用 PyMuPDF 裁剪 PDF 的所有页面空白
    通过渲染为图片来精确检测内容边界
    margin: 裁剪后保留的边距 (points), 0 = 像素级贴边
    """
    try:
        import fitz  # PyMuPDF
        import numpy as np
    except ImportError:
        print("    [警告] PyMuPDF 未安装，跳过裁剪")
        import shutil
        shutil.copy(input_path, output_path)
        return False
    
    doc = fitz.open(input_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 渲染页面为高分辨率图片
        zoom = 4  # 更高精度
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # 转换为 numpy 数组
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        
        # 检测非白色像素（白色 = 255）
        threshold = 254  # 更严格的阈值
        non_white = np.any(img < threshold, axis=2)
        
        # 找到内容边界
        rows_with_content = np.any(non_white, axis=1)
        cols_with_content = np.any(non_white, axis=0)
        
        if not np.any(rows_with_content) or not np.any(cols_with_content):
            continue
        
        row_indices = np.where(rows_with_content)[0]
        col_indices = np.where(cols_with_content)[0]
        
        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]
        
        # 转换回页面坐标 (考虑 zoom)
        min_x = min_col / zoom
        min_y = min_row / zoom
        max_x = (max_col + 1) / zoom  # +1 确保包含最后一个像素
        max_y = (max_row + 1) / zoom
        
        # 设置裁剪框（无边距）
        crop_rect = fitz.Rect(min_x, min_y, max_x, max_y)
        page.set_cropbox(crop_rect)
    
    doc.save(output_path)
    doc.close()
    return True


def crop_pdf_with_ghostscript(input_path: Path, output_path: Path):
    """
    备选方案：使用 Ghostscript 裁剪
    """
    try:
        # 先检测边界框
        result = subprocess.run(
            ['gs', '-q', '-dBATCH', '-dNOPAUSE', '-sDEVICE=bbox', str(input_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False
        
        # 解析 bbox 输出并裁剪... (复杂，暂不实现)
        return False
    except FileNotFoundError:
        return False


# =============================================================================
# 工具函数
# =============================================================================

def read_csv(csv_path: Path) -> Tuple[List[str], List[List[str]]]:
    if not csv_path.exists():
        return [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def detect_table_type(filename: str) -> str:
    filename_lower = filename.lower()
    if 'appendix_a' in filename_lower or 'square' in filename_lower:
        return 'kernel_square'
    elif 'appendix_b' in filename_lower or 'model_kernel' in filename_lower:
        return 'kernel_model'
    elif 'appendix_c' in filename_lower or 'prefill' in filename_lower:
        return 'e2e_prefill'
    elif 'appendix_d' in filename_lower or 'decode' in filename_lower:
        return 'e2e_decode'
    return 'kernel_model'


def get_id_columns(table_type: str) -> List[str]:
    if table_type == 'kernel_square':
        return ['GPU']
    return ['GPU', 'Model']


def is_speedup_column(col_name: str) -> bool:
    return col_name.startswith('2:')


def is_efficiency_column(col_name: str) -> bool:
    return col_name.startswith('Eff_')


def convert_sparsity_header(name: str) -> str:
    """
    转换稀疏度表头: 2:L → (L-2):L
    例如: 2:4 → 2:4, 2:6 → 4:6, 2:8 → 6:8, 2:∞ → ∞:∞
    """
    if not name.startswith('2:'):
        return name
    
    suffix = name[2:]  # 去掉 "2:"
    
    if suffix == '∞' or suffix == 'inf':
        return '∞:∞'
    
    try:
        L = int(suffix)
        new_Z = L - 2  # (L-2):L
        return f"{new_Z}:{L}"
    except ValueError:
        return name


def get_cell_color(value: str, col_name: str, row_idx: int, baseline_type: str) -> str:
    base_color = ROW_EVEN_COLOR if row_idx % 2 == 0 else ROW_ODD_COLOR
    
    if baseline_type == 'cublas' and is_speedup_column(col_name) and value.strip():
        try:
            val = float(value)
            if val >= 1.5:
                return SPEEDUP_GREAT_COLOR
            elif val >= 1.0:
                return SPEEDUP_GOOD_COLOR
        except:
            pass
    elif baseline_type == 'cusparse' and is_efficiency_column(col_name) and value.strip():
        try:
            val = float(value.rstrip('%'))
            if val >= 105:
                return SPEEDUP_GREAT_COLOR
            elif val >= 100:
                return SPEEDUP_GOOD_COLOR
        except:
            pass
    return base_color


class DisplayState:
    def __init__(self):
        self.prev_ids = {}
    
    def reset(self):
        self.prev_ids = {}
    
    def build_display_rows(self, rows: List[List[str]], header: List[str], 
                           id_columns: List[str]) -> List[List[str]]:
        id_indices = [header.index(col) for col in id_columns if col in header]
        display_rows = []
        for row in rows:
            new_row = list(row)
            for col_idx in id_indices:
                if col_idx < len(row):
                    current_val = row[col_idx]
                    prev_key = f"col_{col_idx}"
                    if prev_key in self.prev_ids and self.prev_ids[prev_key] == current_val:
                        new_row[col_idx] = ''
                    else:
                        self.prev_ids[prev_key] = current_val
            display_rows.append(new_row)
        return display_rows


def calculate_col_widths(header: List[str], table_width: float, baseline_type: str) -> List[float]:
    """
    计算列宽度，优先给 Speedup/Efficiency 列更多空间
    
    典型表头: GPU, Model, M, Latency, 2:4, 2:6, 2:8, 2:10, 2:12, 2:14, 2:16, 2:∞
    总宽度固定为 table_width，按权重比例分配
    """
    weights = []
    for col in header:
        if col == 'GPU':
            weights.append(0.9)   # 减小
        elif col == 'Model':
            weights.append(1.1)   # 减小
        elif col == 'M':
            weights.append(0.7)   # 减小
        elif 'Latency' in col or 'Throughput' in col:
            weights.append(1.2)   # 恢复到 1.2
        elif col.startswith('2:'):
            weights.append(1.0)   # 增大 (重要数据列)
        elif col.startswith('Eff_'):
            weights.append(1.0)   # 增大 (重要数据列)
        else:
            weights.append(1.0)
    total_weight = sum(weights)
    return [w / total_weight * table_width for w in weights]


# =============================================================================
# 绘制函数
# =============================================================================

def draw_header(ax, header: List[str], col_widths: List[float],
                start_x: float, start_y: float, table_type: str, baseline_type: str):
    """绘制双行表头"""
    row_height = HEADER_HEIGHT / 2
    
    id_cols = []
    m_col = None
    baseline_col = None
    data_cols = []
    
    for i, col in enumerate(header):
        if col in ['GPU', 'Model']:
            id_cols.append((i, col))
        elif col == 'M':
            m_col = (i, col)
        elif 'Latency' in col or 'Throughput' in col:
            baseline_col = (i, col)
        elif col.startswith('2:') or col.startswith('Eff_'):
            data_cols.append((i, col))
    
    y1 = start_y
    
    # GPU/Model 列
    for col_idx, col_name in id_cols:
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        rect = mpatches.FancyBboxPatch(
            (x, y1 - HEADER_HEIGHT), w, HEADER_HEIGHT,
            boxstyle="square,pad=0", facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR, linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y1 - HEADER_HEIGHT/2, col_name,
                ha='center', va='center', fontsize=HEADER_FONT_SIZE,
                fontweight='bold', color=HEADER_TEXT_COLOR)
    
    # M 列
    if m_col:
        col_idx, _ = m_col
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        rect = mpatches.FancyBboxPatch(
            (x, y1 - HEADER_HEIGHT), w, HEADER_HEIGHT,
            boxstyle="square,pad=0", facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR, linewidth=0.5)
        ax.add_patch(rect)
        
        if table_type == 'e2e_prefill':
            line1 = 'Batchsize'
        elif table_type == 'e2e_decode':
            line1 = 'Concurrency'
        else:
            line1 = ''
        
        if line1:
            ax.text(x + w/2, y1 - row_height/2, line1,
                    ha='center', va='center', fontsize=HEADER_FONT_SIZE - 0.5,
                    fontweight='bold', color=HEADER_TEXT_COLOR)
            ax.text(x + w/2, y1 - HEADER_HEIGHT + row_height/2, 'M',
                    ha='center', va='center', fontsize=HEADER_FONT_SIZE,
                    fontweight='bold', color=HEADER_TEXT_COLOR)
        else:
            ax.text(x + w/2, y1 - HEADER_HEIGHT/2, 'M',
                    ha='center', va='center', fontsize=HEADER_FONT_SIZE,
                    fontweight='bold', color=HEADER_TEXT_COLOR)
    
    # Baseline 列
    if baseline_col:
        col_idx, col_name = baseline_col
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        rect = mpatches.FancyBboxPatch(
            (x, y1 - HEADER_HEIGHT), w, HEADER_HEIGHT,
            boxstyle="square,pad=0", facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR, linewidth=0.5)
        ax.add_patch(rect)
        
        line1 = 'cuSPARSELt' if baseline_type == 'cusparse' else 'cuBLASLt'
        ax.text(x + w/2, y1 - row_height/2, line1,
                ha='center', va='center', fontsize=HEADER_FONT_SIZE,
                fontweight='bold', color=HEADER_TEXT_COLOR)
        
        line2 = 'Latency(μs)' if 'Latency' in col_name else 'Throughput'
        ax.text(x + w/2, y1 - HEADER_HEIGHT + row_height/2, line2,
                ha='center', va='center', fontsize=HEADER_FONT_SIZE - 0.5,
                fontweight='bold', color=HEADER_TEXT_COLOR)
    
    # 数据列
    if data_cols:
        first_idx = data_cols[0][0]
        last_idx = data_cols[-1][0]
        x_start = start_x + sum(col_widths[:first_idx])
        total_w = sum(col_widths[first_idx:last_idx+1])
        
        rect = mpatches.FancyBboxPatch(
            (x_start, y1 - row_height), total_w, row_height,
            boxstyle="square,pad=0", facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR, linewidth=0.5)
        ax.add_patch(rect)
        
        if baseline_type == 'cusparse':
            merged_title = 'Algorithmic Efficiency under Different Sparsity'
        else:
            merged_title = 'Speedup Ratio under Different Sparsity'
        ax.text(x_start + total_w/2, y1 - row_height/2, merged_title,
                ha='center', va='center', fontsize=HEADER_FONT_SIZE,
                fontweight='bold', color=HEADER_TEXT_COLOR)
        
        for col_idx, col_name in data_cols:
            x = start_x + sum(col_widths[:col_idx])
            w = col_widths[col_idx]
            rect = mpatches.FancyBboxPatch(
                (x, y1 - HEADER_HEIGHT), w, row_height,
                boxstyle="square,pad=0", facecolor=HEADER_BG_COLOR_L2,
                edgecolor=GRID_COLOR, linewidth=0.5)
            ax.add_patch(rect)
            
            # 转换表头: 2:L → (L-2):L
            display_name = col_name[4:] if col_name.startswith('Eff_') else col_name
            display_name = convert_sparsity_header(display_name)
            ax.text(x + w/2, y1 - HEADER_HEIGHT + row_height/2, display_name,
                    ha='center', va='center', fontsize=HEADER_FONT_SIZE,
                    fontweight='bold', color=HEADER_TEXT_COLOR)


# =============================================================================
# 主生成器
# =============================================================================

class ContinuousPDFGenerator:
    """连续表格 PDF 生成器"""
    
    def __init__(self, baseline_type: str, table_width: float = 7.0):
        self.baseline_type = baseline_type
        self.bt = 'cublas' if 'cublas' in baseline_type.lower() else 'cusparse'
        self.table_width = table_width
        self.tables = []
    
    def add_table(self, title: str, header: List[str], rows: List[List[str]], table_type: str):
        self.tables.append((title, header, rows, table_type))
    
    def generate(self, output_path: Path, crop: bool = True):
        """生成 PDF，可选裁剪"""
        
        if not self.tables:
            print("    无表格数据")
            return
        
        # 临时文件（如果需要裁剪）
        if crop:
            temp_path = output_path.with_suffix('.temp.pdf')
            target_path = temp_path
        else:
            target_path = output_path
        
        # 预计算所有页面内容
        pages_content = []
        current_page = []
        current_y = PAGE_HEIGHT_INCH - 0.1
        start_x = (PAGE_WIDTH_INCH - self.table_width) / 2
        
        for title, header, rows, table_type in self.tables:
            id_columns = get_id_columns(table_type)
            col_widths = calculate_col_widths(header, self.table_width, self.bt)
            
            display_state = DisplayState()
            display_rows = display_state.build_display_rows(rows, header, id_columns)
            
            # 检查标题+表头能否放下
            if current_y - TITLE_HEIGHT - HEADER_HEIGHT < 0.3:
                pages_content.append(current_page)
                current_page = []
                current_y = PAGE_HEIGHT_INCH - 0.1
            
            # 添加标题
            current_page.append(('title', title, start_x, self.table_width, current_y))
            current_y -= TITLE_HEIGHT
            
            # 添加表头
            current_page.append(('header', header, col_widths, start_x, current_y, table_type, self.bt))
            current_y -= HEADER_HEIGHT
            
            # 逐行添加
            for row_idx, row in enumerate(display_rows):
                if current_y - ROW_HEIGHT < 0.2:
                    pages_content.append(current_page)
                    current_page = []
                    current_y = PAGE_HEIGHT_INCH - 0.1
                    
                    # 续表
                    current_page.append(('title_cont', f"{title} (cont.)", start_x, self.table_width, current_y))
                    current_y -= TITLE_HEIGHT
                    current_page.append(('header', header, col_widths, start_x, current_y, table_type, self.bt))
                    current_y -= HEADER_HEIGHT
                
                current_page.append(('row', row, header, col_widths, start_x, current_y, row_idx, self.bt))
                current_y -= ROW_HEIGHT
            
            # 表格间距
            current_y -= TABLE_GAP
        
        if current_page:
            pages_content.append(current_page)
        
        # 绘制 PDF
        with PdfPages(target_path) as pdf:
            for page_idx, drawings in enumerate(pages_content):
                fig, ax = plt.subplots(figsize=(PAGE_WIDTH_INCH, PAGE_HEIGHT_INCH))
                ax.set_xlim(0, PAGE_WIDTH_INCH)
                ax.set_ylim(0, PAGE_HEIGHT_INCH)
                ax.axis('off')
                
                for d in drawings:
                    if d[0] == 'title':
                        _, text, sx, tw, y = d
                        ax.text(sx + tw/2, y - TITLE_HEIGHT/2, text,
                                ha='center', va='center', fontsize=TITLE_FONT_SIZE, fontweight='bold')
                    
                    elif d[0] == 'title_cont':
                        _, text, sx, tw, y = d
                        ax.text(sx + tw/2, y - TITLE_HEIGHT/2, text,
                                ha='center', va='center', fontsize=TITLE_FONT_SIZE,
                                fontweight='bold', style='italic')
                    
                    elif d[0] == 'header':
                        _, header, col_widths, sx, y, table_type, bt = d
                        draw_header(ax, header, col_widths, sx, y, table_type, bt)
                    
                    elif d[0] == 'row':
                        _, row, header, col_widths, sx, y, ridx, bt = d
                        x = sx
                        for col_idx in range(len(header)):
                            value = row[col_idx] if col_idx < len(row) else ''
                            col_name = header[col_idx] if col_idx < len(header) else ''
                            bg_color = get_cell_color(value, col_name, ridx, bt)
                            
                            rect = mpatches.FancyBboxPatch(
                                (x, y - ROW_HEIGHT), col_widths[col_idx], ROW_HEIGHT,
                                boxstyle="square,pad=0", facecolor=bg_color,
                                edgecolor=GRID_COLOR, linewidth=0.3)
                            ax.add_patch(rect)
                            ax.text(x + col_widths[col_idx]/2, y - ROW_HEIGHT/2,
                                    value, ha='center', va='center', fontsize=CELL_FONT_SIZE)
                            x += col_widths[col_idx]
                
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
        
        print(f"    生成 {len(pages_content)} 页")
        
        # 裁剪
        if crop:
            print(f"    裁剪空白...")
            success = crop_pdf_with_pymupdf(temp_path, output_path, margin=5)
            if success:
                temp_path.unlink()  # 删除临时文件
                print(f"    ✓ 裁剪完成: {output_path.name}")
            else:
                # 裁剪失败，使用原文件
                temp_path.rename(output_path)
                print(f"    ✓ 输出 (未裁剪): {output_path.name}")
        else:
            print(f"    ✓ 输出: {output_path.name}")


# =============================================================================
# 生成函数
# =============================================================================

def generate_appendix(csv_folder: Path, output_path: Path, baseline_type: str,
                      appendix_type: str, precision_order: List[str]):
    """通用 Appendix 生成函数"""
    
    # 根据 appendix 类型确定文件名前缀和标题
    if appendix_type == 'a':
        file_prefix = 'appendix_a_square'
        title_prefix = 'Square Kernel'
    elif appendix_type == 'b':
        file_prefix = 'appendix_b_model_kernel'
        title_prefix = 'Model Kernel'
    elif appendix_type == 'c':
        file_prefix = 'appendix_c_prefill'
        title_prefix = 'Prefill'
    elif appendix_type == 'd':
        file_prefix = 'appendix_d_decode'
        title_prefix = 'Decode'
    else:
        raise ValueError(f"Unknown appendix type: {appendix_type}")
    
    gen = ContinuousPDFGenerator(baseline_type)
    
    for precision in precision_order:
        csv_file = f"{file_prefix}_{precision}.csv"
        csv_path = csv_folder / csv_file
        header, rows = read_csv(csv_path)
        
        if not rows:
            print(f"    跳过: {csv_file} (无数据)")
            continue
        
        table_type = detect_table_type(csv_file)
        gen.add_table(f"{title_prefix} ({precision})", header, rows, table_type)
        print(f"    添加: {csv_file} ({len(rows)} 行)")
    
    if gen.tables:
        gen.generate(output_path, crop=True)
    else:
        print(f"    [警告] 无数据，跳过生成")


def main():
    print("=" * 60)
    print("ICML Appendix PDF 生成器 v3 (自动裁剪版)")
    print("=" * 60)
    
    # 检查 PyMuPDF
    try:
        import fitz
        print(f"[OK] PyMuPDF 版本: {fitz.version[0]}")
    except ImportError:
        print("[警告] PyMuPDF 未安装，将跳过裁剪")
        print("       安装命令: pip install pymupdf")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 精度顺序
    square_precision = ['FP4', 'INT8', 'FP8', 'FP16', 'BF16']  # appendix-a
    model_precision = ['INT8', 'FP8']  # appendix-b
    e2e_precision = ['INT8', 'FP8']    # appendix-c, d
    
    # 定义所有8个组
    groups = [
        # cuBLASLt baseline
        ('cuBLASLt', 'a', square_precision, 'appendix_a_cuBLASLt.pdf'),
        ('cuBLASLt', 'b', model_precision, 'appendix_b_cuBLASLt.pdf'),
        ('cuBLASLt', 'c', e2e_precision, 'appendix_c_cuBLASLt.pdf'),
        ('cuBLASLt', 'd', e2e_precision, 'appendix_d_cuBLASLt.pdf'),
        # cuSPARSELt baseline
        ('cuSPARSELt', 'a', square_precision, 'appendix_a_cuSPARSELt.pdf'),
        ('cuSPARSELt', 'b', model_precision, 'appendix_b_cuSPARSELt.pdf'),
        ('cuSPARSELt', 'c', e2e_precision, 'appendix_c_cuSPARSELt.pdf'),
        ('cuSPARSELt', 'd', e2e_precision, 'appendix_d_cuSPARSELt.pdf'),
    ]
    
    total = len(groups)
    for idx, (baseline, appendix_type, precisions, filename) in enumerate(groups, 1):
        print(f"\n[{idx}/{total}] Appendix {appendix_type.upper()} - {baseline} Baseline")
        
        csv_folder = INPUT_BASE_DIR / f"{baseline}_baseline_csv"
        output_path = OUTPUT_DIR / filename
        
        generate_appendix(csv_folder, output_path, baseline, appendix_type, precisions)
    
    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n生成的文件:")
    
    # 显示页面尺寸
    try:
        import fitz
        for pdf_file in sorted(OUTPUT_DIR.glob("*.pdf")):
            doc = fitz.open(pdf_file)
            pages_info = []
            for i, page in enumerate(doc):
                r = page.rect
                pages_info.append(f"{r.width/72:.1f}x{r.height/72:.1f}")
            doc.close()
            print(f"  {pdf_file.name}: {len(pages_info)} 页")
    except:
        for pdf_file in sorted(OUTPUT_DIR.glob("*.pdf")):
            print(f"  {pdf_file.name}")


if __name__ == "__main__":
    main()
