#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CSV 转 PDF 表格工具 (双基线版)

将 all_results_table 目录下的 CSV 文件转换为 PDF 表格。
支持两种基线:
1. cuBLASLt baseline: 展示相对于 Dense 的加速比
2. cuSPARSELt baseline: 展示相对于 2:4 的算法效率

Usage:
    python3 csv_to_pdf_table.py
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import csv
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# =============================================================================
# 配置
# =============================================================================

PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
MARGIN = 0.2

# 表格样式
HEADER_BG_COLOR_L1 = '#1E3A5F'
HEADER_BG_COLOR_L2 = '#2E4A6F'
HEADER_TEXT_COLOR = 'white'
ROW_EVEN_COLOR = '#F8F9FA'
ROW_ODD_COLOR = 'white'
SPEEDUP_GOOD_COLOR = '#D4EDDA'   # 浅绿色
SPEEDUP_GREAT_COLOR = '#B8E6C1'  # 深绿色
GRID_COLOR = '#DEE2E6'

# 字体设置
HEADER_FONT_SIZE = 5.5
CELL_FONT_SIZE = 5
TITLE_FONT_SIZE = 8

# 行高设置
ROW_HEIGHT = 0.011
HEADER_HEIGHT = 0.028

# 固定表格宽度
FIXED_TABLE_WIDTH = 0.72

# 每页最大行数
MAX_ROWS_PER_PAGE = 65

# 目录
SCRIPT_DIR = Path(__file__).parent
INPUT_BASE_DIR = SCRIPT_DIR / "all_results_table"
OUTPUT_BASE_DIR = SCRIPT_DIR / "all_results_table"


# =============================================================================
# 表格类型检测
# =============================================================================

def detect_table_type(header: List[str], filename: str) -> str:
    """检测表格类型"""
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


def detect_baseline_type(folder_name: str) -> str:
    """检测基线类型: 'cublas' 或 'cusparse'"""
    if 'cusparse' in folder_name.lower():
        return 'cusparse'
    return 'cublas'


def get_id_columns(table_type: str) -> List[str]:
    """获取标识列"""
    if table_type == 'kernel_square':
        return ['GPU']
    else:
        return ['GPU', 'Model']


# =============================================================================
# 工具函数
# =============================================================================

def read_csv(csv_path: Path) -> Tuple[List[str], List[List[str]]]:
    """读取 CSV 文件"""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def is_speedup_column(col_name: str) -> bool:
    """判断是否是加速比列"""
    return col_name.startswith('2:')


def is_efficiency_column(col_name: str) -> bool:
    """判断是否是效率列"""
    return col_name.startswith('Eff_')


def get_cell_color(value: str, col_name: str, row_idx: int, baseline_type: str) -> str:
    """获取单元格背景色"""
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
        # 效率 >= 100% 为绿色
        try:
            # 解析百分比 (如 "98.5%")
            val = float(value.rstrip('%'))
            if val >= 105:
                return SPEEDUP_GREAT_COLOR
            elif val >= 100:
                return SPEEDUP_GOOD_COLOR
        except:
            pass
    
    return base_color


def build_display_rows(rows: List[List[str]], header: List[str], 
                       id_columns: List[str], start_global_idx: int) -> List[List[str]]:
    """构建显示行：省略重复值"""
    if not hasattr(build_display_rows, 'prev_ids'):
        build_display_rows.prev_ids = {}
    
    id_indices = []
    for col_name in id_columns:
        if col_name in header:
            id_indices.append(header.index(col_name))
    
    display_rows = []
    for local_idx, row in enumerate(rows):
        new_row = list(row)
        
        for col_idx in id_indices:
            if col_idx < len(row):
                current_val = row[col_idx]
                prev_key = f"col_{col_idx}"
                
                if prev_key in build_display_rows.prev_ids and \
                   build_display_rows.prev_ids[prev_key] == current_val:
                    new_row[col_idx] = ''
                else:
                    build_display_rows.prev_ids[prev_key] = current_val
        
        display_rows.append(new_row)
    
    return display_rows


def reset_display_state():
    """重置显示状态"""
    if hasattr(build_display_rows, 'prev_ids'):
        build_display_rows.prev_ids = {}


def calculate_col_widths(header: List[str], table_width: float, 
                         table_type: str, baseline_type: str) -> List[float]:
    """计算列宽度"""
    weights = []
    for col in header:
        if col == 'GPU':
            weights.append(1.4)
        elif col == 'Model':
            weights.append(2.0)
        elif col == 'M':
            weights.append(0.9)
        elif 'Latency' in col or 'Throughput' in col:
            weights.append(1.8)
        elif col.startswith('2:'):
            weights.append(0.9)
        elif col.startswith('Eff_'):
            weights.append(1.0)
        else:
            weights.append(1.0)
    
    total_weight = sum(weights)
    widths = [w / total_weight * table_width for w in weights]
    
    return widths


# =============================================================================
# 绘制函数
# =============================================================================

def draw_two_row_header(ax, header: List[str], col_widths: List[float],
                        start_x: float, start_y: float, 
                        header_height: float, table_type: str, baseline_type: str):
    """绘制双行表头"""
    row_height = header_height / 2
    
    # 分类列
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
    
    # === GPU/Model 列 (跨两行) ===
    for col_idx, col_name in id_cols:
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        
        rect = mpatches.FancyBboxPatch(
            (x, y1 - header_height), w, header_height,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR,
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        ax.text(x + w/2, y1 - header_height/2, col_name,
                ha='center', va='center',
                fontsize=HEADER_FONT_SIZE, fontweight='bold',
                color=HEADER_TEXT_COLOR)
    
    # === M 列 ===
    if m_col:
        col_idx, col_name = m_col
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        
        rect = mpatches.FancyBboxPatch(
            (x, y1 - header_height), w, header_height,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR,
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        if table_type == 'e2e_prefill':
            line1_text = 'Batch Size'
        elif table_type == 'e2e_decode':
            line1_text = 'Concurrency'
        else:
            line1_text = ''
        
        if line1_text:
            ax.text(x + w/2, y1 - row_height/2, line1_text,
                    ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE - 0.5, fontweight='bold',
                    color=HEADER_TEXT_COLOR)
            ax.text(x + w/2, y1 - header_height + row_height/2, 'M',
                    ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE, fontweight='bold',
                    color=HEADER_TEXT_COLOR)
        else:
            ax.text(x + w/2, y1 - header_height/2, 'M',
                    ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE, fontweight='bold',
                    color=HEADER_TEXT_COLOR)
    
    # === Baseline 列 (双行) ===
    if baseline_col:
        col_idx, col_name = baseline_col
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        
        rect = mpatches.FancyBboxPatch(
            (x, y1 - header_height), w, header_height,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR,
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        # 根据基线类型设置第一行文字
        if baseline_type == 'cusparse':
            line1 = 'cuSPARSELt 2:4'
        else:
            line1 = 'cuBLASLt'
        
        ax.text(x + w/2, y1 - row_height/2, line1,
                ha='center', va='center',
                fontsize=HEADER_FONT_SIZE, fontweight='bold',
                color=HEADER_TEXT_COLOR)
        
        # 第二行
        if 'Latency' in col_name:
            line2 = 'Latency (μs)'
        else:
            line2 = 'Throughput (token/s)'
        
        ax.text(x + w/2, y1 - header_height + row_height/2, line2,
                ha='center', va='center',
                fontsize=HEADER_FONT_SIZE - 0.5, fontweight='bold',
                color=HEADER_TEXT_COLOR)
    
    # === 数据列 (第一行合并，第二行分开) ===
    if data_cols:
        first_idx = data_cols[0][0]
        last_idx = data_cols[-1][0]
        x_start = start_x + sum(col_widths[:first_idx])
        total_w = sum(col_widths[first_idx:last_idx+1])
        
        # 第一行: 合并标题
        rect = mpatches.FancyBboxPatch(
            (x_start, y1 - row_height), total_w, row_height,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR,
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        if baseline_type == 'cusparse':
            merged_title = 'Algorithmic Efficiency (vs Theory)'
        else:
            merged_title = 'cuSPARSELt Speedup Ratio'
        
        ax.text(x_start + total_w/2, y1 - row_height/2, merged_title,
                ha='center', va='center',
                fontsize=HEADER_FONT_SIZE, fontweight='bold',
                color=HEADER_TEXT_COLOR)
        
        # 第二行: 各列名
        for col_idx, col_name in data_cols:
            x = start_x + sum(col_widths[:col_idx])
            w = col_widths[col_idx]
            
            rect = mpatches.FancyBboxPatch(
                (x, y1 - header_height), w, row_height,
                boxstyle="square,pad=0",
                facecolor=HEADER_BG_COLOR_L2,
                edgecolor=GRID_COLOR,
                linewidth=0.5
            )
            ax.add_patch(rect)
            
            # 简化列名: "Eff_2:6" -> "2:6", "2:4" -> "2:4"
            display_name = col_name
            if col_name.startswith('Eff_'):
                display_name = col_name[4:]  # 去掉 "Eff_"
            
            ax.text(x + w/2, y1 - header_height + row_height/2, display_name,
                    ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE, fontweight='bold',
                    color=HEADER_TEXT_COLOR)


def draw_table_page(ax, header: List[str], rows: List[List[str]], 
                    title: str, page_num: int, total_pages: int,
                    col_widths: List[float], table_type: str, baseline_type: str):
    """绘制表格页面"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    table_width = sum(col_widths)
    start_x = (1 - table_width) / 2
    start_y = 0.97
    
    # 标题
    title_text = title
    if total_pages > 1:
        title_text += f" (Page {page_num}/{total_pages})"
    ax.text(0.5, 0.99, title_text, ha='center', va='top',
            fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    # 表头
    draw_two_row_header(ax, header, col_widths, start_x, start_y, 
                        HEADER_HEIGHT, table_type, baseline_type)
    
    # 数据行
    y = start_y - HEADER_HEIGHT
    for row_idx, row in enumerate(rows):
        x = start_x
        for col_idx in range(len(header)):
            value = row[col_idx] if col_idx < len(row) else ''
            col_name = header[col_idx] if col_idx < len(header) else ''
            
            bg_color = get_cell_color(value, col_name, row_idx, baseline_type)
            
            rect = mpatches.FancyBboxPatch(
                (x, y - ROW_HEIGHT), col_widths[col_idx], ROW_HEIGHT,
                boxstyle="square,pad=0",
                facecolor=bg_color,
                edgecolor=GRID_COLOR,
                linewidth=0.3
            )
            ax.add_patch(rect)
            
            ax.text(x + col_widths[col_idx]/2, y - ROW_HEIGHT/2,
                    value, ha='center', va='center',
                    fontsize=CELL_FONT_SIZE)
            x += col_widths[col_idx]
        
        y -= ROW_HEIGHT
    
    # 图例
    if page_num == 1:
        legend_y = y - 0.012
        if baseline_type == 'cusparse':
            ax.add_patch(mpatches.Rectangle((start_x, legend_y - 0.006), 0.01, 0.006,
                                            facecolor=SPEEDUP_GREAT_COLOR, edgecolor='gray', linewidth=0.3))
            ax.text(start_x + 0.012, legend_y - 0.003, '≥105%', fontsize=4, va='center')
            
            ax.add_patch(mpatches.Rectangle((start_x + 0.045, legend_y - 0.006), 0.01, 0.006,
                                            facecolor=SPEEDUP_GOOD_COLOR, edgecolor='gray', linewidth=0.3))
            ax.text(start_x + 0.057, legend_y - 0.003, '≥100%', fontsize=4, va='center')
        else:
            ax.add_patch(mpatches.Rectangle((start_x, legend_y - 0.006), 0.01, 0.006,
                                            facecolor=SPEEDUP_GREAT_COLOR, edgecolor='gray', linewidth=0.3))
            ax.text(start_x + 0.012, legend_y - 0.003, '≥1.5×', fontsize=4, va='center')
            
            ax.add_patch(mpatches.Rectangle((start_x + 0.045, legend_y - 0.006), 0.01, 0.006,
                                            facecolor=SPEEDUP_GOOD_COLOR, edgecolor='gray', linewidth=0.3))
            ax.text(start_x + 0.057, legend_y - 0.003, '≥1.0×', fontsize=4, va='center')


def csv_to_pdf(csv_path: Path, output_path: Path, baseline_type: str):
    """将 CSV 转换为 PDF"""
    header, rows = read_csv(csv_path)
    
    if not rows:
        print(f"    跳过空文件: {csv_path.name}")
        return
    
    table_type = detect_table_type(header, csv_path.name)
    id_columns = get_id_columns(table_type)
    
    reset_display_state()
    
    total_rows = len(rows)
    total_pages = (total_rows + MAX_ROWS_PER_PAGE - 1) // MAX_ROWS_PER_PAGE
    
    col_widths = calculate_col_widths(header, FIXED_TABLE_WIDTH, table_type, baseline_type)
    
    title = csv_path.stem.replace('_', ' ').title()
    
    with PdfPages(output_path) as pdf:
        for page in range(total_pages):
            start_idx = page * MAX_ROWS_PER_PAGE
            end_idx = min(start_idx + MAX_ROWS_PER_PAGE, total_rows)
            page_rows = rows[start_idx:end_idx]
            
            display_rows = build_display_rows(page_rows, header, id_columns, start_idx)
            
            fig, ax = plt.subplots(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
            fig.subplots_adjust(left=MARGIN/PAGE_WIDTH, right=1-MARGIN/PAGE_WIDTH,
                              top=1-MARGIN/PAGE_HEIGHT, bottom=MARGIN/PAGE_HEIGHT)
            
            draw_table_page(ax, header, display_rows, title, 
                          page + 1, total_pages, col_widths, table_type, baseline_type)
            
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    
    print(f"    ✓ {csv_path.name} -> {output_path.name} ({total_pages} 页, {total_rows} 行)")


def process_folder(csv_folder: Path, pdf_folder: Path, baseline_type: str):
    """处理一个文件夹"""
    if not csv_folder.exists():
        print(f"  文件夹不存在: {csv_folder}")
        return
    
    pdf_folder.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(csv_folder.glob("*.csv"))
    if not csv_files:
        print(f"  无 CSV 文件: {csv_folder}")
        return
    
    print(f"  处理 {len(csv_files)} 个文件...")
    for csv_path in csv_files:
        output_path = pdf_folder / (csv_path.stem + ".pdf")
        csv_to_pdf(csv_path, output_path, baseline_type)


def main():
    print("=" * 60)
    print("CSV 转 PDF 表格工具 (双基线版)")
    print("=" * 60)
    
    # 处理 cuBLASLt baseline
    print("\n[cuBLASLt Baseline]")
    csv_folder = INPUT_BASE_DIR / "cuBLASLt_baseline_csv"
    pdf_folder = OUTPUT_BASE_DIR / "cuBLASLt_baseline_pdf"
    process_folder(csv_folder, pdf_folder, 'cublas')
    
    # 处理 cuSPARSELt baseline
    print("\n[cuSPARSELt Baseline]")
    csv_folder = INPUT_BASE_DIR / "cuSPARSELt_baseline_csv"
    pdf_folder = OUTPUT_BASE_DIR / "cuSPARSELt_baseline_pdf"
    process_folder(csv_folder, pdf_folder, 'cusparse')
    
    print()
    print("=" * 60)
    print("转换完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
