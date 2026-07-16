"""
Create neighbor comparison figures for paper appendix
- Organized by dataset
- Each dataset: 3 models + metaclip = 4 rows
- Each row: query image + 9 neighbors
- Width: 6.75 in (approx. 17.1 cm)
- Annotated with similarity scores
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse

# Configuration
FIGURE_WIDTH_INCH = 6.75
DPI = 300  # Paper quality
FIGURE_WIDTH_PX = int(FIGURE_WIDTH_INCH * DPI)  # 2025 px

NUM_IMAGES_PER_ROW = 10  # 1 query + 9 neighbors
IMAGE_PADDING = 5  # Image padding (px)
LABEL_HEIGHT = 45  # Label height (px)
TITLE_HEIGHT = 55  # Title height (px)

# Model configuration: short name -> full name
MODEL_FULL_NAMES = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl": "OpenGVLab/InternVL3_5-8B",
    "sailvl": "BytedanceDouyinContent/SAIL-VL2-8B",
    "metaclip": "facebook/metaclip-2-worldwide-huge-quickgelu",
}

MODELS = ["qwen", "internvl", "sailvl", "metaclip"]

# Dataset configuration
DATASETS = {
    "cifar10": "CIFAR-10",
    "imagenet": "ImageNet-1k", 
    "coco": "MS-COCO",
    "flickr30k": "Flickr30k",
}


def get_font(size=20):
    """Get font"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    return ImageFont.load_default()


def load_images_for_model(appendix_dir: Path, dataset: str, idx: int, model: str):
    """Load all neighbor images for a model"""
    subdir = appendix_dir / f"{dataset}_idx{idx}"
    if not subdir.exists():
        return None, []
    
    # Load query image
    query_path = subdir / f"00_query_{idx}.png"
    if not query_path.exists():
        query_files = list(subdir.glob("*query*.png"))
        if query_files:
            query_path = query_files[0]
        else:
            return None, []
    
    query_img = Image.open(query_path).convert('RGB')
    
    # Load neighbor images
    neighbors = []
    for rank in range(9):
        pattern = f"{model}_{rank}_*_cos*.png"
        matches = list(subdir.glob(pattern))
        if matches:
            neighbor_path = matches[0]
            filename = neighbor_path.name
            cos_start = filename.find("cos") + 3
            cos_end = filename.find(".png")
            try:
                cosine = float(filename[cos_start:cos_end])
            except:
                cosine = 0.0
            
            neighbor_img = Image.open(neighbor_path).convert('RGB')
            neighbors.append((neighbor_img, cosine))
        else:
            neighbors.append((None, 0.0))
    
    return query_img, neighbors


def create_row_image(query_img, neighbors, model: str, row_width: int = FIGURE_WIDTH_PX):
    """Create one row of images (query + 9 neighbors)"""
    # Calculate each image size
    total_padding = IMAGE_PADDING * (NUM_IMAGES_PER_ROW + 1)
    img_width = (row_width - total_padding) // NUM_IMAGES_PER_ROW
    img_height = img_width  # Square

    # Row height = image height + label height
    row_height = img_height + LABEL_HEIGHT
    
    # Create row image
    row_img = Image.new('RGB', (row_width, row_height), 'white')
    draw = ImageDraw.Draw(row_img)
    font = get_font(14)
    small_font = get_font(11)
    
    # Draw query image
    x = IMAGE_PADDING
    if query_img:
        resized = query_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        row_img.paste(resized, (x, 0))
        # Label: "Query"
        label = "Query"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        draw.text((x + (img_width - label_width) // 2, img_height + 8), 
                  label, fill='black', font=font)
    
    x += img_width + IMAGE_PADDING
    
    # Draw neighbor images
    for i, (neighbor_img, cosine) in enumerate(neighbors):
        if neighbor_img:
            resized = neighbor_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
            row_img.paste(resized, (x, 0))
        else:
            draw.rectangle([x, 0, x + img_width, img_height], fill='lightgray')
        
        # Label: "Sim: 0.xxx"
        label = f"Sim: {cosine:.3f}"
        bbox = draw.textbbox((0, 0), label, font=small_font)
        label_width = bbox[2] - bbox[0]
        draw.text((x + (img_width - label_width) // 2, img_height + 10), 
                  label, fill='black', font=small_font)
        
        x += img_width + IMAGE_PADDING
    
    return row_img


def create_dataset_figure(appendix_dir: Path, dataset: str, idx: int, output_dir: Path):
    """Create comparison figure for all models in one dataset"""
    rows = []
    
    for model in MODELS:
        query_img, neighbors = load_images_for_model(appendix_dir, dataset, idx, model)
        if query_img is None:
            print(f"    {MODEL_FULL_NAMES[model]}: no data")
            continue
        
        row_img = create_row_image(query_img, neighbors, model)
        rows.append((model, row_img))
        print(f"    {MODEL_FULL_NAMES[model]}: created successfully")
    
    if not rows:
        return None
    
    # Merge all rows
    total_height = sum(row.height + TITLE_HEIGHT for _, row in rows)
    combined = Image.new('RGB', (FIGURE_WIDTH_PX, total_height), 'white')
    draw = ImageDraw.Draw(combined)
    title_font = get_font(22)
    
    y = 0
    title_font_small = get_font(16)  # Smaller font to fit long model names
    for model, row_img in rows:
        # Draw model title
        title = MODEL_FULL_NAMES[model]
        bbox = draw.textbbox((0, 0), title, font=title_font_small)
        title_width = bbox[2] - bbox[0]
        draw.text(((FIGURE_WIDTH_PX - title_width) // 2, y + 18), 
                  title, fill='black', font=title_font_small)
        y += TITLE_HEIGHT
        
        # Paste row image
        combined.paste(row_img, (0, y))
        y += row_img.height
    
    # Save
    dataset_name = DATASETS[dataset].replace("-", "").replace(" ", "_")
    output_path = output_dir / f"appendix_{dataset_name}_idx{idx}.png"
    combined.save(output_path, dpi=(DPI, DPI))
    print(f"  Saved to: {output_path}")
    
    return output_path


def create_all_figures(appendix_dir: Path, idx: int, output_dir: Path):
    """Create figures for all datasets"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating appendix figures (idx={idx})")
    print("=" * 50)
    
    for dataset, dataset_name in DATASETS.items():
        print(f"\nDataset: {dataset_name}")
        create_dataset_figure(appendix_dir, dataset, idx, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Create paper appendix figures')
    parser.add_argument('--appendix-dir', type=str, 
                        default='./appendix',
                        help='Appendix directory path')
    parser.add_argument('--idx', type=int, default=1824, help='Sample index')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: appendix_figures)')
    args = parser.parse_args()
    
    appendix_dir = Path(args.appendix_dir)
    output_dir = Path(args.output_dir) if args.output_dir else appendix_dir / "figures"
    
    create_all_figures(appendix_dir, args.idx, output_dir)
    
    print("\n" + "=" * 50)
    print(f"Done! Figures saved in: {output_dir}")


if __name__ == "__main__":
    main()
