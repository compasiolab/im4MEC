import time
from tqdm import tqdm
import openslide
from preprocess import crop_rect_from_slide, tile_is_not_empty, create_tissue_tiles, create_tissue_mask, make_tile_QC_fig
from datetime import timedelta
import os

wsi_path = r"D:\original_scans\panoramic\23 B 28240 A1.mrxs"
tile_size_um = 360
n_tiles = 100
output_dir = "prova/"

# Apri la slide
wsi = openslide.open_slide(wsi_path)
slide_id = os.path.basename(wsi_path).split(".")[0]

QC_DIR = os.path.join(output_dir, "QC")
TILE_DIR = os.path.join(output_dir, "train")
slide_dir = os.path.join(TILE_DIR, slide_id)

os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(TILE_DIR, exist_ok=True)
os.makedirs(slide_dir, exist_ok=True)

# --- Tissue mask creation ---
start_mask = time.time()
seg_level = wsi.get_best_level_for_downsample(64)
tissue_mask_scaled = create_tissue_mask(wsi, seg_level)
end_mask = time.time()
print(f"Tissue mask creation took: {timedelta(seconds=end_mask - start_mask)}")

# --- Create candidate tiles ---
start_tiles = time.time()
tiles_from_tissue = create_tissue_tiles(wsi, tissue_mask_scaled, tile_size_um)
end_tiles = time.time()
print(f"Candidate tile extraction took: {timedelta(seconds=end_tiles - start_tiles)}")
print(f"Candidate tiles before RGB filtering: {len(tiles_from_tissue)}")

# --- Benchmark loop originale ---
total_crop_time = 0.0
total_filter_time = 0.0
filtered_tiles_RGB = []

# Build a figure for quality control purposes, to check if the tiles are where we expect them.
qc_img = make_tile_QC_fig(tiles_from_tissue, wsi, seg_level, 2, extra_tiles=None)
qc_img_target_width = 1920
qc_img = qc_img.resize(
    (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
)
qc_img_file_path = os.path.join(
    QC_DIR, f"{slide_id}_tiles_QC.png"
)
qc_img.save(qc_img_file_path)

for rect in tqdm(tiles_from_tissue[:n_tiles], desc="Filtering empty tiles"):
    # misura crop_rect_from_slide
    start_crop = time.time()
    img = crop_rect_from_slide(wsi, rect)
    end_crop = time.time()
    total_crop_time += (end_crop - start_crop)

    # misura tile_is_not_empty
    start_filter = time.time()
    keep = tile_is_not_empty(img, threshold_white=20)
    end_filter = time.time()
    total_filter_time += (end_filter - start_filter)

    if keep:
        filtered_tiles_RGB.append(rect)

from datetime import timedelta

def format_td(seconds):
    return str(timedelta(seconds=seconds))

print("\n========== TILE FILTERING REPORT ==========\n")
print(f"Tiles kept: {len(filtered_tiles_RGB)}/{n_tiles}\n")

header = f"{'Phase':<25} {'Calls':<10} {'Total Time':<15} {'Avg / Tile':<15}"
line = "-" * len(header)

print(header)
print(line)

print(f"{'crop_rect_from_slide':<25} "
      f"{n_tiles:<10} "
      f"{format_td(total_crop_time):<15} "
      f"{format_td(total_crop_time / n_tiles):<15}")

print(f"{'tile_is_not_empty':<25} "
      f"{n_tiles:<10} "
      f"{format_td(total_filter_time):<15} "
      f"{format_td(total_filter_time / n_tiles):<15}")

print(f"{'TOTAL LOOP':<25} "
      f"{n_tiles:<10} "
      f"{format_td(total_crop_time + total_filter_time):<15} "
      f"{'-':<15}")

print("\n==========================================\n")

