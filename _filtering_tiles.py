import time
from tqdm import tqdm
import openslide
from preprocess import crop_rect_from_slide, tile_is_not_empty, create_tissue_tiles, create_tissue_mask
from datetime import timedelta

wsi_path = r"/mnt/d/original_scans/NikonS60/22 B 07247 D1.ndpi"
tile_size_um = 360
n_tiles = 100

# Apri la slide
wsi = openslide.open_slide(wsi_path)

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
print(f"Tiles from tissue: {tiles_from_tissue}")

# --- Benchmark loop originale ---
total_crop_time = 0.0
total_filter_time = 0.0
filtered_tiles_RGB = []

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

