#!/usr/bin/env python3
import os
import random
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
from tqdm import tqdm
from scipy.stats import ttest_rel

from preprocess import (
    crop_rect_from_slide_at_level,
    tile_is_not_empty,
    create_tissue_mask,
    create_tissue_tiles,
    make_tile_QC_fig
)

# ---------------- FUNZIONI ----------------

def generate_quality_check_image(wsi, slide_id, tiles, seg_level, sampled_tiles, save_dir):
    qc_img = make_tile_QC_fig(tiles, wsi, seg_level, 2, extra_tiles=sampled_tiles)
    qc_img_target_width = 1920
    qc_img = qc_img.resize(
        (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
    )
    qc_img_file_path = os.path.join(
        save_dir,
        f"{slide_id}_sampled{len(sampled_tiles)}_{len(tiles)}tiles_QC.png"
    )
    qc_img.save(qc_img_file_path)
    print(f"QC image saved to: {qc_img_file_path}")

def save_tissue_mask_image(mask_geom, save_path):
    plt.figure(figsize=(6,6))
    if mask_geom.geom_type == "MultiPolygon":
        for poly in mask_geom.geoms:
            x, y = poly.exterior.xy
            plt.fill(x, y, color='gray', alpha=0.5)
    else:
        x, y = mask_geom.exterior.xy
        plt.fill(x, y, color='gray', alpha=0.5)
    plt.axis("equal")
    plt.title("Tissue mask geometry")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Tissue mask image saved to: {save_path}")

# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark WSI tile reading times.")
    parser.add_argument("--wsi", type=str, required=True, help="Path to WSI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--tile_size_microns", type=int, default=360, help="Tile size in microns")
    parser.add_argument("--out_size", type=int, default=224, help="Output tile pixel size")
    parser.add_argument("--levels", type=int, nargs="+", default=[0,1,2,3,4], help="Levels to test")
    parser.add_argument("--n_tiles", type=int, default=-1, help="Number of tiles to sample (-1 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tile sampling")
    parser.add_argument("--RGB_threshold", type=float, default=0.20, help="Threshold for non-empty tile")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- OPEN WSI ----------------
    wsi = openslide.OpenSlide(args.wsi)
    slide_id = os.path.splitext(os.path.basename(args.wsi))[0]

    # ---------------- TISSUE MASK ----------------
    seg_level = wsi.get_best_level_for_downsample(64)
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level=seg_level)
    
    mask_image_path = os.path.join(args.output_dir, f"{slide_id}_tissue_mask.png")
    save_tissue_mask_image(tissue_mask_scaled, mask_image_path)

    # ---------------- CREATE TILES ----------------
    tiles = create_tissue_tiles(wsi, tissue_mask_scaled, tile_size_microns=args.tile_size_microns)

    # ---------------- SAMPLE TILES ----------------
    random.seed(args.seed)
    if args.n_tiles == -1 or args.n_tiles >= len(tiles):
        sampled_tiles = tiles
    else:
        sampled_tiles = random.sample(tiles, args.n_tiles)

    # ---------------- QC IMAGE ----------------
    generate_quality_check_image(
        wsi=wsi,
        slide_id=slide_id,
        tiles=tiles,
        seg_level=seg_level,
        sampled_tiles=sampled_tiles,
        save_dir=args.output_dir
    )

    # ---------------- TIMING PER LEVEL ----------------
    tile_times = {i: {} for i in range(len(sampled_tiles))}
    timing_dir = os.path.join(args.output_dir, "TIMING_PER_LEVEL")
    os.makedirs(timing_dir, exist_ok=True)

    for level in args.levels:
        level_dir = os.path.join(timing_dir, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)
        print(f"\n--- Benchmark level {level} ---")

        for i, tile in tqdm(enumerate(sampled_tiles), total=len(sampled_tiles)):
            start = time.time()
            img = crop_rect_from_slide_at_level(wsi, tile, level=level)
            is_not_empty = tile_is_not_empty(img, threshold_white=args.RGB_threshold)

            if is_not_empty:
                img = img.resize((args.out_size, args.out_size))
                img = img.convert("RGB")
                out_file = os.path.join(level_dir, f"tile_{i}.png")
                img.save(out_file, format="png", quality=100, subsampling=0)
                elapsed = time.time() - start
            else:
                elapsed = np.nan

            tile_times[i][f"level_{level} (s)"] = elapsed

    # ---------------- SAVE TILE TIMINGS ----------------
    df_tiles = pd.DataFrame.from_dict(tile_times, orient="index")
    df_tiles.insert(0, "tile_index", df_tiles.index)
    df_tiles.reset_index(drop=True, inplace=True)
    excel_path = os.path.join(timing_dir, f"tile_timings_{args.tile_size_microns}um.xlsx")
    df_tiles.to_excel(excel_path, index=False)
    print(f"Tile timings saved to: {excel_path}")

    # ---------------- TOTAL TIME PER LEVEL PLOT ----------------
    total_times = []
    for level in args.levels:
        col = f"level_{level} (s)"
        df_curr = df_tiles[col].dropna()
        total_times.append(df_curr.sum())

    plt.figure(figsize=(8,5))
    plt.plot(args.levels, total_times, marker='o', linestyle='-', color='blue')
    plt.xlabel("Level")
    plt.ylabel("Total Extraction Time (s)")
    plt.title(f"Tile Extraction Total Time per Level (tile size = {args.tile_size_microns} μm, output size = {args.out_size} px)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(args.levels)
    plt.tight_layout()

    total_time_plot_path = os.path.join(timing_dir, f"total_time_per_level_{args.tile_size_microns}um.png")
    plt.savefig(total_time_plot_path)
    plt.close()
    print(f"Total time per level plot saved to: {total_time_plot_path}")

    # ---------------- T-TEST ----------------
    ttest_rows = []
    for i, level in enumerate(args.levels):
        col_curr = f"level_{level} (s)"
        df_curr = df_tiles[col_curr].dropna()
        mean_time = df_curr.mean()
        std_time = df_curr.std()
        t_prev = p_prev = t_0 = p_0 = np.nan

        if i > 0:
            col_prev = f"level_{args.levels[i-1]} (s)"
            df_prev = df_tiles[col_prev].dropna()
            min_len = min(len(df_prev), len(df_curr))
            if min_len > 1:
                t_prev, p_prev = ttest_rel(df_prev.values[:min_len], df_curr.values[:min_len], alternative="greater")

            col_0 = f"level_0 (s)"
            df_0 = df_tiles[col_0].dropna()
            min_len = min(len(df_0), len(df_curr))
            if min_len > 1:
                t_0, p_0 = ttest_rel(df_0.values[:min_len], df_curr.values[:min_len], alternative="greater")

        ttest_rows.append({
            "level": level,
            "mean_time": mean_time,
            "std_time": std_time,
            "t_stat_vs_prev": t_prev,
            "p_value_vs_prev": p_prev,
            "t_stat_vs_level0": t_0,
            "p_value_vs_level0": p_0
        })

    df_ttest = pd.DataFrame(ttest_rows)
    ttest_excel_path = os.path.join(timing_dir, f"ttest_results_{args.tile_size_microns}um.xlsx")
    df_ttest.to_excel(ttest_excel_path, index=False)
    print(f"T-test results saved to: {ttest_excel_path}")

    # ---------------- BOXPLOT ----------------
    data_per_level = []
    x_labels = []
    resolutions = []
    mpp_x = float(wsi.properties['openslide.mpp-x'])
    mpp_y = float(wsi.properties['openslide.mpp-y'])
    wsi_res = max(mpp_x, mpp_y)

    for level in args.levels:
        col = f"level_{level} (s)"
        data_per_level.append(df_tiles[col].dropna().values)
        downsample_factor = wsi.level_downsamples[level]
        res_at_level = wsi_res * downsample_factor
        resolutions.append(res_at_level)
        x_labels.append(f"Level {level}\n{res_at_level:.2f} μm/px")

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data_per_level,
        widths=0.6,
        showfliers=True,
        medianprops=dict(color="green", linewidth=2)
    )
    plt.xlabel("Downsample level and resolution")
    plt.ylabel("Time per tile (s)")
    plt.title(f"Reading Time (tile size = {args.tile_size_microns} μm, output size = {args.out_size}x{args.out_size}, res = {(args.tile_size_microns/args.out_size):.2f} μm/px)")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(range(1, len(x_labels)+1), x_labels)
    plt.tight_layout()
    boxplot_path = os.path.join(timing_dir, f"timing_distribution_{args.tile_size_microns}um.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Boxplot saved to: {boxplot_path}")

if __name__ == "__main__":
    main()
