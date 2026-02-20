#!/usr/bin/env python3
import os
import random
import argparse
import matplotlib.pyplot as plt
import openslide

from preprocess import (
    crop_rect_from_slide_at_level,
    tile_is_not_empty,
    create_tissue_mask,
    create_tissue_tiles,
    compute_scaling_factor,
    make_tile_QC_fig
)


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


def generate_quality_check_image(wsi, tiles, seg_level, sampled_tiles, save_path):
    qc_img = make_tile_QC_fig(tiles, wsi, seg_level, 2, extra_tiles=sampled_tiles)
    qc_img_target_width = 1920
    qc_img = qc_img.resize(
        (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
    )
    qc_img.save(save_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract tiles from a WSI for dataset creation.")
    parser.add_argument("--wsi", type=str, required=True, help="Path to WSI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted tiles and QC images")
    parser.add_argument("--tile_size_microns", type=int, default=360, help="Tile size in microns")
    parser.add_argument("--out_size", type=int, default=224, help="Tile pixel output size")
    parser.add_argument("--RGB_threshold", type=int, default=20, help="Threshold for non-empty tile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (optional if sampling)")
    parser.add_argument("--n_tiles", type=int, default=-1, help="Number of tiles to sample (-1 = all)")

    args = parser.parse_args()

    return args


def create_output_folder(output_dir : str):
    os.makedirs(output_dir, exist_ok=True)


def process_wsi(wsi_path : str,
                tile_size_microns : float,
                out_size : int,
                output_dir : str,
                n_tiles : int,
                RGB_threshold : int = 20,
                seed : int = 42,
                level : int = None):
    
    # ---------------- OPEN WSI ----------------
    wsi = openslide.OpenSlide(wsi_path)
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    this_wsi_output_folder = output_dir + f"/{slide_id}"
    create_output_folder(this_wsi_output_folder + "/train/")

    # ---------------- COMPUTE OPENING LEVEL ----------------
    opening_level = 0
    if level is None :  
        scaling_factor = compute_scaling_factor(
            wsi,
            tile_size_microns=tile_size_microns,
            n_pixel_resized=out_size
        )
        opening_level = wsi.get_best_level_for_downsample(scaling_factor)
    else :
        opening_level = level

    # ---------------- CREATE TISSUE MASK ----------------
    seg_level = wsi.get_best_level_for_downsample(64)
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level=seg_level)
    mask_image_path = os.path.join(this_wsi_output_folder, f"tissue_mask.png")
    save_tissue_mask_image(tissue_mask_scaled, mask_image_path)

    # ---------------- CREATE TILES ----------------
    tiles = create_tissue_tiles(wsi, tissue_mask_scaled, tile_size_microns=tile_size_microns)
    random.seed(seed)
    sampled_tiles = None
    if n_tiles == -1 or n_tiles >= len(tiles):
        sampled_tiles = tiles
    else:
        sampled_tiles = random.sample(tiles, n_tiles)

    # ---------------- GENERATE QC IMAGE ----------------
    qc_image_path = os.path.join(this_wsi_output_folder, f"QC.png")
    generate_quality_check_image(
        wsi=wsi,
        tiles=tiles,
        seg_level=seg_level,
        sampled_tiles=sampled_tiles,
        save_path=qc_image_path
    )

    # ---------------- SAVE TILES ----------------
    for i, tile in enumerate(sampled_tiles):
        try :
            img = crop_rect_from_slide_at_level(wsi, tile, level=opening_level)
            if tile_is_not_empty(img, threshold_white=RGB_threshold):
                img = img.resize((out_size, out_size))
                img = img.convert("RGB")
                out_file = os.path.join(this_wsi_output_folder, f"train/tile_{i}.png")
                img.save(out_file, format="png", quality=100, subsampling=0)
        except Exception as e:
            continue

def main():
    
    args = parse_arguments()

    create_output_folder(output_dir=args.output_dir)
    
    process_wsi(
        wsi_path=args.wsi,
        tile_size_microns=args.tile_size_microns,
        out_size=args.out_size,
        output_dir=args.output_dir,
        n_tiles=args.n_tiles,
        RGB_threshold=args.RGB_threshold,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
