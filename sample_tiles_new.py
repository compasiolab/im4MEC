#!/usr/bin/env python3
import os
import random
import argparse
from PIL import Image
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
    print(f"Tissue mask image saved to: {save_path}")

def generate_quality_check_image(wsi, slide_id, tiles, seg_level, sampled_tiles, save_path):
    qc_img = make_tile_QC_fig(tiles, wsi, seg_level, 2, extra_tiles=sampled_tiles)
    qc_img_target_width = 1920
    qc_img = qc_img.resize(
        (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
    )
    qc_img.save(save_path)
    print(f"QC image saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract tiles from a WSI for dataset creation.")
    parser.add_argument("--wsi", type=str, required=True, help="Path to WSI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted tiles and QC images")
    parser.add_argument("--tile_size_microns", type=int, default=360, help="Tile size in microns")
    parser.add_argument("--out_size", type=int, default=224, help="Tile pixel output size")
    parser.add_argument("--RGB_threshold", type=float, default=0.20, help="Threshold for non-empty tile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (optional if sampling)")
    parser.add_argument("--n_tiles", type=int, default=-1, help="Number of tiles to sample (-1 = all)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_dir = os.path.join(args.output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    # ---------------- OPEN WSI ----------------
    wsi = openslide.OpenSlide(args.wsi)
    slide_id = os.path.splitext(os.path.basename(args.wsi))[0]

    # ---------------- COMPUTE OPENING LEVEL ----------------
    scaling_factor = compute_scaling_factor(
        wsi,
        tile_size_microns=args.tile_size_microns,
        n_pixel_resized=args.out_size
    )
    opening_level = wsi.get_best_level_for_downsample(scaling_factor)
    print(f"Opening level selected: {opening_level}")

    # ---------------- CREATE TISSUE MASK ----------------
    seg_level = wsi.get_best_level_for_downsample(64)
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level=seg_level)

    # salva tissue mask come immagine
    mask_image_path = os.path.join(args.output_dir, f"{slide_id}_tissue_mask.png")
    save_tissue_mask_image(tissue_mask_scaled, mask_image_path)

    # ---------------- CREATE TILES ----------------
    tiles = create_tissue_tiles(wsi, tissue_mask_scaled, tile_size_microns=args.tile_size_microns)
    print(f"Total tiles from tissue mask: {len(tiles)}")

    # ---------------- SAMPLE TILES ----------------
    random.seed(args.seed)
    if args.n_tiles == -1 or args.n_tiles >= len(tiles):
        sampled_tiles = tiles
    else:
        sampled_tiles = random.sample(tiles, args.n_tiles)
    print(f"Tiles selected for QC image and saving: {len(sampled_tiles)}")

    # ---------------- GENERATE QC IMAGE ----------------
    qc_image_path = os.path.join(args.output_dir, f"{slide_id}_QC.png")
    generate_quality_check_image(
        wsi=wsi,
        slide_id=slide_id,
        tiles=tiles,
        seg_level=seg_level,
        sampled_tiles=sampled_tiles,
        save_path=qc_image_path
    )

    # ---------------- SAVE TILES ----------------
    for i, tile in enumerate(sampled_tiles):
        img = crop_rect_from_slide_at_level(wsi, tile, level=opening_level)
        if tile_is_not_empty(img, threshold_white=args.RGB_threshold):
            img = img.resize((args.out_size, args.out_size))
            img = img.convert("RGB")
            out_file = os.path.join(train_dir, f"tile_{i}.png")
            img.save(out_file, format="png", quality=100, subsampling=0)

    print(f"All tiles saved in: {train_dir}")

if __name__ == "__main__":
    main()
