import argparse
import os
import random
import time
from datetime import timedelta

import openslide
from tqdm import tqdm

from preprocess import (
    create_tissue_mask,
    create_tissue_tiles,
    crop_rect_from_slide,
    make_tile_QC_fig,
    tile_is_not_empty,
)


def write_report_block(report_file, title, rows):
    header = f"\n========== {title} ==========\n"
    columns = f"{'Phase':<35} {'Value':<40}\n"
    line = "-" * 80 + "\n"

    report_file.write(header)
    report_file.write(columns)
    report_file.write(line)

    print(header.strip())
    print(columns.strip())
    print(line.strip())

    for phase, value in rows:
        value_str = str(value)  # <-- FIX CRITICO
        row = f"{phase:<35} {value_str:<40}\n"
        report_file.write(row)
        print(row.strip())


parser = argparse.ArgumentParser(description="Script to sample tiles from a WSI and save them as individual image files")
parser.add_argument("--input_slide", type=str, help="Path to input WSI file")
parser.add_argument("--output_dir", type=str, help="Directory to save output tile files")
parser.add_argument("--tile_size", type=int, required=True)
parser.add_argument("--n", type=int, default=2048)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_size", type=int, default=224)
args = parser.parse_args()

random.seed(args.seed)

start_total = time.time()

slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
wsi = openslide.open_slide(args.input_slide)

QC_DIR = os.path.join(args.output_dir, "QC")
TILE_DIR = os.path.join(args.output_dir, "train")
slide_dir = os.path.join(TILE_DIR, slide_id)
REPORT_PATH = os.path.join(args.output_dir, f"{slide_id}_report.txt")

os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(TILE_DIR, exist_ok=True)
os.makedirs(slide_dir, exist_ok=True)

with open(REPORT_PATH, "w") as report_file:

    # --- WSI SUMMARY ---
    write_report_block(report_file, "WSI SUMMARY", [
        ("Slide ID", slide_id),("Slide Path", args.input_slide),
        ("Dimensions", str(wsi.dimensions)),
        ("Levels", str(wsi.level_count)),
        ("Level dimensions", str(wsi.level_dimensions)),
        ("Objective power", str(wsi.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 'unknown'))),
        ("MPP", f"{wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, 'unknown')} x {wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y, 'unknown')}")
    ])

    # --- Tissue mask creation ---
    start_mask = time.time()
    seg_level = wsi.get_best_level_for_downsample(64)
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level)
    end_mask = time.time()
    mask_time = end_mask - start_mask

    write_report_block(report_file, "TISSUE MASK CREATION", [
        ("Segmentation level", seg_level),
        ("Elapsed time", timedelta(seconds=mask_time))
    ])

    # --- Create candidate tiles ---
    start_tiles = time.time()
    tiles_from_tissue = create_tissue_tiles(wsi, tissue_mask_scaled, args.tile_size)
    end_tiles = time.time()
    tiles_time = end_tiles - start_tiles

    write_report_block(report_file, "CANDIDATE TILE EXTRACTION", [
        ("Candidate tiles", len(tiles_from_tissue)),
        ("Elapsed time", timedelta(seconds=tiles_time))
    ])

    # --- Filter empty tiles with trackers ---
    total_crop_time = 0.0
    total_filter_time = 0.0
    crop_calls = 0
    filter_calls = 0
    filtered_tiles = []

    start_filter = time.time()

    for rect in tqdm(tiles_from_tissue, desc="Filtering empty tiles"):
        # crop tracker
        start_crop = time.time()
        img = crop_rect_from_slide(wsi, rect)
        end_crop = time.time()
        total_crop_time += (end_crop - start_crop)
        crop_calls += 1

        # filter tracker
        start_tile_filter = time.time()
        keep = tile_is_not_empty(img, threshold_white=20)
        end_tile_filter = time.time()
        total_filter_time += (end_tile_filter - start_tile_filter)
        filter_calls += 1

        if keep:
            filtered_tiles.append(rect)

    end_filter = time.time()
    filter_total_time = end_filter - start_filter

    write_report_block(report_file, "EMPTY TILE FILTERING", [
        ("Input tiles", len(tiles_from_tissue)),
        ("Tiles kept", len(filtered_tiles)),
        ("crop_rect_from_slide calls", crop_calls),
        ("tile_is_not_empty calls", filter_calls),
        ("Total crop time", timedelta(seconds=total_crop_time)),
        ("Avg crop time / tile", timedelta(seconds=total_crop_time / crop_calls)),
        ("Total filter time", timedelta(seconds=total_filter_time)),
        ("Avg filter time / tile", timedelta(seconds=total_filter_time / filter_calls)),
        ("Total filtering phase time", timedelta(seconds=filter_total_time))
    ])

    # --- Sample tiles ---
    sampled_tiles = random.sample(filtered_tiles, min(args.n, len(filtered_tiles)))

    write_report_block(report_file, "SAMPLING", [
        ("Requested tiles", args.n),
        ("Available non-empty tiles", len(filtered_tiles)),
        ("Sampled tiles", len(sampled_tiles))
    ])

    # --- QC figure ---
    start_qc = time.time()
    qc_img = make_tile_QC_fig(filtered_tiles, wsi, seg_level, 2, extra_tiles=sampled_tiles)
    qc_img_target_width = 1920
    qc_img = qc_img.resize(
        (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
    )
    qc_img_file_path = os.path.join(
        QC_DIR, f"{slide_id}_sampled{len(sampled_tiles)}_{len(filtered_tiles)}tiles_QC.png"
    )
    qc_img.save(qc_img_file_path)
    end_qc = time.time()

    qc_time = end_qc - start_qc

    write_report_block(report_file, "QC FIGURE CREATION", [
        ("QC image path", qc_img_file_path),
        ("Elapsed time", timedelta(seconds=qc_time))
    ])

    # --- Save tiles ---
    start_save = time.time()
    for i, tile in enumerate(tqdm(sampled_tiles, desc="Saving sampled tiles")):
        img = crop_rect_from_slide(wsi, tile)
        img = img.resize((args.out_size, args.out_size))
        img = img.convert("RGB")
        out_file = os.path.join(slide_dir, f"{slide_id}_tile_{i}.png")
        img.save(out_file, format="png", quality=100, subsampling=0)
    end_save = time.time()

    save_time = end_save - start_save

    write_report_block(report_file, "TILE SAVING", [
        ("Saved tiles", len(sampled_tiles)),
        ("Elapsed time", timedelta(seconds=save_time))
    ])

    # --- TOTAL ---
    total_time = time.time() - start_total
    write_report_block(report_file, "TOTAL EXECUTION TIME", [
        ("Total time elapsed", timedelta(seconds=total_time))
    ])

print(f"\nReport saved to: {REPORT_PATH}")
