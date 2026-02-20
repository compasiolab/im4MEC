from multiprocessing import Pool
import pandas as pd
import os
from sample_tiles_new import process_wsi

def process_wsi_safe(wsi_path, tile_size_microns, out_size, output_dir, n_tiles, rgb_threshold, seed, level = None):
    try:
        process_wsi(
            wsi_path=wsi_path,
            tile_size_microns=tile_size_microns,
            out_size=out_size,
            output_dir=output_dir,
            n_tiles=n_tiles,
            RGB_threshold=rgb_threshold,
            seed=seed,
            level = level 
        )
        return (True, wsi_path, None)
    except Exception as e:
        return (False, wsi_path, str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel WSI tiling using metadata file")
    parser.add_argument("--metadata_file", type=str, required=True, help="Excel/CSV file with WSI metadata")
    parser.add_argument("--wsi_column", type=str, required=True, help="Column name containing WSI paths")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel WSI to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save tiles and QC images")
    parser.add_argument("--tile_size_microns", type=int, default=360)
    parser.add_argument("--out_size", type=int, default=224)
    parser.add_argument("--rgb_threshold", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_tiles", type=int, default=-1)

    args = parser.parse_args()

    # ------------------ Read metadata file ------------------
    if args.metadata_file.endswith(".xlsx") or args.metadata_file.endswith(".xls"):
        metadata_df = pd.read_excel(args.metadata_file)
    elif args.metadata_file.endswith(".csv"):
        metadata_df = pd.read_csv(args.metadata_file)
    else:
        raise ValueError("metadata_file must be an Excel or CSV file")

    if args.wsi_column not in metadata_df.columns:
        raise ValueError(f"The column '{args.wsi_column}' does not exist in the metadata file")

    # Generate WSI list
    wsi_list = metadata_df[args.wsi_column].tolist()
    print(f"Found {len(wsi_list)} WSI in the metadata file")

    # ------------------ Prepare arguments for starmap ------------------
    worker_args = [
        (
            wsi_path,
            args.tile_size_microns,
            args.out_size,
            args.output_dir,
            args.n_tiles,
            args.rgb_threshold,
            args.seed
        )
        for wsi_path in wsi_list
    ]

    # ------------------ Parallel execution ------------------
    with Pool(args.num_workers) as pool:
        results = pool.starmap(process_wsi_safe, worker_args)

    # ------------------ Collect errors ------------------
    error_rows = []
    for success, wsi_path, error in results:
        if not success:
            error_rows.append({"wsi_path": wsi_path, "preprocessing_error": error})

    if len(error_rows) > 0:
        error_df = pd.DataFrame(error_rows)
        error_df.to_excel("tiling_errors.xlsx", index=False)
        print(f"Saved {len(error_df)} errors to tiling_errors.xlsx")
    else:
        print("No tiling errors encountered.")
