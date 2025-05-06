import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend suitable for multiprocessing
import matplotlib.colors as mcolors
import os
import argparse
import glob
import re
import multiprocessing
from functools import partial
import time
from datetime import timedelta


# --- Utility Functions ---
# (Keep mkdir, load_nifti_slice, load_npy_slice, apply_window, denormalize_neg1_1_to_hu functions as before)
def mkdir(path):
    if not os.path.exists(path): os.makedirs(path); print(f"--- Created output directory: {path} ---")


def load_nifti_slice(nii_data, slice_index, axis=2):
    try:
        if axis == 0:
            slice_data = nii_data[slice_index, :, :]
        elif axis == 1:
            slice_data = nii_data[:, slice_index, :]
        else:
            slice_data = nii_data[:, :, slice_index]
        return np.rot90(slice_data)
    except Exception as e:
        print(f"Error extracting slice {slice_index}: {e}"); return None


def load_npy_slice(filepath):
    try:
        return np.rot90(np.load(filepath))
    except Exception as e:
        print(f"Error loading NumPy {filepath}: {e}"); return None


def apply_window(image_data, window_center, window_width):
    min_val = window_center - window_width / 2;
    max_val = window_center + window_width / 2
    windowed_image = np.clip(image_data, min_val, max_val)
    if max_val > min_val:
        return (windowed_image - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(windowed_image)


def denormalize_neg1_1_to_hu(normalized_data, original_win_min=-1000, original_win_max=1000):
    data_01 = (normalized_data + 1.0) / 2.0
    return data_01 * (original_win_max - original_win_min) + original_win_min


# --- Worker Function for Processing One Volume ---

def process_volume(volume_info, args, bone_mask_cmap=None, cmap_norm=None):  # Made cmap/norm optional
    """
    Loads data for one volume and processes/visualizes all its relevant slices.
    volume_info: tuple (base_filename_core, original_nii_filepath, bone_mask_filepath OR None) # Bone mask path is now optional
    args: Namespace object with command line arguments
    bone_mask_cmap: Pre-defined colormap for bone mask (optional)
    cmap_norm: Pre-defined normalization for the colormap (optional)
    Returns: tuple (processed_count, error_count) for this volume
    """
    base_filename_core, original_nii_filepath, bone_mask_filepath = volume_info  # Unpack
    processed_count_vol = 0
    error_count_vol = 0
    pid = os.getpid()

    print(f"[PID {pid}] Processing volume: {base_filename_core}")

    try:
        # Load required volumes
        nii_low = nib.load(original_nii_filepath)
        data_low = nii_low.get_fdata()

        # Load bone mask volume ONLY if path was provided and exists
        data_bone_mask = None
        if bone_mask_filepath and os.path.exists(bone_mask_filepath):
            try:
                nii_bone_mask = nib.load(bone_mask_filepath)
                data_bone_mask = nii_bone_mask.get_fdata()
                if data_low.shape != data_bone_mask.shape:
                    print(f"[PID {pid}] Warning: Shape mismatch for bone mask {base_filename_core}. Ignoring mask.")
                    data_bone_mask = None  # Ignore if shape doesn't match
            except Exception as e:
                print(f"[PID {pid}] Warning: Could not load bone mask {bone_mask_filepath}. Error: {e}. Ignoring mask.")
                data_bone_mask = None
        elif bone_mask_filepath:  # Path provided but file doesn't exist
            print(f"[PID {pid}] Warning: Bone mask file not found: {bone_mask_filepath}. Ignoring mask.")

        num_slices = data_low.shape[args.slice_axis]
        base_filename_lh = os.path.basename(original_nii_filepath).replace('.nii.gz', '')

        for slice_index in range(num_slices):
            lh_filename = f"{base_filename_lh}_slice{slice_index:04d}_lh.npy"
            lh_filepath = os.path.join(args.input_lh_dir, lh_filename)

            if not os.path.exists(lh_filepath):
                continue

            slice_lh_neg1_1 = load_npy_slice(lh_filepath)
            if slice_lh_neg1_1 is None: error_count_vol += 1; continue

            slice_low_hu = load_nifti_slice(data_low, slice_index, args.slice_axis)
            if slice_low_hu is None: error_count_vol += 1; continue

            # Extract bone mask slice if volume was loaded
            slice_bone_mask = None
            if data_bone_mask is not None:
                slice_bone_mask = load_nifti_slice(data_bone_mask, slice_index, args.slice_axis)
                # No need to check for None here again, load_nifti_slice handles internal errors

            metal_mask_overlay = (slice_low_hu > args.metal_threshold).astype(float)
            if np.sum(metal_mask_overlay) == 0: continue

            slice_low_display = apply_window(slice_low_hu, args.wc, args.ww)
            slice_lh_hu_approx = denormalize_neg1_1_to_hu(slice_lh_neg1_1, args.input_norm_min, args.input_norm_max)
            slice_lh_display = apply_window(slice_lh_hu_approx, args.wc, args.ww)

            # --- Create Plot (2 or 3 panels) ---
            num_panels = 3 if slice_bone_mask is not None else 2
            fig_width = 13 if num_panels == 2 else 19  # Adjust width based on panels
            fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 6.5))
            # Ensure axes is always iterable, even if num_panels=1 (though we expect 2 or 3)
            if num_panels == 1: axes = [axes]

            fig.suptitle(f"{base_filename_core} - Slice {slice_index} (Win: C={args.wc} W={args.ww})", fontsize=14,
                         y=0.98)
            img_height, img_width = slice_low_display.shape
            extent = (0, img_width, img_height, 0)

            # Panel 0: Original Input (X_low)
            axes[0].imshow(slice_low_display, cmap='gray', vmin=0, vmax=1, extent=extent, aspect='auto')
            axes[0].set_xlabel("Original Input (X_low)", fontsize=12);
            axes[0].set_xticks([]);
            axes[0].set_yticks([]);
            [s.set_visible(False) for s in axes[0].spines.values()]

            # Panel 1: SemiMAR Output (X_lh) + Metal Mask Overlay
            axes[1].imshow(slice_lh_display, cmap='gray', vmin=0, vmax=1, extent=extent, aspect='auto')
            masked_overlay = np.ma.masked_where(metal_mask_overlay == 0, metal_mask_overlay)
            axes[1].imshow(masked_overlay, cmap='autumn', alpha=0.4, extent=extent, aspect='auto')
            axes[1].set_xlabel("SemiMAR Output (X_lh) + Metal Overlay", fontsize=12)
            axes[1].set_xticks([]);
            axes[1].set_yticks([]);
            [s.set_visible(False) for s in axes[1].spines.values()]

            # Panel 2: Original Bone Segmentation Mask (Conditional)
            if num_panels == 3 and slice_bone_mask is not None:
                im = axes[2].imshow(slice_bone_mask.astype(int), cmap=bone_mask_cmap, norm=cmap_norm, extent=extent,
                                    aspect='auto')
                axes[2].set_xlabel("Original Bone Mask", fontsize=12)
                axes[2].set_xticks([]);
                axes[2].set_yticks([]);
                [s.set_visible(False) for s in axes[2].spines.values()]

            # Adjust layout
            wspace = 0.1 if num_panels == 3 else 0.05
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.93, wspace=wspace)

            # --- Save Plot ---
            output_png_filename = f"{base_filename_core}_slice{slice_index:04d}_semimar_comparison.png"
            output_png_filepath = os.path.join(args.output_png_dir, output_png_filename)
            try:
                plt.savefig(output_png_filepath, dpi=150)
                processed_count_vol += 1
            except Exception as e:
                print(f"[PID {pid}] Error saving plot {output_png_filepath}: {e}")
                error_count_vol += 1
            finally:
                plt.close(fig)

    except Exception as e:
        print(f"[PID {pid}] Critical error processing volume {base_filename_core}: {e}")
        return processed_count_vol, error_count_vol + 1

    if processed_count_vol + error_count_vol > 0:
        print(
            f"[PID {pid}] Finished volume {base_filename_core}. Processed: {processed_count_vol}, Errors: {error_count_vol}")
    return processed_count_vol, error_count_vol


# --- Main Execution ---
if __name__ == "__main__":
    start_time_main = time.time()
    parser = argparse.ArgumentParser(
        description="Visualize All SemiMAR Clinical Slice Results (Parallel, Opt. Bone Mask)")
    # --- Arguments ---
    parser.add_argument("--input_lh_dir", type=str, required=True,
                        help="Directory containing the processed SemiMAR output slices (.npy files, X_lh)")
    parser.add_argument("--input_low_dir", type=str, required=True,
                        help="Directory containing the original metal-corrupted NIfTI files (X_low)")
    parser.add_argument("--input_bone_mask_dir", type=str, default=None,
                        help="(Optional) Directory containing original bone segmentation mask NIfTI files")  # Made optional
    parser.add_argument("--output_png_dir", type=str, required=True,
                        help="Directory to save the output comparison PNG files")
    parser.add_argument("--slice_axis", type=int, default=2, help="Axis along which slices were extracted")
    parser.add_argument("--wc", type=int, default=400, help="Window Center for CT display")
    parser.add_argument("--ww", type=int, default=1800, help="Window Width for CT display")
    parser.add_argument("--metal_threshold", type=int, default=2500, help="HU threshold for metal overlay")
    parser.add_argument("--input_norm_min", type=int, default=-1000,
                        help="Min HU value used for original normalization")
    parser.add_argument("--input_norm_max", type=int, default=1000, help="Max HU value used for original normalization")
    parser.add_argument("--num_mask_labels", type=int, default=4,
                        help="Number of non-background labels IF bone mask is provided")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel processes")
    args = parser.parse_args()

    mkdir(args.output_png_dir)

    # --- Prepare list of volumes to process ---
    volume_info_list = []
    low_files = sorted(glob.glob(os.path.join(args.input_low_dir, '*.nii.gz')))
    filename_pattern_base = re.compile(r"^(.*)_data\.nii\.gz$")
    for low_filepath in low_files:
        base_low_filename = os.path.basename(low_filepath)
        match = filename_pattern_base.match(base_low_filename)
        if not match: continue

        base_filename_core_prefix = match.group(1)
        core_name = base_filename_core_prefix.replace('dataset7_', '')

        bone_mask_filepath = None  # Default to None
        if args.input_bone_mask_dir:  # Only construct path if dir was provided
            bone_mask_filename = f"{core_name}_mask_4label.nii.gz"
            bone_mask_filepath = os.path.join(args.input_bone_mask_dir, bone_mask_filename)
            # We will check if it actually exists inside the worker function

        if os.path.exists(low_filepath):
            # Pass None for bone_mask_filepath if directory wasn't given
            volume_info_list.append((core_name, low_filepath, bone_mask_filepath))
        else:
            print(f"Warning: Skipping volume {core_name} due to missing input file.")

    if not volume_info_list: print("Error: No valid volumes found to process."); exit()
    print(f"Found {len(volume_info_list)} volumes to process.")

    # --- Setup Colormap (defined once, only used if masks are provided) ---
    bone_mask_cmap = None
    cmap_norm = None
    if args.input_bone_mask_dir:  # Only define if needed
        base_cmap_name = 'tab10';
        base_cmap = plt.cm.get_cmap(base_cmap_name, args.num_mask_labels)
        colors = [(0, 0, 0, 1)] + [base_cmap(i) for i in range(args.num_mask_labels)]
        bone_mask_cmap = mcolors.ListedColormap(colors)
        cmap_bounds = np.arange(args.num_mask_labels + 2) - 0.5
        cmap_norm = mcolors.BoundaryNorm(cmap_bounds, bone_mask_cmap.N)

    # --- Setup Multiprocessing Pool ---
    if args.num_workers is None:
        pool_size = max(1, multiprocessing.cpu_count() - 1)
    else:
        pool_size = args.num_workers
    print(f"Starting parallel processing with {pool_size} workers...")

    worker_partial = partial(process_volume, args=args, bone_mask_cmap=bone_mask_cmap, cmap_norm=cmap_norm)

    results = []
    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.map(worker_partial, volume_info_list)

    # --- Aggregate Results ---
    total_processed = sum(r[0] for r in results if r is not None)
    total_errors = sum(r[1] for r in results if r is not None)

    end_time_main = time.time()
    print("\n" + "=" * 30)
    print("Parallel Visualization Complete.")
    print(f"Successfully processed plots for {total_processed} slices.")
    if total_errors > 0: print(f"Encountered errors during processing for {total_errors} slices/volumes.")
    print(f"Output PNG files saved in: {args.output_png_dir}")
    print(f"Total execution time: {timedelta(seconds=int(end_time_main - start_time_main))}")  # Corrected formatting
    print("=" * 30)

