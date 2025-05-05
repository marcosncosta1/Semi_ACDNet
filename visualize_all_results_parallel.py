import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for multiprocessing
import matplotlib.colors as mcolors
import os
import argparse
import glob
import re
import multiprocessing
from functools import partial
import time

# --- Utility Functions ---

# (Keep mkdir, load_nifti_slice, load_npy_slice, apply_window, denormalize_neg1_1_to_hu functions as before)
def mkdir(path):
    if not os.path.exists(path): os.makedirs(path); print(f"--- Created output directory: {path} ---")
def load_nifti_slice(nii_data, slice_index, axis=2): # Modified to take data array
    """Extracts a specific slice from pre-loaded NIfTI data."""
    try:
        if axis == 0: slice_data = nii_data[slice_index, :, :]
        elif axis == 1: slice_data = nii_data[:, slice_index, :]
        else: slice_data = nii_data[:, :, slice_index]
        return np.rot90(slice_data)
    except Exception as e: print(f"Error extracting slice {slice_index}: {e}"); return None
def load_npy_slice(filepath):
    try: return np.rot90(np.load(filepath))
    except Exception as e: print(f"Error loading NumPy {filepath}: {e}"); return None
def apply_window(image_data, window_center, window_width):
    min_val = window_center - window_width / 2; max_val = window_center + window_width / 2
    windowed_image = np.clip(image_data, min_val, max_val)
    if max_val > min_val: return (windowed_image - min_val) / (max_val - min_val)
    else: return np.zeros_like(windowed_image)
def denormalize_neg1_1_to_hu(normalized_data, original_win_min=-1000, original_win_max=1000):
    data_01 = (normalized_data + 1.0) / 2.0
    return data_01 * (original_win_max - original_win_min) + original_win_min

# --- Worker Function for Processing One Volume ---

def process_volume(volume_info, args, bone_mask_cmap, cmap_norm):
    """
    Loads data for one volume and processes/visualizes all its relevant slices.
    volume_info: tuple (base_filename_core, original_nii_filepath, bone_mask_filepath)
    args: Namespace object with command line arguments
    bone_mask_cmap: Pre-defined colormap for bone mask
    cmap_norm: Pre-defined normalization for the colormap
    Returns: tuple (processed_count, error_count) for this volume
    """
    base_filename_core, original_nii_filepath, bone_mask_filepath = volume_info
    processed_count_vol = 0
    error_count_vol = 0
    pid = os.getpid() # Get process ID for logging

    print(f"[PID {pid}] Processing volume: {base_filename_core}")

    try:
        # Load full volumes once
        nii_low = nib.load(original_nii_filepath)
        nii_bone_mask = nib.load(bone_mask_filepath)
        data_low = nii_low.get_fdata()
        data_bone_mask = nii_bone_mask.get_fdata()

        if data_low.shape != data_bone_mask.shape:
            print(f"[PID {pid}] Error: Shape mismatch for {base_filename_core}. Skipping.")
            return 0, 1 # Count as one volume error

        num_slices = data_low.shape[args.slice_axis]

        for slice_index in range(num_slices):
            # Construct Xp filename pattern (more specific)
            base_filename_xp = os.path.basename(original_nii_filepath).replace('.nii.gz', '') # e.g. dataset7_CLINIC_metal_0000_data
            xp_filename = f"{base_filename_xp}_slice{slice_index:04d}_p.npy"
            xp_filepath = os.path.join(args.input_xp_dir, xp_filename)

            if not os.path.exists(xp_filepath):
                 # This slice was likely skipped during generation (no metal) or doesn't exist
                 continue

            # --- Load Xp slice ---
            slice_xp_neg1_1 = load_npy_slice(xp_filepath)
            if slice_xp_neg1_1 is None:
                error_count_vol += 1
                continue

            # --- Extract slices from pre-loaded volumes ---
            slice_low_hu = load_nifti_slice(data_low, slice_index, args.slice_axis)
            slice_bone_mask = load_nifti_slice(data_bone_mask, slice_index, args.slice_axis)
            if slice_low_hu is None or slice_bone_mask is None:
                error_count_vol += 1
                continue

            # --- Generate Metal Mask (for overlay only) ---
            metal_mask_overlay = (slice_low_hu > args.metal_threshold).astype(float)
            # No need to skip here, we already know Xp exists, implying metal was present during generation

            # --- Process Data for Visualization ---
            slice_low_display = apply_window(slice_low_hu, args.wc, args.ww)
            slice_xp_hu_approx = denormalize_neg1_1_to_hu(slice_xp_neg1_1, args.xp_norm_min, args.xp_norm_max)
            slice_xp_display = apply_window(slice_xp_hu_approx, args.wc, args.ww)

            # --- Create Plot ---
            fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
            fig.suptitle(f"{base_filename_core} - Slice {slice_index} (Win: C={args.wc} W={args.ww})", fontsize=14, y=0.98)
            img_height, img_width = slice_low_display.shape
            extent = (0, img_width, img_height, 0)

            axes[0].imshow(slice_low_display, cmap='gray', vmin=0, vmax=1, extent=extent, aspect='auto')
            axes[0].set_xlabel("Original Input (X_low)", fontsize=12); axes[0].set_xticks([]); axes[0].set_yticks([]); [s.set_visible(False) for s in axes[0].spines.values()]

            axes[1].imshow(slice_xp_display, cmap='gray', vmin=0, vmax=1, extent=extent, aspect='auto')
            masked_overlay = np.ma.masked_where(metal_mask_overlay == 0, metal_mask_overlay)
            axes[1].imshow(masked_overlay, cmap='autumn', alpha=0.4, extent=extent, aspect='auto')
            axes[1].set_xlabel("ACDNet Output (X_p) + Metal Overlay", fontsize=12); axes[1].set_xticks([]); axes[1].set_yticks([]); [s.set_visible(False) for s in axes[1].spines.values()]

            im = axes[2].imshow(slice_bone_mask.astype(int), cmap=bone_mask_cmap, norm=cmap_norm, extent=extent, aspect='auto')
            axes[2].set_xlabel("Original Bone Mask", fontsize=12); axes[2].set_xticks([]); axes[2].set_yticks([]); [s.set_visible(False) for s in axes[2].spines.values()]

            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.93, wspace=0.1)

            # --- Save Plot ---
            output_png_filename = f"{base_filename_core}_slice{slice_index:04d}_comparison.png"
            output_png_filepath = os.path.join(args.output_png_dir, output_png_filename)
            try:
                plt.savefig(output_png_filepath, dpi=150)
                processed_count_vol += 1
            except Exception as e:
                print(f"[PID {pid}] Error saving plot {output_png_filepath}: {e}")
                error_count_vol += 1
            finally:
                plt.close(fig) # IMPORTANT: Close figure in worker process

    except Exception as e:
        print(f"[PID {pid}] Critical error processing volume {base_filename_core}: {e}")
        return processed_count_vol, error_count_vol + 1 # Count volume as error

    if processed_count_vol + error_count_vol > 0:
         print(f"[PID {pid}] Finished volume {base_filename_core}. Processed: {processed_count_vol}, Errors: {error_count_vol}")
    return processed_count_vol, error_count_vol


# --- Main Execution ---
if __name__ == "__main__":
    start_time_main = time.time()
    parser = argparse.ArgumentParser(description="Visualize All ACDNet Results (Parallel)")
    # --- (Keep all argument definitions as before) ---
    parser.add_argument("--input_xp_dir", type=str, required=True, help="Directory containing the processed ACDNet output slices (.npy files, X_p)")
    parser.add_argument("--input_low_dir", type=str, required=True, help="Directory containing the original metal-corrupted NIfTI files (X_low)")
    parser.add_argument("--input_bone_mask_dir", type=str, required=True, help="Directory containing the original bone segmentation mask NIfTI files (e.g., CLINIC_metal_XXXX_mask_4label.nii.gz)")
    parser.add_argument("--output_png_dir", type=str, required=True, help="Directory to save the output comparison PNG files")
    parser.add_argument("--slice_axis", type=int, default=2, help="Axis along which slices were extracted (default: 2 for axial)")
    parser.add_argument("--wc", type=int, default=400, help="Window Center for CT display")
    parser.add_argument("--ww", type=int, default=1800, help="Window Width for CT display")
    parser.add_argument("--metal_threshold", type=int, default=2500, help="HU threshold to generate metal mask *for overlay only*")
    parser.add_argument("--xp_norm_min", type=int, default=-1000, help="Min HU value used for original normalization before saving Xp")
    parser.add_argument("--xp_norm_max", type=int, default=1000, help="Max HU value used for original normalization before saving Xp")
    parser.add_argument("--num_mask_labels", type=int, default=4, help="Number of non-background labels in the bone mask (e.g., 4 for labels 1, 2, 3, 4)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel processes (default: CPU count - 1)")


    args = parser.parse_args()

    mkdir(args.output_png_dir)

    # --- Prepare list of volumes to process ---
    volume_info_list = []
    low_files = sorted(glob.glob(os.path.join(args.input_low_dir, '*.nii.gz')))
    for low_filepath in low_files:
        base_filename_xp = os.path.basename(low_filepath).replace('.nii.gz', '') # e.g. dataset7_CLINIC_metal_0000_data
        core_name = base_filename_xp.replace('dataset7_', '').replace('_data', '') # e.g. CLINIC_metal_0000
        bone_mask_filename = f"{core_name}_mask_4label.nii.gz"
        bone_mask_filepath = os.path.join(args.input_bone_mask_dir, bone_mask_filename)

        if os.path.exists(low_filepath) and os.path.exists(bone_mask_filepath):
            volume_info_list.append((core_name, low_filepath, bone_mask_filepath))
        else:
            print(f"Warning: Skipping volume {core_name} due to missing input/mask file.")

    if not volume_info_list:
        print("Error: No valid volumes found to process.")
        exit()

    print(f"Found {len(volume_info_list)} volumes to process.")

    # --- Setup Colormap (defined once) ---
    base_cmap_name = 'tab10'; base_cmap = plt.cm.get_cmap(base_cmap_name, args.num_mask_labels)
    colors = [(0, 0, 0, 1)] + [base_cmap(i) for i in range(args.num_mask_labels)]
    bone_mask_cmap = mcolors.ListedColormap(colors)
    cmap_bounds = np.arange(args.num_mask_labels + 2) - 0.5
    cmap_norm = mcolors.BoundaryNorm(cmap_bounds, bone_mask_cmap.N)

    # --- Setup Multiprocessing Pool ---
    if args.num_workers is None:
        pool_size = max(1, multiprocessing.cpu_count() - 1) # Default to N-1 cores
    else:
        pool_size = args.num_workers
    print(f"Starting parallel processing with {pool_size} workers...")

    # Use partial to pass fixed arguments to the worker function
    worker_partial = partial(process_volume, args=args, bone_mask_cmap=bone_mask_cmap, cmap_norm=cmap_norm)

    results = []
    with multiprocessing.Pool(processes=pool_size) as pool:
        # Use map to distribute work and collect results
        results = pool.map(worker_partial, volume_info_list)

    # --- Aggregate Results ---
    total_processed = sum(r[0] for r in results if r is not None)
    total_errors = sum(r[1] for r in results if r is not None)

    end_time_main = time.time()
    print("\n" + "="*30)
    print("Parallel Visualization Complete.")
    print(f"Successfully processed plots for {total_processed} slices.")
    if total_errors > 0:
        print(f"Encountered errors during processing for {total_errors} slices/volumes.")
    print(f"Output PNG files saved in: {args.output_png_dir}")
    print(f"Total execution time: {end_time_main - start_time_main:.2f} seconds")
    print("="*30)

