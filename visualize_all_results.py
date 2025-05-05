# --- (Keep imports and helper functions as before) ---
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For colormaps
import os
import argparse
import glob
import re # For parsing filenames

# --- (Keep utility functions: mkdir, load_nifti_slice, load_npy_slice, apply_window, denormalize_neg1_1_to_hu) ---
def mkdir(path):
    if not os.path.exists(path): os.makedirs(path); print(f"--- Created output directory: {path} ---")
def load_nifti_slice(filepath, slice_index, axis=2):
    try:
        nii_img = nib.load(filepath); data = nii_img.get_fdata()
        if axis == 0: slice_data = data[slice_index, :, :]
        elif axis == 1: slice_data = data[:, slice_index, :]
        else: slice_data = data[:, :, slice_index]
        return np.rot90(slice_data)
    except Exception as e: print(f"Error loading NIfTI {filepath} slice {slice_index}: {e}"); return None
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

# --- (Keep Argument Parsing as before) ---
parser = argparse.ArgumentParser(description="Visualize All ACDNet Clinical Slice Results with Bone Mask (v5 - Labels Below)")
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
args = parser.parse_args()

# --- (Keep Main Script Setup: mkdir, glob, pattern, counts, colormap definition) ---
mkdir(args.output_png_dir)
xp_files = sorted(glob.glob(os.path.join(args.input_xp_dir, '*_slice????_p.npy')))
if not xp_files: print(f"Error: No '*_slice????_p.npy' files found in {args.input_xp_dir}"); exit()
print(f"Found {len(xp_files)} processed slices to visualize.")
filename_pattern = re.compile(r"^(.*)_slice(\d{4})_p\.npy$")
processed_count = 0; error_count = 0
base_cmap_name = 'tab10'; base_cmap = plt.cm.get_cmap(base_cmap_name, args.num_mask_labels)
colors = [(0, 0, 0, 1)] + [base_cmap(i) for i in range(args.num_mask_labels)]
bone_mask_cmap = mcolors.ListedColormap(colors)
cmap_bounds = np.arange(args.num_mask_labels + 2) - 0.5
cmap_norm = mcolors.BoundaryNorm(cmap_bounds, bone_mask_cmap.N)

# --- Loop through files ---
for i, xp_filepath in enumerate(xp_files):
    if i % 50 == 0 or i == len(xp_files) - 1:
         print(f"\nProcessing file {i+1}/{len(xp_files)}: {os.path.basename(xp_filepath)}")

    # --- (Keep file parsing and data loading logic) ---
    match = filename_pattern.match(os.path.basename(xp_filepath))
    if not match: error_count += 1; continue
    base_filename_xp = match.group(1); slice_index = int(match.group(2))
    original_nii_filename = f"{base_filename_xp}.nii.gz"
    original_nii_filepath = os.path.join(args.input_low_dir, original_nii_filename)
    core_name = base_filename_xp.replace('dataset7_', '').replace('_data', '')
    bone_mask_filename = f"{core_name}_mask_4label.nii.gz"
    bone_mask_filepath = os.path.join(args.input_bone_mask_dir, bone_mask_filename)
    slice_xp_neg1_1 = load_npy_slice(xp_filepath)
    slice_low_hu = load_nifti_slice(original_nii_filepath, slice_index, args.slice_axis)
    slice_bone_mask = load_nifti_slice(bone_mask_filepath, slice_index, args.slice_axis)
    if slice_low_hu is None or slice_xp_neg1_1 is None or slice_bone_mask is None: error_count += 1; continue
    metal_mask_overlay = (slice_low_hu > args.metal_threshold).astype(float)
    if np.sum(metal_mask_overlay) == 0: continue
    slice_low_display = apply_window(slice_low_hu, args.wc, args.ww)
    slice_xp_hu_approx = denormalize_neg1_1_to_hu(slice_xp_neg1_1, args.xp_norm_min, args.xp_norm_max)
    slice_xp_display = apply_window(slice_xp_hu_approx, args.wc, args.ww)

    # --- Create Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(19, 7)) # Increased height slightly for labels
    # Set main title higher up
    fig.suptitle(f"{core_name} - Slice {slice_index} (Win: C={args.wc} W={args.ww})", fontsize=14, y=0.98)

    img_height, img_width = slice_low_display.shape
    extent = (0, img_width, img_height, 0)

    # --- Plot 1: Original Input (X_low) ---
    axes[0].imshow(slice_low_display, cmap='gray', vmin=0, vmax=1, extent=extent, aspect='auto')
    axes[0].set_xlabel("Original Input (X_low)", fontsize=12) # Label below
    axes[0].set_xticks([]) # Remove x-axis ticks
    axes[0].set_yticks([]) # Remove y-axis ticks
    axes[0].spines['top'].set_visible(False) # Remove frame lines if desired
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)


    # --- Plot 2: ACDNet Output (X_p) + Metal Mask Overlay ---
    axes[1].imshow(slice_xp_display, cmap='gray', vmin=0, vmax=1, extent=extent, aspect='auto')
    masked_overlay = np.ma.masked_where(metal_mask_overlay == 0, metal_mask_overlay)
    axes[1].imshow(masked_overlay, cmap='autumn', alpha=0.4, extent=extent, aspect='auto')
    axes[1].set_xlabel("ACDNet Output (X_p) + Metal Overlay", fontsize=12) # Label below
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)

    # --- Plot 3: Original Bone Segmentation Mask ---
    im = axes[2].imshow(slice_bone_mask.astype(int), cmap=bone_mask_cmap, norm=cmap_norm, extent=extent, aspect='auto')
    axes[2].set_xlabel("Original Bone Mask", fontsize=12) # Label below
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['bottom'].set_visible(False)
    axes[2].spines['left'].set_visible(False)

    # Adjust layout AFTER plotting everything - focus on bottom spacing
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.93, wspace=0.1) # Increased bottom margin, adjusted wspace

    # --- (Keep Saving Logic as before) ---
    output_png_filename = f"{core_name}_slice{slice_index:04d}_comparison.png"
    output_png_filepath = os.path.join(args.output_png_dir, output_png_filename)
    try:
        plt.savefig(output_png_filepath, dpi=150)
        processed_count += 1
    except Exception as e:
        print(f"  Error saving plot {output_png_filepath}: {e}")
        error_count += 1
    finally:
        plt.close(fig)

# --- (Keep Final Summary Print Statements) ---
print("\n" + "="*30)
print("Visualization Complete.")
print(f"Successfully processed and saved plots for {processed_count} slices.")
if error_count > 0:
    print(f"Encountered errors or skipped {error_count} slices.")
print(f"Output PNG files saved in: {args.output_png_dir}")
print("="*30)
