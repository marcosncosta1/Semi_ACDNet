import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import argparse


def load_nifti_slice(filepath, slice_index, axis=2):
    """Loads a specific slice from a NIfTI file."""
    try:
        nii_img = nib.load(filepath)
        data = nii_img.get_fdata()

        # Extract slice based on axis
        if axis == 0:
            slice_data = data[slice_index, :, :]
        elif axis == 1:
            slice_data = data[:, slice_index, :]
        else:  # Default axis 2
            slice_data = data[:, :, slice_index]
        return np.rot90(slice_data)  # Rotate for standard viewing orientation
    except FileNotFoundError:
        print(f"Error: NIfTI file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading NIfTI file {filepath}: {e}")
        return None


def load_npy_slice(filepath):
    """Loads a slice from a .npy file."""
    try:
        slice_data = np.load(filepath)
        # --- ADD THIS LINE ---
        # Apply the same rotation as applied to the NIfTI slice
        return np.rot90(slice_data)
        # --- END OF ADDED LINE ---
    except FileNotFoundError:
        print(f"Error: NumPy file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading NumPy file {filepath}: {e}")
        return None


def apply_window(image_data, window_center, window_width):
    """Applies CT windowing (center/width) and scales to [0, 1]."""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2

    windowed_image = np.clip(image_data, min_val, max_val)

    # Scale to [0, 1]
    if max_val > min_val:
        scaled_image = (windowed_image - min_val) / (max_val - min_val)
    else:
        scaled_image = np.zeros_like(windowed_image)  # Avoid division by zero

    return scaled_image


def denormalize_neg1_1_to_hu(normalized_data, original_win_min=-1000, original_win_max=1000):
    """Converts data normalized to [-1, 1] back towards original HU range."""
    # Scale [-1, 1] back to [0, 1]
    data_01 = (normalized_data + 1.0) / 2.0
    # Scale [0, 1] back to [win_min, win_max]
    hu_data = data_01 * (original_win_max - original_win_min) + original_win_min
    return hu_data


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Visualize ACDNet Clinical Slice Results")
parser.add_argument("--input_low_nii", type=str, required=True,
                    help="Path to the original metal-corrupted NIfTI file (X_low)")
parser.add_argument("--input_xp_npy", type=str, required=True,
                    help="Path to the corresponding processed ACDNet output slice (.npy file, X_p)")
parser.add_argument("--input_mask_nii", type=str, default=None,
                    help="(Optional) Path to the corresponding metal mask NIfTI file")
parser.add_argument("--slice_index", type=int, required=True, help="Index of the 2D slice to visualize")
parser.add_argument("--slice_axis", type=int, default=2,
                    help="Axis along which the slice was extracted (default: 2 for axial)")
parser.add_argument("--wc", type=int, default=40, help="Window Center (e.g., 40 for soft tissue, 400 for bone)")
parser.add_argument("--ww", type=int, default=400, help="Window Width (e.g., 400 for soft tissue, 1800 for bone)")
parser.add_argument("--save_png", type=str, default=None,
                    help="(Optional) Path to save the comparison plot as a PNG file")
# Add arguments for the original normalization range used when creating Xp if different from default
parser.add_argument("--xp_norm_min", type=int, default=-1000,
                    help="Min HU value used for original normalization before saving Xp")
parser.add_argument("--xp_norm_max", type=int, default=1000,
                    help="Max HU value used for original normalization before saving Xp")

args = parser.parse_args()

# --- Load Data ---
print(f"Loading slice {args.slice_index} from {args.input_low_nii}...")
slice_low_hu = load_nifti_slice(args.input_low_nii, args.slice_index, args.slice_axis)

print(f"Loading processed slice from {args.input_xp_npy}...")
# This loads the data normalized to [-1, 1]
slice_xp_neg1_1 = load_npy_slice(args.input_xp_npy)

slice_mask = None
if args.input_mask_nii:
    print(f"Loading mask slice {args.slice_index} from {args.input_mask_nii}...")
    # Load mask and ensure it's binary
    slice_mask_raw = load_nifti_slice(args.input_mask_nii, args.slice_index, args.slice_axis)
    if slice_mask_raw is not None:
        slice_mask = (slice_mask_raw > 0).astype(float)  # Make binary (0 or 1)

# --- Check if loading failed ---
if slice_low_hu is None or slice_xp_neg1_1 is None:
    print("Failed to load necessary image data. Exiting.")
    exit()

# --- Process Data for Visualization ---

# 1. Apply windowing to the original HU data (X_low)
slice_low_display = apply_window(slice_low_hu, args.wc, args.ww)

# 2. Denormalize Xp data back towards HU range, then apply display windowing
#    (Assumes Xp was saved normalized from xp_norm_min/max HU range to [-1, 1])
slice_xp_hu_approx = denormalize_neg1_1_to_hu(slice_xp_neg1_1, args.xp_norm_min, args.xp_norm_max)
slice_xp_display = apply_window(slice_xp_hu_approx, args.wc, args.ww)

# --- Create Plot ---
num_plots = 2
if slice_mask is not None:
    num_plots = 3  # Add plot for mask or overlay

fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
fig.suptitle(f"Slice {args.slice_index} Comparison (Window: C={args.wc} W={args.ww})", fontsize=16)

# Plot 1: Original Input (X_low)
axes[0].imshow(slice_low_display, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Original Input (X_low)")
axes[0].axis('off')

# Plot 2: ACDNet Output (X_p)
axes[1].imshow(slice_xp_display, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("ACDNet Output (X_p)")
axes[1].axis('off')

# Plot 3: Mask (Optional)
if slice_mask is not None:
    axes[2].imshow(slice_mask, cmap='viridis')  # Use a different colormap for mask
    axes[2].set_title("Original Bone Mask (4 Labels)")
    axes[2].axis('off')
    # --- OR Overlay Mask on ACDNet Output (Alternative) ---
    # axes[1].imshow(slice_xp_display, cmap='gray', vmin=0, vmax=1)
    # # Create a masked array where mask is 0
    # masked_overlay = np.ma.masked_where(slice_mask == 0, slice_mask)
    # axes[1].imshow(masked_overlay, cmap='autumn', alpha=0.4) # Overlay with transparency
    # axes[1].set_title("ACDNet Output + Mask")
    # axes[1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

# --- Save or Show Plot ---
if args.save_png:
    print(f"Saving plot to {args.save_png}")
    plt.savefig(args.save_png, dpi=300)
else:
    print("Displaying plot...")
    plt.show()

print("Done.")

'''

**How to Use:**

1.  **Save:** Save the code above as a Python file (e.g., `visualize_result.py`).
2.  **Install Libraries:** Make sure you have `matplotlib`, `nibabel`, and `numpy` in your environment (`pip install matplotlib nibabel numpy`).
3.  **Find Files:**
    * Identify the full path to an original `X_low` NIfTI file (e.g., `data/test/CTPelvic1K_METAL/dataset7_CLINIC_metal_0007_data.nii.gz`).
    * Identify the full path to the corresponding `X_p` `.npy` file you generated for a specific slice (e.g., `/home/mcosta/PycharmProjects/Semi_ACDNet/clinic_metal_results/dataset7_CLINIC_metal_0007_data_slice0150_p.npy`).
    * (Optional) Identify the full path to the corresponding mask NIfTI file (e.g., `data/test/CTPelvic1K_dataset7_MASK/CLINIC_metal_0007_mask_4label.nii.gz`).
4.  **Run from Terminal:**
   bash
    conda activate acdnet_113_env # Or your working env

    python visualize_result.py \
        --input_low_nii data/test/CTPelvic1K_METAL/dataset7_CLINIC_metal_0007_data.nii.gz \
        --input_xp_npy /home/mcosta/PycharmProjects/Semi_ACDNet/clinic_metal_results/dataset7_CLINIC_metal_0007_data_slice0150_p.npy \
        --input_mask_nii data/test/CTPelvic1K_dataset7_MASK/CLINIC_metal_0007_mask_4label.nii.gz \
        --slice_index 150 \
        --wc 40 \
        --ww 400 \
        --save_png ./visualization_slice_150.png
        # Remove --save_png to display interactively instead
        # Adjust --xp_norm_min/max if you used different values than -1000/1000 when generating Xp
    
    * Adjust file paths, slice index, window center (`--wc`), and window width (`--ww`) as needed. Common windows: Soft Tissue (C:40, W:400), Lung (C:-600, W:1500), Bone (C:400, W:1800).

This script will generate a side-by-side comparison image similar to the panels shown in Figure 7.

**2. Reproducing Figure 8 (Segmentation Visualization)**

Figure 8 requires segmenting the 3D pelvic structures from the artifact-reduced volume. As mentioned in the ACDNet paper's supplementary material (Section C), this involves a separate segmentation model.

**Steps:**

1.  **Train/Obtain Segmentation Model:** You need a U-Net or similar model trained specifically for pelvic bone segmentation (Sacrum, Left Hip, Right Hip, Lumbar Spine - matching the CLINIC-metal annotations) using *clean* CT data (like the CLINIC dataset subset mentioned in the paper). Training such a model is a significant task itself. If the authors of ACDNet or CTPelvic1K provide a pre-trained segmentation model, you could use that.
2.  **Reconstruct 3D Volume:** Combine the individual 2D `X_p` slices (`.npy` files) you generated for a *single patient volume* back into a 3D NumPy array. You'll need to:
    * Identify all `_sliceXXXX_p.npy` files belonging to one patient volume (e.g., `dataset7_CLINIC_metal_0007_data_slice*_p.npy`).
    * Load them in the correct slice order.
    * Stack them into a 3D NumPy array.
    * Optionally save this reconstructed volume as a new NIfTI file using `nibabel`, making sure to copy the affine transformation matrix from the original `X_low` NIfTI file so the geometry is correct.
3.  **Preprocess for Segmentation:** Adapt the reconstructed 3D volume's intensity range, normalization, and orientation to match exactly what the pre-trained segmentation model expects as input.
4.  **Run Segmentation:** Feed the preprocessed 3D volume into the segmentation model to obtain a 3D label map (where different integers represent different bones or background).
5.  **Visualize 3D Segmentation:** Use specialized medical image software or libraries:
    * **Software:** ITK-SNAP (free), 3D Slicer (free) are excellent for loading the reconstructed volume and the segmentation mask NIfTI files and creating 3D renderings.
    * **Libraries (Python):** Libraries like `vedo` or `matplotlib` (with its limited 3D capabilities) can be used to script the generation of 3D surface renderings from the segmentation mask, similar to Figure 8. This requires more coding effort (e.g., using marching cubes algorithm to create meshes from the label map).

**Code Snippet for Reconstructing 3D Volume:**
'''
#python

import numpy as np
import nibabel as nib
import glob
import os
import re  # For sorting slices numerically


def reconstruct_volume(slice_dir, base_filename_prefix, num_slices, original_nii_path, output_nii_path):
    """Reconstructs a 3D volume from 2D .npy slices."""

    all_slices = []
    print(f"Looking for slices matching: {base_filename_prefix}_slice*_p.npy in {slice_dir}")

    # Find and sort slice files numerically
    slice_files = glob.glob(os.path.join(slice_dir, f"{base_filename_prefix}_slice????_p.npy"))

    # Define a function to extract the slice number for sorting
    def get_slice_num(filepath):
        match = re.search(r'_slice(\d{4})_p\.npy$', os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    slice_files.sort(key=get_slice_num)  # Sort based on slice number

    if len(slice_files) == 0:
        print("Error: No slice files found.")
        return None

    print(f"Found {len(slice_files)} slice files. Expected ~{num_slices} based on original.")

    for i, slice_file in enumerate(slice_files):
        slice_num = get_slice_num(slice_file)
        if slice_num != i:
            print(f"Warning: Missing or out-of-order slice? Expected index {i}, found {slice_num} in {slice_file}")
            # Handle missing slices appropriately (e.g., fill with zeros or skip volume)
            # For now, we'll attempt to continue but the volume might be incomplete/incorrect

        slice_data = np.load(slice_file)
        all_slices.append(slice_data)

    if not all_slices:
        print("Error: No slices loaded.")
        return None

    # Stack slices along the correct axis (usually axis 2)
    # Ensure all slices have the same shape before stacking
    first_shape = all_slices[0].shape
    if not all(s.shape == first_shape for s in all_slices):
        print("Error: Not all loaded slices have the same shape.")
        # Implement logic to handle shape mismatches if necessary (e.g., padding/cropping)
        return None

    volume_data = np.stack(all_slices, axis=2)
    print(f"Reconstructed volume shape: {volume_data.shape}")

    # Load original NIfTI to get affine transformation
    try:
        original_nii = nib.load(original_nii_path)
        affine = original_nii.affine
        header = original_nii.header  # Copy header too if desired
    except Exception as e:
        print(f"Warning: Could not load original NIfTI {original_nii_path} for affine. Using identity. Error: {e}")
        affine = np.eye(4)
        header = None

    # Create and save the new NIfTI file
    reconstructed_nii = nib.Nifti1Image(volume_data, affine, header=header)
    nib.save(reconstructed_nii, output_nii_path)
    print(f"Reconstructed volume saved to {output_nii_path}")
    return output_nii_path


# --- Example Usage ---
# slice_directory = "/home/mcosta/PycharmProjects/Semi_ACDNet/clinic_metal_results"
# file_prefix = "dataset7_CLINIC_metal_0007_data"
# original_volume_path = "data/test/CTPelvic1K_METAL/dataset7_CLINIC_metal_0007_data.nii.gz"
# # You might need to get the expected number of slices from the original volume info
# expected_slices = 350 # Example for case 0007
# output_volume_path = "/home/mcosta/PycharmProjects/Semi_ACDNet/reconstructed_volumes/case_0007_reconstructed.nii.gz"

# reconstructed_file = reconstruct_volume(slice_directory, file_prefix, expected_slices, original_volume_path, output_volume_path)
# if reconstructed_file:
#     # Now you can load reconstructed_file and feed it to the segmentation model
#     pass

# Focus on getting the Figure 7 visualization working first, as it directly uses the output you already have.
# Figure 8 requires the additional step of obtaining and running the segmentation mod
