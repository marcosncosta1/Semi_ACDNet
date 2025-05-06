import os
import os.path
import argparse
import numpy as np
import torch
import time
import nibabel as nib # To load .nii.gz files
from skimage.transform import resize # For resizing slices
from skimage.filters import gaussian # For blurring XLI alternative
from acdnet import ACDNet # Assuming acdnet.py is in the same directory or path
import glob
from PIL import Image # Only needed if saving as PNG

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="ACDNet_Generate_Xp")
# Model loading arguments
parser.add_argument("--model_dir", type=str, default="models/ACDNet_latest.pt", help='path to trained ACDNet model file')
parser.add_argument('--N', type=int, default=6, help='the number of feature maps')
parser.add_argument('--Np', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--d', type=int, default=32, help='the number of convolutional filters in common dictionary D')
parser.add_argument('--num_res', type=int, default=3, help='Resblocks number in each ResNet')
parser.add_argument('--T', type=int, default=10, help='Stage number T')
parser.add_argument('--Mtau', type=float, default=1.5, help='for sparse feature map')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='stepsize for updating X')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size for inference (usually 1)')

# Data path arguments
parser.add_argument("--input_low_dir", type=str, default='data/test/CTPelvic1K_METAL' ,required=True, help='Directory containing X_low .nii.gz files (e.g., CLINIC-metal)')
parser.add_argument("--input_mask_dir", type=str, default='data/test/CTPelvic1K_dataset7_MASK', required=True, help='Directory containing corresponding metal mask .nii.gz files')
parser.add_argument("--output_xp_dir", type=str, default='/home/mcosta/PycharmProjects/Semi_ACDNet/clinic_metal_results', required=True, help='Directory to save generated X_p files')
# Processing arguments
parser.add_argument("--img_size", type=int, default=416, help='Target size for input slices (MATCH ACDNet TRAINING)')
parser.add_argument("--window_min", type=int, default=-1000, help='Minimum HU value for windowing (MATCH ACDNet TRAINING)')
parser.add_argument("--window_max", type=int, default=1000, help='Maximum HU value for windowing (MATCH ACDNet TRAINING)')
parser.add_argument("--slice_axis", type=int, default=2, help='Axis along which to extract 2D slices (usually 2 for axial)')
parser.add_argument("--save_format", type=str, default="npy", choices=["npy", "png"], help='Format to save X_p slices (npy recommended)')
parser.add_argument("--xli_mode", type=str, default="blur", choices=["copy", "blur"], help='Method to generate XLI input: "copy" (XLI=Xma) or "blur" (XLI=blurred Xma)')
parser.add_argument("--blur_sigma", type=float, default=1.5, help='Sigma for Gaussian blur if xli_mode is "blur"')
# GPU arguments
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')

opt = parser.parse_args()

# --- GPU Setup ---
if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {opt.gpu_id}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# --- Utility Functions ---
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"--- Created output directory: {path} ---")
    else:
        print(f"--- Output directory exists: {path} ---")

def preprocess_slice(slice_data, mask_slice, target_size, win_min, win_max):
    """Preprocesses a single 2D slice and its mask.
       Outputs tensors scaled to [0, 255] as expected by ACDNet training.
    """
    # 1. Apply Windowing to image slice
    img_slice_windowed = np.clip(slice_data, win_min, win_max)

    # 2. Normalize image slice to [0, 1]
    if win_max > win_min:
        img_slice_normalized_01 = (img_slice_windowed - win_min) / (win_max - win_min)
    else: # Handle constant image case
        img_slice_normalized_01 = np.zeros_like(img_slice_windowed)

    # 3. Scale image slice to [0, 255]
    img_slice_scaled = img_slice_normalized_01 * 255.0

    # 4. Normalize mask slice to [0, 1]
    mask_normalized = np.clip(mask_slice, 0, 1).astype(np.float32)

    # 5. Resize both to target size
    img_resized = resize(img_slice_scaled, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    mask_resized = resize(mask_normalized, (target_size, target_size), order=0, anti_aliasing=False, preserve_range=True)
    mask_resized = np.clip(mask_resized, 0, 1)

    # 6. Convert to PyTorch Tensor -> [1, 1, H, W]
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0)

    return img_tensor, mask_tensor

# No changes needed for generate_xli function

def save_output_slice(output_tensor_0_255, filename, save_format="npy"):
    """Saves the processed output slice (Xp).
       Normalizes the [0, 255] output from ACDNet to [-1, 1] for SemiMAR.
    """
    # Remove batch and channel dimensions, move to CPU, convert to NumPy
    output_slice_0_255 = output_tensor_0_255.squeeze().cpu().numpy() # Result is [H, W] in [0, 255] range

    # Normalize from [0, 255] to [-1, 1]
    output_slice_neg1_1 = (output_slice_0_255 / 255.0) * 2.0 - 1.0
    output_slice_neg1_1 = np.clip(output_slice_neg1_1, -1.0, 1.0) # Ensure range


    # Save the [-1, 1] normalized data
    if save_format == "npy":
        np.save(filename, output_slice_neg1_1) # Saves array in [-1, 1] range
    elif save_format == "png":
        # Scale [-1, 1] to [0, 255] for PNG saving (loses precision!)
        output_scaled_uint8 = ((output_slice_neg1_1 + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(output_scaled_uint8)
        img.save(filename)
    else:
        print(f"Warning: Unknown save format '{save_format}'. Skipping save.")

def generate_xli(xma_tensor, mode="copy", sigma=1.5):
    """Generates the XLI tensor based on the chosen mode."""
    if mode == "copy":
        return xma_tensor.clone()
    elif mode == "blur":
        # Apply Gaussian blur (on CPU with NumPy for simplicity)
        xma_np = xma_tensor.squeeze().cpu().numpy()
        xli_blurred_np = gaussian(xma_np, sigma=sigma) # Use channel_axis=None for 2D
        xli_tensor = torch.from_numpy(xli_blurred_np).float().unsqueeze(0).unsqueeze(0).to(xma_tensor.device)
        return xli_tensor
    else:
        raise ValueError("Invalid xli_mode specified")

def save_output_slice(output_tensor_0_255, filename, save_format="npy"):
    """Saves the processed output slice (Xp).
       Normalizes the [0, 255] output from ACDNet to [-1, 1] for SemiMAR.
    """
    # Remove batch and channel dimensions, move to CPU, convert to NumPy
    output_slice_0_255 = output_tensor_0_255.squeeze().cpu().numpy() # Result is [H, W] in [0, 255] range

    # --- DEBUG PRINT 1 ---
    print(f"    DEBUG save_output_slice: Input tensor range: min={np.min(output_slice_0_255):.4f}, max={np.max(output_slice_0_255):.4f}")
    # --- END DEBUG PRINT 1 ---

    # Normalize from [0, 255] to [-1, 1]
    output_slice_neg1_1 = (output_slice_0_255 / 255.0) * 2.0 - 1.0
    output_slice_neg1_1 = np.clip(output_slice_neg1_1, -1.0, 1.0) # Ensure range

    # --- DEBUG PRINT 2 ---
    print(f"    DEBUG save_output_slice: Normalized tensor range: min={np.min(output_slice_neg1_1):.4f}, max={np.max(output_slice_neg1_1):.4f}")
    # --- END DEBUG PRINT 2 ---

    # Save the [-1, 1] normalized data
    if save_format == "npy":
        # --- DEBUG PRINT 3 ---
        print(f"    DEBUG save_output_slice: Saving as .npy: {filename}")
        # --- END DEBUG PRINT 3 ---
        np.save(filename, output_slice_neg1_1) # Saves array in [-1, 1] range
    elif save_format == "png":
        # Scale [-1, 1] to [0, 255] for PNG saving (loses precision!)
        output_scaled_uint8 = ((output_slice_neg1_1 + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(output_scaled_uint8)
         # --- DEBUG PRINT 3 ---
        print(f"    DEBUG save_output_slice: Saving as .png: {filename}")
        # --- END DEBUG PRINT 3 ---
        img.save(filename)
    else:
        print(f"Warning: Unknown save format '{save_format}'. Skipping save.")


# --- Main Function ---
def main():
    mkdir(opt.output_xp_dir)

    print('Loading ACDNet model ...')
    model = ACDNet(opt).to(device)
    try:
        model.load_state_dict(torch.load(opt.model_dir, map_location=device))
        print(f"Model loaded successfully from {opt.model_dir}")
    except Exception as e:
        print(f"Error loading model from {opt.model_dir}: {e}")
        return
    model.eval()

    low_files = sorted(glob.glob(os.path.join(opt.input_low_dir, '*.nii.gz')))
    mask_files = sorted(glob.glob(os.path.join(opt.input_mask_dir, '*.nii.gz'))) # Keep for consistency check

    print(f"Found {len(low_files)} input NIfTI files.")
    if len(low_files) == 0:
        print("Error: No input .nii.gz files found in", opt.input_low_dir)
        return

    total_slices_processed = 0
    total_time = 0

    for i, low_filepath in enumerate(low_files):
        base_filename = os.path.basename(low_filepath).replace('.nii.gz', '')
        # --- Construct the expected mask filename ---
        base_low_filename = os.path.basename(low_filepath)  # e.g., dataset7_CLINIC_metal_0000_data.nii.gz
        # Remove the prefix and suffix from the input filename
        core_name = base_low_filename.replace('dataset7_', '').replace('_data.nii.gz', '')  # e.g., CLINIC_metal_0000
        # Construct the mask filename using the new pattern
        mask_filename = f"{core_name}_mask_4label.nii.gz"  # e.g., CLINIC_metal_0000_mask_4label.nii.gz
        # Get the full path to the expected mask file
        mask_filepath = os.path.join(opt.input_mask_dir, mask_filename)


        if not os.path.exists(mask_filepath):
             print(f"Warning: Mask file not found for {low_filepath}. Skipping.")
             continue

        print(f"\nProcessing file {i+1}/{len(low_files)}: {base_filename}")

        try:
            nii_low = nib.load(low_filepath)
            nii_mask = nib.load(mask_filepath)
            data_low = nii_low.get_fdata()
            data_mask = nii_mask.get_fdata()

            if data_low.shape != data_mask.shape:
                print(f"Warning: Shape mismatch between {low_filepath} {data_low.shape} and {mask_filepath} {data_mask.shape}. Skipping.")
                continue

            num_slices = data_low.shape[opt.slice_axis]
            print(f"  Volume shape: {data_low.shape}, Slices: {num_slices}")

            for slice_idx in range(num_slices):
                if opt.slice_axis == 0:
                    slice_low = data_low[slice_idx, :, :]
                    slice_mask = data_mask[slice_idx, :, :]
                elif opt.slice_axis == 1:
                    slice_low = data_low[:, slice_idx, :]
                    slice_mask = data_mask[:, slice_idx, :]
                else:
                    slice_low = data_low[:, :, slice_idx]
                    slice_mask = data_mask[:, :, slice_idx]

                # Check if mask slice contains any metal before processing
                if np.sum(slice_mask) == 0:
                    # print(f"    Slice {slice_idx + 1}/{num_slices}: No metal found in mask. Skipping.")
                    continue # Skip slices without metal

                Xma_tensor, M_tensor = preprocess_slice(
                    slice_low, slice_mask, opt.img_size, opt.window_min, opt.window_max
                )
                Xma_tensor = Xma_tensor.to(device)
                M_tensor = M_tensor.to(device)

                # Generate XLI based on selected mode
                XLI_tensor = generate_xli(Xma_tensor, mode=opt.xli_mode, sigma=opt.blur_sigma)

                with torch.no_grad():
                    start_time = time.time()
                    X0, ListX, ListA, ListX_nonK, ListA_nonK = model(Xma_tensor, XLI_tensor, M_tensor)
                    end_time = time.time()

                # --- IMPORTANT: Verify the output range of ListX[-1] ---
                # If ListX[-1] is not in [-1, 1], add normalization here.
                # Example: output_tensor = (ListX[-1] - min_val) / (max_val - min_val) * 2 - 1
                output_tensor = ListX[-1] # Assuming output is already ~[-1, 1]

                # Save Output (Xp)
                output_filename_base = f"{base_filename}_slice{slice_idx:04d}_p"
                output_filepath = os.path.join(opt.output_xp_dir, f"{output_filename_base}.{opt.save_format}")
                save_output_slice(output_tensor, output_filepath, opt.save_format)

                total_slices_processed += 1
                total_time += (end_time - start_time)

                if (total_slices_processed) % 100 == 0: # Print progress periodically
                     print(f"    Processed {total_slices_processed} slices with metal...")

        except Exception as e:
            print(f"Error processing file {low_filepath}: {e}")

    print("\n" + "="*30)
    print("Processing Complete.")
    print(f"Total slices processed (with metal): {total_slices_processed}")
    if total_slices_processed > 0:
        avg_time = total_time / total_slices_processed
        print(f"Average inference time per slice: {avg_time:.4f} seconds")
    print(f"Generated Xp files saved in: {opt.output_xp_dir}")
    print("="*30)

if __name__ == "__main__":
    main()
