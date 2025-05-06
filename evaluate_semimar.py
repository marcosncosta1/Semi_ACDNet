import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import nibabel as nib
from skimage.transform import resize
import argparse
import time
from datetime import timedelta



# Residual Block (Helper)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)

# Encoder (NetE)
class EncoderNet(nn.Module):
    def __init__(self, input_nc=1, ngf=32, n_blocks=4): # Defaults match training script
        super(EncoderNet, self).__init__()
        model = [
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True) ]
        in_features = ngf; out_features = ngf * 2
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        in_features = out_features; out_features = in_features * 2
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        self.initial_layers = nn.Sequential(*model)
        res_blocks = []
        for _ in range(n_blocks): res_blocks += [ResidualBlock(out_features)]
        self.res_blocks_seq = nn.Sequential(*res_blocks)
    def forward(self, input):
        initial_out = self.initial_layers(input); res_out = self.res_blocks_seq(initial_out); return res_out

# Decoder (NetD)
class DecoderNet(nn.Module):
    def __init__(self, output_nc=1, ngf=128, n_blocks=4): # Defaults match training script
        super(DecoderNet, self).__init__()
        res_blocks = [];
        for _ in range(n_blocks): res_blocks += [ResidualBlock(ngf)]
        self.res_blocks_seq = nn.Sequential(*res_blocks)
        model = []
        in_features = ngf; out_features = in_features // 2
        model += [
            nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        in_features = out_features; out_features = in_features // 2
        model += [
             nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        model += [
            nn.ReflectionPad2d(3), nn.Conv2d(out_features, output_nc, kernel_size=7, padding=0), nn.Tanh() ]
        self.upsample_layers = nn.Sequential(*model)
    def forward(self, input):
        res_out = self.res_blocks_seq(input); output = self.upsample_layers(res_out); return output

# Generator
class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf_enc=32, ngf_dec_in=128, n_blocks=4): # Defaults match training script
        super(Generator, self).__init__()
        self.encoder = EncoderNet(input_nc, ngf_enc, n_blocks)
        self.decoder = DecoderNet(output_nc, ngf_dec_in, n_blocks)
    def forward(self, x):
        # During inference, we only need the final image output, not the features
        features = self.encoder(x)
        output = self.decoder(features)
        return output # Return only the image
# --- END: Copy Model Definitions Here ---


# --- Utility Functions for Pre/Post-processing ---
def mkdir(path):
    if not os.path.exists(path): os.makedirs(path); print(f"--- Created output directory: {path} ---")

def preprocess_nifti_slice(slice_data, target_size, win_min, win_max):
    """Applies windowing, normalization [-1,1], and resizing to a NIfTI slice."""
    img_slice_windowed = np.clip(slice_data, win_min, win_max)
    if win_max > win_min: img_slice_normalized = 2.0 * (img_slice_windowed - win_min) / (win_max - win_min) - 1.0
    else: img_slice_normalized = np.zeros_like(img_slice_windowed)
    img_resized = resize(img_slice_normalized, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0) # Add channel dim
    return img_tensor

def save_output_slice(output_tensor_neg1_1, filename, save_format="npy"):
    """Saves the processed output slice (X_lh). Assumes input is [-1, 1]."""
    output_slice_neg1_1 = output_tensor_neg1_1.squeeze().cpu().numpy()
    if save_format == "npy":
        np.save(filename, output_slice_neg1_1)
    elif save_format == "png":
        output_scaled_uint8 = ((output_slice_neg1_1 + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        from PIL import Image # Import locally if only used here
        img = Image.fromarray(output_scaled_uint8)
        img.save(filename)
    else:
        print(f"Warning: Unknown save format '{save_format}'. Skipping save.")

def load_nifti_slice_raw(filepath, slice_index, axis=2): # Simplified loader for just getting slice
    """Loads a specific slice from a NIfTI file without rotation."""
    try:
        nii_img = nib.load(filepath)
        data = nii_img.get_fdata()
        if axis == 0: slice_data = data[slice_index, :, :]
        elif axis == 1: slice_data = data[:, slice_index, :]
        else: slice_data = data[:, :, slice_index]
        return slice_data
    except Exception as e: print(f"Error loading NIfTI {filepath} slice {slice_index}: {e}"); return None

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate Trained SemiMAR Model")
parser.add_argument("--generator_path", type=str, default="checkpoints_semimar/netG_epoch_60.pth", help="Path to trained SemiMAR generator weights")
parser.add_argument("--input_low_dir", type=str, required=True, help="Directory containing clinical X_low .nii.gz files to evaluate")
parser.add_argument("--output_lh_dir", type=str, required=True, help="Directory to save the generated corrected images (X_lh)")
parser.add_argument("--img_size", type=int, default=256, help="Target size for input/output slices (MUST MATCH TRAINING)")
parser.add_argument("--window_min", type=int, default=-1000, help="Minimum HU value for windowing")
parser.add_argument("--window_max", type=int, default=1000, help="Maximum HU value for windowing")
parser.add_argument("--slice_axis", type=int, default=2, help="Axis along which to extract 2D slices (default: 2 for axial)")
parser.add_argument("--save_format", type=str, default="npy", choices=["npy", "png"], help="Format to save X_lh slices (npy recommended)")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU id to use if available")

args = parser.parse_args()

# --- Main Execution ---
if __name__ == "__main__":
    start_time_main = time.time()

    # --- Setup Device ---
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {args.gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    # --- Create Output Directory ---
    mkdir(args.output_lh_dir)

    # --- Load Model ---
    print(f"Loading trained SemiMAR generator from: {args.generator_path}")
    # Ensure Generator class definition matches the one used for training
    # Use default parameters matching the training script
    modelG = Generator(ngf_enc=32, ngf_dec_in=128, n_blocks=4).to(device)
    try:
        modelG.load_state_dict(torch.load(args.generator_path, map_location=device))
        print("Generator loaded successfully.")
    except Exception as e:
        print(f"Error loading generator weights: {e}")
        exit()
    modelG.eval() # Set to evaluation mode

    # --- Find Input Files ---
    low_files = sorted(glob.glob(os.path.join(args.input_low_dir, '*.nii.gz')))
    if not low_files:
        print(f"Error: No input .nii.gz files found in {args.input_low_dir}")
        exit()
    print(f"Found {len(low_files)} input NIfTI volumes to process.")

    total_slices_processed = 0
    total_time = 0

    # --- Process Each Volume ---
    for i, low_filepath in enumerate(low_files):
        base_filename = os.path.basename(low_filepath).replace('.nii.gz', '')
        print(f"\nProcessing file {i+1}/{len(low_files)}: {base_filename}")

        try:
            nii_low = nib.load(low_filepath)
            data_low = nii_low.get_fdata()
            num_slices = data_low.shape[args.slice_axis]
            print(f"  Volume shape: {data_low.shape}, Slices: {num_slices}")

            vol_start_time = time.time()

            # --- Process Each Slice ---
            for slice_idx in range(num_slices):
                # Load raw slice data
                slice_low_hu_raw = load_nifti_slice_raw(low_filepath, slice_idx, args.slice_axis)
                if slice_low_hu_raw is None:
                    print(f"  Skipping slice {slice_idx}: Error loading raw data.")
                    continue

                # Preprocess the slice for the network
                input_tensor = preprocess_nifti_slice(
                    slice_low_hu_raw, args.img_size, args.window_min, args.window_max
                )
                input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dimension

                # --- Run Inference ---
                with torch.no_grad():
                    # The Generator forward pass returns (output_image, features)
                    # We only need the output_image for evaluation
                    output_tensor_neg1_1 = modelG(input_tensor)[0] # Get the image output only

                # --- Save Output Slice (X_lh) ---
                output_filename = f"{base_filename}_slice{slice_idx:04d}_lh.{args.save_format}"
                output_filepath = os.path.join(args.output_lh_dir, output_filename)
                save_output_slice(output_tensor_neg1_1.squeeze(0), output_filepath, args.save_format) # Remove batch dim before saving

                total_slices_processed += 1
                if (slice_idx + 1) % 100 == 0:
                    print(f"    Processed slice {slice_idx + 1}/{num_slices}")

            vol_end_time = time.time()
            print(f"  Finished volume in {vol_end_time - vol_start_time:.2f} seconds.")
            total_time += (vol_end_time - vol_start_time)

        except Exception as e:
            print(f"Error processing file {low_filepath}: {e}")

    # --- Final Summary ---
    end_time_main = time.time()
    print("\n" + "="*30)
    print("Evaluation Complete.")
    print(f"Total slices processed: {total_slices_processed}")
    if total_slices_processed > 0:
        avg_time_slice = total_time / total_slices_processed
        print(f"Average inference time per slice: {avg_time_slice:.4f} seconds")
    print(f"Generated X_lh files saved in: {args.output_lh_dir}")
    print(f"Total execution time: {timedelta(seconds=int(end_time_main - start_time_main))}")
    print("="*30)

