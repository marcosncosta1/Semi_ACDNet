import os
import argparse
import time
import numpy as np
import torch
from PIL import Image
import nibabel as nib

from acdnet import ACDNet  # Ensure your ACDNet is defined in acdnet.py

###############################################################################
# Argument Parser
###############################################################################
parser = argparse.ArgumentParser(description="ACDNet Test on NIfTI Data - Process Only Metal Slices")
# Essential paths
parser.add_argument("--model_dir", type=str, default="models/ACDNet_latest.pt", help="Path to model weights")
parser.add_argument("--data_path", type=str, default="data/test/CTPelvic1K_METAL", help="Folder with .nii.gz volumes")
parser.add_argument("--save_path", type=str, default="save_results_nii/", help="Folder to save outputs")

# Model configuration - match exactly with training parameters
parser.add_argument("--batchSize", type=int, default=32, help="Testing batch size")
parser.add_argument("--patchSize", type=int, default=64, help="Height/width of input image to network")
parser.add_argument("--N", type=int, default=6, help="Number of feature maps")
parser.add_argument("--Np", type=int, default=32, help="Number of channel concatenations")
parser.add_argument("--d", type=int, default=32, help="Number of convolutional filters in common dictionary D")
parser.add_argument("--num_res", type=int, default=3, help="Number of Resblocks in each ResNet")
parser.add_argument("--T", type=int, default=10, help="Stage number T")
parser.add_argument("--Mtau", type=float, default=0.5, help="Sparse feature map threshold")
parser.add_argument("--etaM", type=float, default=1.0, help="Stepsize for updating M")
parser.add_argument("--etaX", type=float, default=5.0, help="Stepsize for updating X")

# Hardware settings
parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU or not")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")

# Additional parameters needed for model initialization
parser.add_argument("--niter", type=int, default=300, help="Total number of training epochs")
parser.add_argument("--milestone", type=int, default=[50,100,150,200], nargs='+', help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=0.0002, help="Initial learning rate")
parser.add_argument("--Xl2", type=float, default=1, help="Loss weights")
parser.add_argument("--Xl1", type=float, default=5e-4, help="Loss weights")
parser.add_argument("--Al1", type=float, default=5e-4, help="Loss weights")

# Metal threshold in Hounsfield Units
parser.add_argument("--metal_threshold", type=float, default=2500.0, help="HU threshold for metal detection")

opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

###############################################################################
# Utility Functions
###############################################################################
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder:", path)

def load_nii_as_array(nii_path):
    """
    Loads a .nii.gz file and returns a NumPy array of shape (H, W, D).
    """
    img_nifti = nib.load(nii_path)
    data_3d = img_nifti.get_fdata()
    return data_3d.astype(np.float32)

def normalize_to_255(slice_data, hu_min=-1024, hu_max=3072):
    """
    Clamps slice_data into [hu_min, hu_max] and maps it to [0, 255].
    Adjust hu_min/hu_max to match your training data.
    """
    slice_data = np.clip(slice_data, hu_min, hu_max)
    slice_data = (slice_data - hu_min) / (hu_max - hu_min)  # Now in [0,1]
    slice_data *= 255.0  # Scale to [0,255]
    return slice_data

###############################################################################
# Main Testing Function
###############################################################################
def main():
    mkdir(opt.save_path)
    print("Loading ACDNet model from:", opt.model_dir)
    model = ACDNet(opt)
    
    try:
        checkpoint = torch.load(opt.model_dir, map_location="cpu")
        
        # Initialize model parameters first
        model_dict = model.state_dict()
        
        # Process checkpoint tensors
        for key in model_dict:
            if key in checkpoint:
                # Create new parameter tensor
                param_new = torch.nn.Parameter(
                    checkpoint[key].clone().detach().contiguous(),
                    requires_grad=model_dict[key].requires_grad
                )
                
                # Directly set the parameter
                if '.' in key:
                    # For nested parameters (e.g., proxNet_M_T.0.tau)
                    module_path, param_name = key.rsplit('.', 1)
                    module = model
                    for part in module_path.split('.'):
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    setattr(module, param_name, param_new)
                else:
                    # For top-level parameters
                    setattr(model, key, param_new)
        
        print("Successfully loaded checkpoint")
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise

    if opt.use_gpu:
        model.cuda()
    model.eval()

    # Gather all .nii.gz files in the data_path
    nii_files = [f for f in os.listdir(opt.data_path) if f.endswith(".nii.gz")]
    print("Found {} NIfTI volumes.".format(len(nii_files)))

    for file_name in nii_files:
        volume_path = os.path.join(opt.data_path, file_name)
        volume_3d = load_nii_as_array(volume_path)  # shape (H, W, D)
        H, W, D = volume_3d.shape
        print(f"\nProcessing volume: {file_name} | shape: {H} x {W} x {D}")

        # Prepare an empty array for the reconstructed volume
        reconstructed_3d = np.zeros_like(volume_3d, dtype=np.float32)

        for slice_idx in range(D):
            slice_data = volume_3d[..., slice_idx]  # shape (H, W)

            # Check if this slice contains metal by thresholding HU values
            metal_mask = slice_data > opt.metal_threshold
            if not metal_mask.any():
                # No metal present; copy the original slice
                reconstructed_3d[..., slice_idx] = slice_data
                continue

            # Create the non-metal mask (1 for non-metal regions, 0 for metal)
            non_metal_mask = ~metal_mask
            mask_tensor = torch.from_numpy(non_metal_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

            # Normalize to [0,255] for network input
            slice_255 = normalize_to_255(slice_data, hu_min=-1024, hu_max=3072)

            # Convert to torch tensor with shape [1, 1, H, W]
            slice_tensor = torch.from_numpy(slice_255).float().unsqueeze(0).unsqueeze(0)
            
            if opt.use_gpu:
                slice_tensor = slice_tensor.cuda()
                mask_tensor = mask_tensor.cuda()

            # Run the model with proper mask
            with torch.no_grad():
                start_time = time.time()
                _, ListX, _, _, _ = model(slice_tensor, slice_tensor, mask_tensor)
                end_time = time.time()

            dur_time = end_time - start_time
            print(f"Volume {file_name}, slice {slice_idx:03d} processed in {dur_time:.4f} s")

            # Retrieve the final reconstructed slice; ensure values are in [0,255]
            Xout = ListX[-1].squeeze().detach().cpu().numpy()
            Xout = np.clip(Xout, 0, 255)

            # Optionally, convert back to HU using the inverse of normalize_to_255.
            # For simplicity, here we map Xout back to the original HU range.
            out_hu = (Xout / 255.0) * (3072 - (-1024)) + (-1024)
            reconstructed_3d[..., slice_idx] = out_hu

        # Save the reconstructed volume as a new NIfTI file.
        out_nii_path = os.path.join(opt.save_path, f"reconstructed_{file_name}")
        original_affine = nib.load(volume_path).affine
        out_nifti = nib.Nifti1Image(reconstructed_3d, original_affine)
        nib.save(out_nifti, out_nii_path)
        print(f"Saved reconstructed volume to: {out_nii_path}")

if __name__ == "__main__":
    main()
