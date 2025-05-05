import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
import nibabel as nib


def to_grayscale(image):
    """
    Convert an image to grayscale if it has more than one channel.
    For CT scans, taking the first channel is sufficient.
    """
    if image.ndim >= 3 and image.shape[-1] > 1:
        return image[..., 0]
    return image


def dynamic_win_size(image, default=7):
    """
    Compute an appropriate window size for SSIM based on spatial dimensions.
    Assumes that the input image is grayscale.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    if min_dim < 3:
        return None
    win = min(default, min_dim)
    if win % 2 == 0:
        win -= 1
    return win


def load_nifti(file_path):
    """Load a NIfTI file and return its data as numpy array"""
    return nib.load(file_path).get_fdata().astype(np.float32)


def evaluate_nifti_volumes(
        original_dir,
        reconstructed_dir,
        output_file="acdnet_nifti_metrics.txt",
        metal_threshold=2500.0
):
    """
    Evaluate PSNR and SSIM for original and reconstructed NIfTI volumes.
    Only evaluates slices that contained metal in the original volume.
    """
    # Get list of original NIfTI files
    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.nii.gz')])
    
    psnr_vals, ssim_vals = [], []
    table_rows = []
    table_header = (
        f"{'Volume/Slice':<40}{'PSNR':>10}{'SSIM':>10}\n"
        + "-" * 60
    )

    for orig_file in original_files:
        print(f"\nProcessing {orig_file}")
        
        # Find corresponding reconstructed file
        recon_file = f"reconstructed_{orig_file}"
        recon_path = os.path.join(reconstructed_dir, recon_file)
        
        if not os.path.exists(recon_path):
            print(f"Skipping {orig_file} - missing reconstruction")
            continue

        # Load volumes
        orig_vol = load_nifti(os.path.join(original_dir, orig_file))
        recon_vol = load_nifti(recon_path)

        # Process each slice
        for slice_idx in range(orig_vol.shape[2]):
            orig_slice = orig_vol[..., slice_idx]
            recon_slice = recon_vol[..., slice_idx]

            # Check if slice contains metal
            if not (orig_slice > metal_threshold).any():
                continue

            # Compute metrics
            # Use the same HU range as in test_metal.py
            data_range = 3072 - (-1024)  # HU range
            
            psnr_val = psnr(orig_slice, recon_slice, data_range=data_range)
            
            win_size = min(7, min(orig_slice.shape))
            if win_size % 2 == 0:
                win_size -= 1
                
            ssim_val = ssim(orig_slice, recon_slice,
                           data_range=data_range,
                           win_size=win_size,
                           channel_axis=None)

            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)

            # Fixed formatting
            slice_identifier = f"{orig_file}[{slice_idx:03d}]"
            row = "{:<40}{:>10.2f}{:>10.4f}".format(
                slice_identifier,
                psnr_val,
                ssim_val
            )
            table_rows.append(row)

            print(f"Slice {slice_idx:03d} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

    # Generate summary statistics
    summary = (
        "\nSummary (mean ± std)\n"
        f"PSNR: {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f}\n"
        f"SSIM: {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}\n"
        f"\nTotal metal-containing slices evaluated: {len(psnr_vals)}"
    )

    # Create full report
    full_report = (
        f"ACDNet NIfTI Volume Quality Evaluation\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Original Directory: {os.path.abspath(original_dir)}\n"
        f"Reconstructed Directory: {os.path.abspath(reconstructed_dir)}\n"
        f"Metal Threshold: {metal_threshold} HU\n\n"
        + table_header + "\n"
        + "\n".join(table_rows)
        + "\n" + summary
    )

    print(full_report)
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\nReport saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ACDNet NIfTI reconstruction quality metrics")
    parser.add_argument("--original_dir", type=str, default="data/test/CTPelvic1K_METAL",
                       help="Directory containing original NIfTI volumes")
    parser.add_argument("--reconstructed_dir", type=str, default="save_results_nii",
                       help="Directory containing reconstructed NIfTI volumes")
    parser.add_argument("--output_file", type=str, default="acdnet_nifti_metrics.txt",
                       help="Output filename for the report")
    parser.add_argument("--metal_threshold", type=float, default=2500.0,
                       help="HU threshold for metal detection")
    args = parser.parse_args()

    evaluate_nifti_volumes(
        args.original_dir,
        args.reconstructed_dir,
        args.output_file,
        args.metal_threshold
    )
