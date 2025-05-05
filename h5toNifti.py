import h5py
import numpy as np
import nibabel as nib
import os

input_folder = "/your/full/path/to/train_640geo"  # <-- update this
output_folder = "/home/mcosta/PycharmProjects/Semi_ACDNet/Nifti_output/nifti_slices_ma_CT"              # change to something like "nifti_slices_LI_CT" if needed
key_to_extract = "ma_CT"                          # or 'LI_CT', 'metal_trace', etc.

os.makedirs(output_folder, exist_ok=True)

for root, _, files in os.walk(input_folder):
    for filename in sorted(files):
        if filename.endswith(".h5") and filename != "gt.h5":
            full_path = os.path.join(root, filename)
            try:
                with h5py.File(full_path, 'r') as f:
                    if key_to_extract in f:
                        data = f[key_to_extract][()]
                        data = data.astype(np.float32)
                        # Optional: normalize to 0–255
                        norm_data = (data - data.min()) / (data.max() - data.min()) * 255.0
                        norm_data = np.expand_dims(norm_data, axis=0)  # make it (1, H, W)
                        nifti = nib.Nifti1Image(norm_data, affine=np.eye(4))
                        # Save using a unique name based on path
                        rel_path = os.path.relpath(full_path, input_folder).replace('/', '_').replace('.h5', '')
                        out_path = os.path.join(output_folder, f"{rel_path}_{key_to_extract}.nii.gz")
                        nib.save(nifti, out_path)
                        print(f"✅ Saved {out_path}")
                    else:
                        print(f"⚠️ Key '{key_to_extract}' not found in {full_path}")
            except Exception as e:
                print(f"❌ Error reading {full_path}: {e}")
