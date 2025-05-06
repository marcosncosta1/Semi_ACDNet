# check_nifti_size.py
import nibabel as nib
import argparse
import os

# Argument parser to get the file path
parser = argparse.ArgumentParser(description="Check NIfTI file dimensions")
parser.add_argument("--filepath", type=str, required=True, help="Path to the .nii.gz file")
args = parser.parse_args()

# Check if file exists
if not os.path.exists(args.filepath):
    print(f"Error: File not found at {args.filepath}")
else:
    try:
        # Load the NIfTI file
        nii_image = nib.load(args.filepath)

        # Get the data array
        data = nii_image.get_fdata()

        # Get the shape (dimensions)
        shape = data.shape

        print(f"File: {os.path.basename(args.filepath)}")
        print(f"Dimensions (Shape): {shape}")

        # Typically, the first two dimensions are height and width of slices
        if len(shape) >= 3:
            print(f"Slice dimensions (approx Height x Width): {shape[0]} x {shape[1]}")
            print(f"Number of slices (on axis 2): {shape[2]}")
        elif len(shape) == 2:
            print(f"Image dimensions (Height x Width): {shape[0]} x {shape[1]} (Is this a 2D file?)")

    except Exception as e:
        print(f"Error loading or reading file {args.filepath}: {e}")