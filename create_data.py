import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
import shutil

def clean_directory(output_dir):
    """Remove all .nii.gz files from directory"""
    print(f"Cleaning directory: {output_dir}")
    nii_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    for file in nii_files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    print(f"Removed {len(nii_files)} .nii.gz files")

def find_lung_slices(lung_mask):
    """Find all slices with lung tissue present and return every third one"""
    slice_sums = lung_mask.sum((0, 2))  # Sum across height and width
    lung_slices = np.where(slice_sums > 0)[0]  # Get indices where lungs are present
    return lung_slices[::3]  # Take every third slice

def process_files_from_list(input_dir, lung_dir, output_dir, coord_file, files_list):
    """Process specific files from files.txt"""
    # Load coordinates data
    df_coords = pd.read_csv(coord_file, low_memory=False)
    
    # Read files list
    with open(files_list, 'r') as f:
        files = f.read().splitlines()
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    for filename in files:
        try:
            idno = filename.split('_')[0]  # Extract ID from filename
            print(f"\nProcessing {filename} (ID: {idno})")
            
            # Load image and lung mask
            img_path = os.path.join(input_dir, f"{idno}_IMG.nii.gz")
            lung_path = os.path.join(lung_dir, f"{idno}_LUNG.nii.gz")
            print(img_path)
            
            if not os.path.exists(lung_path):
                print(f"No lung mask found for {idno}")
                continue
                
            img = nib.load(img_path)
            lung_mask = nib.load(lung_path)
            
            img_data = img.get_fdata()
            lung_data = lung_mask.get_fdata()
            
            # Get coordinates for this ID
            df_idno = df_coords[df_coords.idno == int(idno)]
            
            if df_idno.empty:
                print(f"No coordinates found for ID {idno}")
                continue
            
            # Get trachea and bifurcation coordinates
            trachea_rows = df_idno[df_idno.anatomicalname.str.upper() == "TRACHEA"]
            bifurcation_rows = df_idno[df_idno.anatomicalname.str.upper().isin(["LMB", "RMB"])]
            
            if trachea_rows.empty or bifurcation_rows.empty:
                print(f"Missing anatomical landmarks for {idno}")
                continue
            
            # Calculate crop points
            upper_slice = int((trachea_rows.parent_loc_x.iloc[0] + trachea_rows.loc_x.iloc[0])/2)
            lower_slice = int(bifurcation_rows.loc_x.min())
            
            if upper_slice >= lower_slice:
                print(f"Invalid slice positions for {idno}")
                continue
            
            # Get slices with lung tissue
            lung_slices = find_lung_slices(lung_data)
            
            # Process each third slice
            for slice_idx, curr_slice in enumerate(lung_slices):
                # Extract the current slice
                slice_img = img_data[:, curr_slice, :]
                slice_mask = lung_data[:, curr_slice, :]
                
                # Create cropped versions
                cropped_img = slice_img.copy()
                cropped_img[:, :upper_slice] = 0
                cropped_img[:, lower_slice:] = 0
                
                cropped_mask = slice_mask.copy()
                cropped_mask[:, :upper_slice] = 0
                cropped_mask[:, lower_slice:] = 0
                
                # Save all versions with slice number in filename
                slice_suffix = f"_slice{slice_idx:03d}"
                
                # Save original image slice
                nib.save(
                    nib.Nifti1Image(slice_img, img.affine),
                    os.path.join(output_dir, f"{idno}{slice_suffix}_IMG.nii.gz")
                )
                
                # Save synthetic CAC image
                nib.save(
                    nib.Nifti1Image(cropped_img, img.affine),
                    os.path.join(output_dir, f"{idno}{slice_suffix}_CAC_IMG.nii.gz")
                )
                
                # Save original lung mask slice
                nib.save(
                    nib.Nifti1Image(slice_mask, img.affine),
                    os.path.join(output_dir, f"{idno}{slice_suffix}_LUNG.nii.gz")
                )
                
                # Save cropped lung mask
                nib.save(
                    nib.Nifti1Image(cropped_mask, img.affine),
                    os.path.join(output_dir, f"{idno}{slice_suffix}_CAC_LUNG.nii.gz")
                )
            
            print(f"Successfully processed {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

def main():
    # Define paths
    base_dir = "/projectdir//mesa_synthetic_CAC/"
    input_dir = "/dataset//IMG"
    lung_dir = "/dataset//LUNG"
    coord_file = "/DATA/NetworkShare/ductal_morphology/df_with_coords_2761_post_qc.csv"
    
    # Create output directory for slices
    output_base = "/projectdir//mesa_synthetic_CAC_slices/"
    os.makedirs(output_base, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        # Setup directories
        input_split_dir = os.path.join(base_dir, split)
        output_split_dir = os.path.join(output_base, split)
        os.makedirs(output_split_dir, exist_ok=True)
        
        files_list = os.path.join(input_split_dir, 'files.txt')
        
        if not os.path.exists(files_list):
            print(f"No files.txt found in {input_split_dir}")
            continue
            
        print(f"\nProcessing {split} split...")
        
        # Clean directory
        clean_directory(output_split_dir)
        
        # Process files
        process_files_from_list(input_dir, lung_dir, output_split_dir, coord_file, files_list)

if __name__ == "__main__":
    main()