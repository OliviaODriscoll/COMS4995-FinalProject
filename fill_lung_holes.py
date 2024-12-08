import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from glob import glob
import os

def process_lung_mask(input_path, output_path):
    """
    Process a binary lung segmentation mask by filling holes and smoothing boundaries.
    
    Parameters:
    input_path (str): Path to input .nii.gz file
    output_path (str): Path to save processed .nii.gz file
    
    Returns:
    None
    """
    # Load the NIFTI file
    nifti_img = nib.load(input_path)
    mask_data = nifti_img.get_fdata()
    
    # Convert to binary mask if not already
    binary_mask = (mask_data > 0).astype(np.float32)
    
    # Fill holes in each 2D slice
    processed_mask = np.zeros_like(binary_mask)
    for z in range(binary_mask.shape[2]):
        # Fill holes in 2D slice
        slice_filled = ndimage.binary_fill_holes(binary_mask[:,:,z])
        processed_mask[:,:,z] = slice_filled
    
    # Optional: Fill 3D holes as well
    processed_mask = ndimage.binary_fill_holes(processed_mask)
    
    # Smooth the boundaries
    # First perform binary closing to smooth edges
    struct_element = morphology.ball(2)  # Adjust radius as needed
    processed_mask = morphology.binary_closing(processed_mask, struct_element)
    
    # Then apply Gaussian smoothing and threshold back to binary
    processed_mask = ndimage.gaussian_filter(processed_mask.astype(float), sigma=0.5)
    processed_mask = (processed_mask > 0.5).astype(np.float32)
    
    # Create new NIFTI image with same affine and header as input
    new_img = nib.Nifti1Image(processed_mask, nifti_img.affine, nifti_img.header)
    
    # Save the processed mask
    nib.save(new_img, output_path)
    
    return processed_mask

# Example usage
if __name__ == "__main__":
    for lung in glob(os.path.join("/DATA/process_OHO/COMS4995/pix2pix-main/results/nov29/nifti/COMS4995/pix2pix-main/results/nifti/*_combined_LUNG.nii.gz")):
        processed_mask = process_lung_mask(lung, lung.replace("combined_", ""))