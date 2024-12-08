import os
import torch
import SimpleITK as sitk
import numpy as np
import time
from lungmask import LMInferer

start_time = time.time()

# Check if CUDA is available
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. Using GPU for inference.")
else:
    device = 'cpu'
    print("CUDA is not available. Using CPU for inference.")

def main(input_dir, model_name='R231'):
    print('input_dir: ', input_dir)
    count = 0
    
    # Initialize the lung mask inferer
    inferer = LMInferer(modelname=model_name)
    
    print('Segmentation started.')
    
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Process all .nii.gz files (not just those ending with _combined.nii.gz)
            if file.endswith('_combined.nii.gz'):
                try:
                    input_path = os.path.join(root, file)
                    
                    # Preserve the directory structure in the output
                    rel_path = os.path.relpath(root, input_dir)
                    output_dir = os.path.join('/projectdir/results/nov29/nifti/', rel_path)
                    
                    # Create output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create output filename
                    base_name = os.path.splitext(os.path.splitext(file)[0])[0]  # Remove both .nii and .gz
                    output_path = os.path.join(output_dir, f'{base_name}_LUNG.nii.gz')
                    
                    # Skip if output file already exists
                    if os.path.exists(output_path):
                        print(f'Skipping {file} - output already exists')
                        continue
                    
                    print(f'Processing: {input_path}')
                    
                    # Read the input image
                    input_image = sitk.ReadImage(input_path)
                    
                    # Apply the lung mask segmentation
                    segmentation = inferer.apply(input_image)
                    
                    # Convert the segmentation to a SimpleITK image if it's a NumPy array
                    if isinstance(segmentation, np.ndarray):
                        segmentation = sitk.GetImageFromArray(segmentation)
                        # Copy information from the input image
                        segmentation.CopyInformation(input_image)
                        
                        # Save the segmentation result
                        sitk.WriteImage(segmentation, output_path)
                        print(f'Saved mask to: {output_path}')
                        count += 1
                    else:
                        print(f"Error: Segmentation type wrong for file: {file}")
                
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    continue
    
    print(f'Segmentation finished. Total number of lung masks generated: {count}')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"Time spent: {hours} hours, {minutes} minutes, {seconds} seconds")

# Example usage
if __name__ == "__main__":
    input_directory = "/projectdir/"
    main(input_directory)