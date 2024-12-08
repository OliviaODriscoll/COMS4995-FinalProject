import torch
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from gan.generator import UnetGenerator

def normalize_hu(data):
    """Normalize HU values to [-1, 1] after clipping to [-1024, 700]"""
    data = np.clip(data, -1024, 700)
    return 2.0 * (data - (-1024)) / (700 - (-1024)) - 1.0

def denormalize_hu(data):
    """Convert normalized [-1, 1] values back to HU range [-1024, 700]"""
    return (data + 1.0) * (700 - (-1024)) / 2.0 + (-1024)

def window_image(image, window_center=50, window_width=350):
    """Apply windowing to the image for better visualization"""
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    windowed = np.clip(image, window_min, window_max)
    windowed = (windowed - window_min) / (window_max - window_min)
    return windowed

def save_comparison_png(input_img, generated_img, full_img, mask, save_path):
    """Save side-by-side comparison of input, generated, and target images with transposed orientation"""
    # Convert from HU to windowed visualization
    input_windowed = input_img.T #window_image(input_img).T  # Add transpose here
    generated_windowed = generated_img.T  # Add transpose here
    full_windowed = full_img.T #window_image(full_img).T  # Add transpose here
    
    # Also transpose the mask
    mask = mask.T
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot transposed images
    ax1.imshow(input_windowed, cmap='gray')
    ax1.set_title('Input (CAC)')
    ax1.axis('off')
    
    ax2.imshow(generated_windowed, cmap='gray')
    ax2.set_title('Generated')
    ax2.axis('off')
    
    ax3.imshow(full_windowed, cmap='gray')
    ax3.set_title('Ground Truth')
    ax3.axis('off')
    
    # Add mask overlay on generated image (also transposed)
    masked_region = np.ma.masked_where(mask == 0, generated_windowed)
    ax2.imshow(masked_region, cmap='gray', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.cac_files = sorted(list(self.root_dir.glob('*CAC_IMG.nii.gz')))
        self.full_files = []
        
        for cac_file in self.cac_files:
            base_name = str(cac_file).replace('CAC_IMG.nii.gz', 'IMG.nii.gz')
            full_file = Path(base_name)
            if full_file.exists():
                self.full_files.append(full_file)
            else:
                raise ValueError(f"Missing full scan for {cac_file}")

        print(f"Found {len(self.cac_files)} test image pairs")

    def __len__(self):
        return len(self.cac_files)

    def __getitem__(self, idx):
        cac_nii = nib.load(str(self.cac_files[idx]))
        full_nii = nib.load(str(self.full_files[idx]))
        
        # Get data and affine transform
        cac_data = cac_nii.get_fdata().astype(np.float32)
        full_data = full_nii.get_fdata().astype(np.float32)
        affine = cac_nii.affine
        
        # Create binary mask (1 where CAC is zero, 0 elsewhere)
        mask = (cac_data == 0).astype(np.float32)
        
        # Clip and normalize to [-1, 1]
        cac_data = normalize_hu(cac_data)
        full_data = normalize_hu(full_data)
        
        # Convert to tensors and add channel dimension
        cac_data = torch.FloatTensor(cac_data).unsqueeze(0)
        full_data = torch.FloatTensor(full_data).unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        if self.transform:
            cac_data = self.transform(cac_data)
            full_data = self.transform(full_data)
            mask = self.transform(mask)
        
        return {
            'cac': cac_data,
            'full': full_data,
            'mask': mask,
            'filename': self.cac_files[idx].stem,
            'affine': affine
        }

def main():
    parser = argparse.ArgumentParser(description='Test CAC Expansion Model')
    parser.add_argument('--weights', type=str, required=True, help='Path to generator weights')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save generated images')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different outputs
    nifti_dir = output_dir / 'nifti'
    png_dir = output_dir / 'png'
    nifti_dir.mkdir(exist_ok=True)
    png_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = UnetGenerator(in_channels=1, out_channels=1).to(device)
    generator.load_state_dict(torch.load(args.weights, map_location=device))
    generator.eval()
    
    transform = Resize((256, 256))
    test_dataset = TestDataset(args.data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    total_psnr = 0
    total_ssim = 0
    
    print("Starting testing...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            cac = batch['cac'].to(device)
            full = batch['full'].to(device)
            mask = batch['mask'].to(device)
            filenames = batch['filename']
            affines = batch['affine']
            
            # Generate images
            generated = generator(cac)
            
            # Apply mask to only fill in zero regions
            final = generated * mask + cac * (1 - mask)
            
            # Process each image in batch
            for i in range(len(filenames)):
                curr_gen = final[i].squeeze()
                curr_full = full[i].squeeze()
                curr_mask = mask[i].squeeze()
                curr_cac = cac[i].squeeze()
                
                # Convert back to HU values
                curr_gen_hu = denormalize_hu(curr_gen.cpu().numpy())
                curr_full_hu = denormalize_hu(curr_full.cpu().numpy())
                curr_cac_hu = denormalize_hu(curr_cac.cpu().numpy())
                curr_mask = curr_mask.cpu().numpy()
                
                # Save NIFTI
                gen_nii = nib.Nifti1Image(curr_gen_hu, affines[i])
                fname = filenames[i].split("_")
                nifti_path = nifti_dir / f"{fname[0]}_{fname[1]}_generated.nii.gz"
                nib.save(gen_nii, nifti_path)
                
                # Save PNG comparison
                png_path = png_dir / f"{fname[0]}_{fname[1]}_comparison.png"
                save_comparison_png(
                    curr_cac_hu,
                    curr_gen_hu,
                    curr_full_hu,
                    curr_mask,
                    png_path
                )
                
                # Calculate metrics on HU values
                psnr_val = psnr(curr_full_hu, curr_gen_hu, data_range=1724)  # 700-(-1024)
                ssim_val = ssim(curr_full_hu, curr_gen_hu, data_range=1724)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
    
    avg_psnr = total_psnr / len(test_dataset)
    avg_ssim = total_ssim / len(test_dataset)
    
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f'Average PSNR: {avg_psnr:.2f}\n')
        f.write(f'Average SSIM: {avg_ssim:.4f}\n')
    
    print(f'Testing completed!')
    print(f'Average PSNR: {avg_psnr:.2f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Generated images saved to: {output_dir}')
    print(f'NIFTI files in: {nifti_dir}')
    print(f'PNG comparisons in: {png_dir}')

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        if x.dim() == 3:  # [C,H,W]
            x = x.unsqueeze(0)
            x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)
            x = x.squeeze(0)
            return x
        return F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)

def calculate_metrics(pred, target, mask):
    """Calculate PSNR and SSIM for masked regions"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    mask = mask.cpu().numpy()
    
    # Only compare masked regions
    masked_pred = pred[mask > 0.5]
    masked_target = target[mask > 0.5]
    
    if len(masked_pred) == 0:
        return 0.0, 0.0
    
    # Convert back to HU values for metric calculation
    masked_pred = denormalize_hu(masked_pred)
    masked_target = denormalize_hu(masked_target)
    
    data_range = 700 - (-1024)  # HU value range
    
    try:
        psnr_val = psnr(masked_target, masked_pred, data_range=data_range)
    except:
        psnr_val = 0.0
        
    try:
        ssim_val = ssim(masked_target, masked_pred, data_range=data_range)
    except:
        ssim_val = 0.0
    
    return psnr_val, ssim_val

if __name__ == '__main__':
    main()