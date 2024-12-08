import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import argparse
from progress.bar import IncrementalBar
import nibabel as nib
from pathlib import Path
import numpy as np
from scipy import ndimage

from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.utils import Logger

class SurfaceDiceLoss(nn.Module):
    def __init__(self, threshold=-600, tolerance=1.0):
        super().__init__()
        self.threshold = threshold
        self.tolerance = tolerance
        # Define HU range constants
        self.min_hu = -1024
        self.max_hu = 700

    def denormalize_to_hu(self, x):
        """Convert from [-1,1] range to HU values and clip"""
        hu_values = x * (self.max_hu - self.min_hu) / 2 + (self.max_hu + self.min_hu) / 2
        return torch.clamp(hu_values, self.min_hu, self.max_hu)

    def compute_distances(self, pred_mask, gt_mask):
        """Compute surface distances using distance transform"""
        # Move to CPU and convert to numpy for distance transform
        pred_np = pred_mask.detach().cpu().numpy()
        gt_np = gt_mask.detach().cpu().numpy()
        
        # Convert to boolean arrays for morphological operations
        pred_np = pred_np.astype(bool)
        gt_np = gt_np.astype(bool)
        
        # Get boundaries using morphological operations
        struct = ndimage.generate_binary_structure(2, 2)
        pred_eroded = ndimage.binary_erosion(pred_np, struct)
        gt_eroded = ndimage.binary_erosion(gt_np, struct)
        
        # Get boundaries
        pred_boundary = np.logical_xor(pred_np, pred_eroded)
        gt_boundary = np.logical_xor(gt_np, gt_eroded)
        
        # Compute distance transforms
        pred_dist = ndimage.distance_transform_edt(~pred_boundary)
        gt_dist = ndimage.distance_transform_edt(~gt_boundary)
        
        # Convert back to torch tensors
        pred_dist = torch.from_numpy(pred_dist).to(pred_mask.device).float()
        gt_dist = torch.from_numpy(gt_dist).to(gt_mask.device).float()
        
        return pred_dist, gt_dist

    def forward(self, pred, target):
        """
        Compute Surface Dice loss
        Args:
            pred: predicted image (B, 1, H, W) in [-1,1] range
            target: target image (B, 1, H, W) in [-1,1] range
        """
        # Convert normalized values to HU range
        pred_hu = self.denormalize_to_hu(pred)
        target_hu = self.denormalize_to_hu(target)
        
        # Threshold the volumes
        pred_mask = (pred_hu > self.threshold).float()
        target_mask = (target_hu > self.threshold).float()
        
        batch_size = pred.shape[0]
        device = pred.device
        total_surface_dice = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            for c in range(pred.shape[1]):
                # Compute surface distances for current sample and channel
                pred_dist, gt_dist = self.compute_distances(
                    pred_mask[i, c], target_mask[i, c]
                )
                
                # Find surface elements within tolerance
                pred_surface = (pred_dist <= self.tolerance).float()
                gt_surface = (gt_dist <= self.tolerance).float()
                
                # Calculate surface dice
                intersection = (pred_surface * gt_surface).sum()
                denominator = pred_surface.sum() + gt_surface.sum()
                
                # Avoid division by zero
                if denominator > 0:
                    surface_dice = (2.0 * intersection) / denominator
                else:
                    surface_dice = torch.tensor(1.0, device=device)
                
                total_surface_dice += surface_dice
        
        # Average over batch and channels
        avg_surface_dice = total_surface_dice / (batch_size * pred.shape[1])
        return torch.tensor(1.0 - avg_surface_dice, device=device, requires_grad=True)



class NiftiSliceDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir (str): Path to the main directory containing train/val/test folders
            transform (callable, optional): Optional transform to be applied
            mode (str): 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir) / mode
        self.transform = transform
        self.mode = mode
        
        # Get all CAC images and their corresponding full scans
        self.cac_files = sorted(list(self.root_dir.glob('*CAC_IMG.nii.gz')))
        self.full_files = []
        
        # Match each CAC file with its corresponding full scan
        for cac_file in self.cac_files:
            base_name = str(cac_file).replace('CAC_IMG.nii.gz', 'IMG.nii.gz')
            full_file = Path(base_name)
            if full_file.exists():
                self.full_files.append(full_file)
            else:
                raise ValueError(f"Missing full scan for {cac_file}")

        print(f"Found {len(self.cac_files)} image pairs in {mode} set")

    def __len__(self):
        return len(self.cac_files)

    def __getitem__(self, idx):
        # Load the NIFTI files
        cac_nii = nib.load(str(self.cac_files[idx]))
        full_nii = nib.load(str(self.full_files[idx]))
        
        # Get the data as 2D arrays
        cac_data = cac_nii.get_fdata().astype(np.float32)
        full_data = full_nii.get_fdata().astype(np.float32)
        
        # Create binary mask (1 where CAC is zero, 0 elsewhere)
        mask = (cac_data == 0).astype(np.float32)
        
        # Normalize to [-1, 1]
        cac_data = (cac_data - cac_data.min()) / (cac_data.max() - cac_data.min() + 1e-8) * 2 - 1
        full_data = (full_data - full_data.min()) / (full_data.max() - full_data.min() + 1e-8) * 2 - 1
        
        # Convert to tensors and add channel dimension
        cac_data = torch.FloatTensor(cac_data).unsqueeze(0)
        full_data = torch.FloatTensor(full_data).unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        if self.transform:
            cac_data = self.transform(cac_data)
            full_data = self.transform(full_data)
            mask = self.transform(mask)
            
        return cac_data, full_data, mask

class Resize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.resizer = nn.Upsample(size=size, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # Handle 2D input
        if x.dim() == 3:  # [C,H,W]
            x = x.unsqueeze(0)  # Add batch dimension
            x = self.resizer(x)
            x = x.squeeze(0)  # Remove batch dimension
            return x
        return self.resizer(x)

class MaskedOutput(nn.Module):
    """Module to combine generator output with input image using mask"""
    def forward(self, generator_output, input_image, mask):
        return generator_output * mask + input_image * (1 - mask)

class GANTrainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, device, 
                 lambda_l1=100, lambda_surface_dice=50, label_smoothing=0.1):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.lambda_l1 = lambda_l1
        self.lambda_surface_dice = lambda_surface_dice
        self.label_smoothing = label_smoothing
        self.masked_output = MaskedOutput().to(device)
        self.surface_dice_loss = SurfaceDiceLoss().to(device)
        
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            d_optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )

    def train_step(self, cac, full, mask):
        should_train_discriminator = torch.rand(1).item() < 0.5
        metrics = {}
        
        if should_train_discriminator:
            self.discriminator.zero_grad()
            
            with torch.no_grad():
                fake_image = self.generator(cac)
                fake = self.masked_output(fake_image, cac, mask)
            
            real_pred = self.discriminator(full, cac)
            real_target = torch.ones_like(real_pred) * (1 - self.label_smoothing)
            real_loss = F.binary_cross_entropy_with_logits(real_pred, real_target)
            
            fake_pred = self.discriminator(fake.detach(), cac)
            fake_target = torch.zeros_like(fake_pred)
            fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_target)
            
            d_loss = (real_loss + fake_loss) * 0.5
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.d_optimizer.step()
            
            metrics['d_loss'] = d_loss.item()
        else:
            metrics['d_loss'] = 0.0
        
        # Update generator
        for _ in range(2 if not should_train_discriminator else 1):
            self.generator.zero_grad()
            
            fake_image = self.generator(cac)
            fake = self.masked_output(fake_image, cac, mask)
            fake_pred = self.discriminator(fake, cac)
            
            g_target = torch.ones_like(fake_pred) * (1 - self.label_smoothing)
            g_loss = F.binary_cross_entropy_with_logits(fake_pred, g_target)
            
            # # Print value ranges for debugging
            # with torch.no_grad():
            #     print(f"\nValue ranges:")
            #     print(f"Normalized - Fake: [{fake.min().item():.2f}, {fake.max().item():.2f}]")
            #     print(f"Normalized - Real: [{full.min().item():.2f}, {full.max().item():.2f}]")
                
            #     # Convert to HU values for display
            #     fake_hu = self.surface_dice_loss.denormalize_to_hu(fake)
            #     full_hu = self.surface_dice_loss.denormalize_to_hu(full)
            #     print(f"HU values - Fake: [{fake_hu.min().item():.1f}, {fake_hu.max().item():.1f}]")
            #     print(f"HU values - Real: [{full_hu.min().item():.1f}, {full_hu.max().item():.1f}]")
            #     print(f"Threshold: {self.surface_dice_loss.threshold} HU")
            
            # Calculate losses
            l1_loss = F.l1_loss(fake, full) * self.lambda_l1
            surface_dice_loss = self.surface_dice_loss(fake, full) * self.lambda_surface_dice
            
            total_g_loss = g_loss + l1_loss + surface_dice_loss
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()
            
            metrics.update({
                'g_loss': g_loss.item(),
                'l1_loss': l1_loss.item(),
                'surface_dice_loss': surface_dice_loss.item(),
                'total_g_loss': total_g_loss.item()
            })
        
        return metrics
    
    def validate(self, val_dataloader):
        self.generator.eval()
        self.discriminator.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for cac, full, mask in val_dataloader:
                cac = cac.to(self.device)
                full = full.to(self.device)
                mask = mask.to(self.device)
                
                fake_image = self.generator(cac)
                fake = self.masked_output(fake_image, cac, mask)
                fake_pred = self.discriminator(fake, cac)
                
                g_target = torch.ones_like(fake_pred)
                g_loss = F.binary_cross_entropy_with_logits(fake_pred, g_target)
                l1_loss = F.l1_loss(fake, full) * self.lambda_l1
                surface_dice_loss = self.surface_dice_loss(fake, full) * self.lambda_surface_dice
                
                val_loss = g_loss + l1_loss + surface_dice_loss
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        self.g_scheduler.step(avg_val_loss)
        self.d_scheduler.step(avg_val_loss)
        
        return avg_val_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Pix2Pix for CAC Expansion')
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of the batches")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data directory")
    parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
    parser.add_argument("--d_lr", type=float, default=0.00005, help="Discriminator learning rate")
    parser.add_argument("--lambda_l1", type=float, default=100, help="L1 loss weight")
    parser.add_argument("--lambda_surface_dice", type=float, default=50, help="Surface Dice loss weight")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")

    args = parser.parse_args()

    # Set up device
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Define transforms
    transforms = Resize((256, 256))

    # Initialize models
    print('Defining models!')
    generator = UnetGenerator(in_channels=1, out_channels=1).to(device)
    discriminator = ConditionalDiscriminator(in_channels=1).to(device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Initialize optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    # Initialize trainer
    trainer = GANTrainer(generator, discriminator, g_optimizer, d_optimizer, device, 
                        lambda_l1=100, lambda_surface_dice=50, label_smoothing=0.1)

    # Load datasets
    print('Loading datasets!')
    train_dataset = NiftiSliceDataset(args.data_root, transform=transforms, mode='train')
    val_dataset = NiftiSliceDataset(args.data_root, transform=transforms, mode='val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4)

    print('Start of training process!')
    logger = Logger(filename="cac_expansion_2d")

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        epoch_metrics = {
            'train_g_loss': 0.,
            'train_d_loss': 0.,
            'train_l1_loss': 0.,
            'train_surface_dice_loss': 0.,
            'train_total_loss': 0.
        }
        
        start = time.time()
        bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(train_dataloader))
        
        for cac, full, mask in train_dataloader:
            cac = cac.to(device)
            full = full.to(device)
            mask = mask.to(device)
            
            batch_metrics = trainer.train_step(cac, full, mask)
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                if f'train_{k}' in epoch_metrics:
                    epoch_metrics[f'train_{k}'] += v
            
            bar.next()
        bar.finish()
        
        # Average the metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(train_dataloader)
        
        # Validation
        val_loss = trainer.validate(val_dataloader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.save_weights(generator.state_dict(), 'generator_best')
            logger.save_weights(discriminator.state_dict(), 'discriminator_best')
        
        # Regular checkpoint save
        if (epoch + 1) % 10 == 0:
            logger.save_weights(generator.state_dict(), f'generator_epoch_{epoch+1}')
            logger.save_weights(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}')
        
        end = time.time()
        
        # Log metrics
        for k, v in epoch_metrics.items():
            logger.add_scalar(k, v, epoch+1)
        logger.add_scalar('val_loss', val_loss, epoch+1)
        
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"[G loss: {epoch_metrics['train_g_loss']:.3f}] "
              f"[D loss: {epoch_metrics['train_d_loss']:.3f}] "
              f"[Surface Dice loss: {epoch_metrics['train_surface_dice_loss']:.3f}] "
              f"[Val loss: {val_loss:.3f}] "
              f"ETA: {end-start:.3f}s")

    logger.close()
    print('End of training process!')

if __name__ == '__main__':
    main()