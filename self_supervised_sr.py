#!/usr/bin/env python3
"""
Self-Supervised Super-Resolution for 16-bit Grayscale DICOM Images
Multiple approaches: Cycle Consistency, Contrastive Learning, Degradation Modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pydicom
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Tuple, Optional, Dict, List
import random

class SelfSupervisedSRDataset(Dataset):
    """Dataset for self-supervised super-resolution training."""
    
    def __init__(self, image_path: str, patch_size: int = 64, num_patches: int = 1000):
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Load DICOM image
        if image_path.endswith('.dcm'):
            self.image, self.original_range, self.dicom_data = self.load_dicom(image_path)
        else:
            self.image = self.load_image(image_path)
            self.original_range = (0, 65535)
            self.dicom_data = None
        
        # Normalize to [-1, 1]
        self.image = 2.0 * (self.image - self.image.min()) / (self.image.max() - self.image.min()) - 1.0
        
        # Generate patch coordinates
        self.patch_coords = self.generate_patch_coordinates()
    
    def load_dicom(self, path: str) -> Tuple[np.ndarray, Tuple[float, float], pydicom.Dataset]:
        """Load DICOM image."""
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array.astype(np.float32)
        
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            image = image * dicom.RescaleSlope + dicom.RescaleIntercept
        
        return image, (float(image.min()), float(image.max())), dicom
    
    def load_image(self, path: str) -> np.ndarray:
        """Load regular image."""
        import cv2
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image.astype(np.float32)
    
    def generate_patch_coordinates(self) -> List[Tuple[int, int]]:
        """Generate random patch coordinates."""
        h, w = self.image.shape
        coords = []
        
        for _ in range(self.num_patches):
            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)
            coords.append((y, x))
        
        return coords
    
    def __len__(self):
        return self.num_patches
    
    def __getitem__(self, idx):
        y, x = self.patch_coords[idx]
        patch = self.image[y:y+self.patch_size, x:x+self.patch_size]
        
        # Convert to tensor
        patch = torch.from_numpy(patch).float().unsqueeze(0)  # (1, H, W)
        
        return patch

class ResidualBlock(nn.Module):
    """Residual block for SR networks."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)

class SRGenerator(nn.Module):
    """Generator network for super-resolution."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 num_residual_blocks: int = 16, base_channels: int = 64):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, 9, padding=4)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )
        
        # Post-residual convolution
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        
        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.conv3 = nn.Conv2d(base_channels, out_channels, 9, padding=4)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = out + residual
        
        out = self.upsampling(out)
        out = self.conv3(out)
        
        return out

class SRDiscriminator(nn.Module):
    """Discriminator network for adversarial training."""
    
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        self.features = nn.Sequential(
            # 64
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels * 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        validity = self.classifier(features)
        return validity

class DegradationModel(nn.Module):
    """Degradation model for realistic downsampling."""
    
    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Blur kernel
        self.blur = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(1, 1, 5, padding=0, bias=False)
        )
        
        # Initialize blur kernel as Gaussian
        kernel = self.get_gaussian_kernel(5, 1.0)
        self.blur[1].weight.data = kernel.unsqueeze(0).unsqueeze(0)
        self.blur[1].weight.requires_grad = False
    
    def get_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Generate Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        return g
    
    def forward(self, x):
        # Apply blur
        x = self.blur(x)
        
        # Downsample
        x = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False)
        
        return x

class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Remove self-similarity
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)
        
        # Compute loss
        labels = labels[~mask].view(labels.shape[0], -1)
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

class SelfSupervisedSR:
    """Main class for self-supervised super-resolution."""
    
    def __init__(self, device: torch.device, scale_factor: int = 4):
        self.device = device
        self.scale_factor = scale_factor
        
        # Initialize networks
        self.generator = SRGenerator().to(device)
        self.discriminator = SRDiscriminator().to(device)
        self.degradation_model = DegradationModel(scale_factor).to(device)
        
        # Loss functions
        self.content_loss = nn.L1Loss()
        self.adversarial_loss = nn.BCELoss()
        self.contrastive_loss = ContrastiveLoss()
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        
        # Training parameters
        self.lambda_content = 1.0
        self.lambda_adversarial = 0.001
        self.lambda_cycle = 10.0
        self.lambda_contrastive = 0.1
    
    def train_step(self, hr_patches: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        batch_size = hr_patches.size(0)
        real_labels = torch.ones(batch_size, 1, 1, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1, 1, 1).to(self.device)
        
        # Generate low-resolution patches
        lr_patches = self.degradation_model(hr_patches)
        
        # ---------------------
        # Train Generator
        # ---------------------
        self.g_optimizer.zero_grad()
        
        # Generate super-resolved images
        sr_patches = self.generator(lr_patches)
        
        # Content loss
        content_loss = self.content_loss(sr_patches, hr_patches)
        
        # Adversarial loss
        sr_validity = self.discriminator(sr_patches)
        adversarial_loss = self.adversarial_loss(sr_validity, real_labels)
        
        # Cycle consistency loss
        lr_reconstructed = self.degradation_model(sr_patches)
        cycle_loss = self.content_loss(lr_reconstructed, lr_patches)
        
        # Contrastive loss
        features_hr = self.extract_features(hr_patches)
        features_sr = self.extract_features(sr_patches)
        contrastive_loss = self.contrastive_loss(features_sr, features_hr)
        
        # Total generator loss
        g_loss = (self.lambda_content * content_loss + 
                 self.lambda_adversarial * adversarial_loss +
                 self.lambda_cycle * cycle_loss +
                 self.lambda_contrastive * contrastive_loss)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()
        
        # Real images
        hr_validity = self.discriminator(hr_patches)
        d_real_loss = self.adversarial_loss(hr_validity, real_labels)
        
        # Fake images
        sr_validity = self.discriminator(sr_patches.detach())
        d_fake_loss = self.adversarial_loss(sr_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        self.d_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'content_loss': content_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'cycle_loss': cycle_loss.item(),
            'contrastive_loss': contrastive_loss.item()
        }
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for contrastive learning."""
        # Use intermediate features from discriminator
        features = self.discriminator.features(x)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features
    
    def train(self, dataloader: DataLoader, num_epochs: int, save_path: str):
        """Training loop."""
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch_idx, hr_patches in enumerate(pbar):
                    losses = self.train_step(hr_patches)
                    epoch_losses.append(losses)
                    
                    # Update progress bar
                    avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in losses.keys()}
                    pbar.set_postfix(avg_losses)
            
            # Save model
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
    
    def inference(self, lr_image: torch.Tensor) -> torch.Tensor:
        """Inference on low-resolution image."""
        self.generator.eval()
        with torch.no_grad():
            sr_image = self.generator(lr_image)
        return sr_image
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'scale_factor': self.scale_factor,
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Supervised Super-Resolution')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--scale_factor', type=int, default=4, help='Super-resolution scale factor')
    parser.add_argument('--patch_size', type=int, default=64, help='Training patch size')
    parser.add_argument('--num_patches', type=int, default=1000, help='Number of patches per epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Path to save checkpoints')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset and dataloader
    dataset = SelfSupervisedSRDataset(args.image_path, args.patch_size, args.num_patches)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = SelfSupervisedSR(device, args.scale_factor)
    
    # Train
    model.train(dataloader, args.num_epochs, args.save_path)
    
    # Save final model
    model.save_model(os.path.join(args.save_path, 'final_model.pth'))
    print(f'Training completed. Model saved to {args.save_path}')

if __name__ == '__main__':
    main()