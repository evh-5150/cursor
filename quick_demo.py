#!/usr/bin/env python3
"""
Quick demo of the super-resolution system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from simple_unet import SimpleUnet
from utils import calculate_psnr, calculate_ssim
import os

def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image."""
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    
    # Normalize to [-1, 1]
    image = 2.0 * (image - image.min()) / (image.max() - image.min()) - 1.0
    
    # Convert to tensor
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return image

def create_low_resolution(image: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
    """Create low-resolution version of the image."""
    # Downsample
    lr_image = F.interpolate(image, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
    
    # Add some noise to simulate real-world degradation
    noise = torch.randn_like(lr_image) * 0.1
    lr_image = lr_image + noise
    
    return lr_image

def simple_super_resolution(lr_image: torch.Tensor, model: SimpleUnet, device: torch.device) -> torch.Tensor:
    """Perform simple super-resolution using the model."""
    model.eval()
    
    with torch.no_grad():
        # Create dummy timestep and condition
        batch_size = lr_image.size(0)
        timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
        condition = lr_image.clone()
        
        # Forward pass
        sr_image = model(lr_image, timestep, condition)
    
    return sr_image

def evaluate_results(original: torch.Tensor, lr: torch.Tensor, sr: torch.Tensor) -> dict:
    """Evaluate the super-resolution results."""
    # Upsample LR image for comparison
    lr_upsampled = F.interpolate(lr, size=original.shape[2:], mode='bilinear', align_corners=False)
    
    # Calculate metrics
    psnr_lr = calculate_psnr(lr_upsampled, original)
    psnr_sr = calculate_psnr(sr, original)
    
    ssim_lr = calculate_ssim(lr_upsampled, original)
    ssim_sr = calculate_ssim(sr, original)
    
    return {
        'psnr_lr': psnr_lr,
        'psnr_sr': psnr_sr,
        'ssim_lr': ssim_lr,
        'ssim_sr': ssim_sr
    }

def visualize_results(original: torch.Tensor, lr: torch.Tensor, sr: torch.Tensor, 
                     metrics: dict, save_path: str = "demo_results.png"):
    """Visualize the results."""
    # Convert to numpy for visualization
    original_np = original.squeeze().cpu().numpy()
    lr_np = lr.squeeze().cpu().numpy()
    sr_np = sr.squeeze().cpu().numpy()
    
    # Normalize to [0, 1] for visualization
    original_np = (original_np + 1) / 2
    lr_np = (lr_np + 1) / 2
    sr_np = (sr_np + 1) / 2
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Low-resolution image (upsampled)
    lr_upsampled = F.interpolate(lr, size=original.shape[2:], mode='bilinear', align_corners=False)
    lr_upsampled_np = (lr_upsampled.squeeze().cpu().numpy() + 1) / 2
    axes[1].imshow(lr_upsampled_np, cmap='gray')
    axes[1].set_title(f'Low-Resolution (PSNR: {metrics["psnr_lr"]:.2f} dB, SSIM: {metrics["ssim_lr"]:.4f})')
    axes[1].axis('off')
    
    # Super-resolved image
    axes[2].imshow(sr_np, cmap='gray')
    axes[2].set_title(f'Super-Resolution (PSNR: {metrics["psnr_sr"]:.2f} dB, SSIM: {metrics["ssim_sr"]:.4f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to: {save_path}")

def main():
    """Main demo function."""
    print("=== Super-Resolution Demo ===")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if sample images exist
    sample_path = "samples/complex_pattern.png"
    if not os.path.exists(sample_path):
        print(f"Sample image not found: {sample_path}")
        print("Please run create_sample_image.py first to generate sample images.")
        return
    
    # Load sample image
    print(f"Loading sample image: {sample_path}")
    original_image = load_image(sample_path)
    print(f"Original image shape: {original_image.shape}")
    
    # Create low-resolution version
    scale_factor = 4
    print(f"Creating low-resolution version (scale factor: {scale_factor})")
    lr_image = create_low_resolution(original_image, scale_factor)
    print(f"Low-resolution image shape: {lr_image.shape}")
    
    # Create model
    print("Creating super-resolution model...")
    model = SimpleUnet(dropout_rate=0.1).to(device)
    
    # Perform super-resolution
    print("Performing super-resolution...")
    sr_image = simple_super_resolution(lr_image, model, device)
    print(f"Super-resolved image shape: {sr_image.shape}")
    
    # Evaluate results
    print("Evaluating results...")
    metrics = evaluate_results(original_image, lr_image, sr_image)
    
    print("\n=== Results ===")
    print(f"Low-resolution PSNR: {metrics['psnr_lr']:.2f} dB")
    print(f"Super-resolution PSNR: {metrics['psnr_sr']:.2f} dB")
    print(f"Low-resolution SSIM: {metrics['ssim_lr']:.4f}")
    print(f"Super-resolution SSIM: {metrics['ssim_sr']:.4f}")
    
    # Visualize results
    print("Creating visualization...")
    visualize_results(original_image, lr_image, sr_image, metrics, "demo_results.png")
    
    print("\n=== Demo Completed ===")
    print("Note: This is a simple demonstration with an untrained model.")
    print("For better results, you would need to train the model on your data.")

if __name__ == "__main__":
    main()