#!/usr/bin/env python3
"""
Example script demonstrating self-supervised super-resolution for DICOM images.
This script shows how to use the diffusion-based super-resolution system.
"""

import os
import torch
import numpy as np
import argparse
from main import run_training, run_inference
from utils import load_dicom_image, custom_downsample, calculate_psnr, calculate_ssim
import matplotlib.pyplot as plt

def create_sample_dicom():
    """
    Create a sample DICOM-like image for demonstration purposes.
    In practice, you would use real DICOM files.
    """
    # Create a synthetic 16-bit grayscale image
    size = 512
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Create a complex pattern
    image = (
        0.3 * np.sin(10 * x) * np.cos(10 * y) +
        0.5 * np.exp(-((x - 0.3)**2 + (y - 0.7)**2) / 0.1) +
        0.2 * np.random.rand(size, size)
    )
    
    # Normalize to 16-bit range
    image = ((image - image.min()) / (image.max() - image.min()) * 65535).astype(np.uint16)
    
    # Save as a simple image file (in practice, you'd use a real DICOM)
    import cv2
    cv2.imwrite("sample_image.png", image)
    print("Created sample image: sample_image.png")
    return "sample_image.png"

def demonstrate_super_resolution(args):
    """
    Demonstrate the super-resolution process.
    """
    print("=== Self-Supervised Super-Resolution Demo ===")
    
    # Check if we have a real DICOM file or need to create a sample
    if not os.path.exists(args.input_image_path):
        print(f"Input file {args.input_image_path} not found. Creating sample image...")
        args.input_image_path = create_sample_dicom()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the image
    print(f"Loading image from: {args.input_image_path}")
    i_lr, original_range, original_dicom = load_dicom_image(args.input_image_path, device)
    
    # Create low-resolution version
    print(f"Creating low-resolution version (scale factor: {args.upscale_factor})")
    i_llr = custom_downsample(i_lr, args.upscale_factor)
    
    # Display image statistics
    print(f"Original image shape: {i_lr.shape}")
    print(f"Low-resolution image shape: {i_llr.shape}")
    print(f"Original value range: {original_range}")
    
    # Run training
    print("\n=== Starting Training ===")
    model, output_dir, _, _, _ = run_training(args, device)
    
    # Run inference
    print("\n=== Starting Inference ===")
    args.model_path = os.path.join(output_dir, "model.pth")
    run_inference(args, device, model=model, output_dir=output_dir, 
                  i_lr=i_lr, original_range=original_range, original_dicom=original_dicom)
    
    # Evaluate results
    print("\n=== Evaluation ===")
    evaluate_results(output_dir, i_lr, i_llr, args.upscale_factor)
    
    print(f"\nResults saved in: {output_dir}")

def evaluate_results(output_dir, i_lr, i_llr, upscale_factor):
    """
    Evaluate the super-resolution results.
    """
    # Load the super-resolved result
    result_path = os.path.join(output_dir, f"inferred_mean_3samples.dcm")
    
    if os.path.exists(result_path):
        # In a real scenario, you would load the DICOM result
        # For this demo, we'll create a simple comparison
        print("Super-resolution completed successfully!")
        
        # Calculate metrics for the low-resolution vs original
        lr_upsampled = torch.nn.functional.interpolate(
            i_llr, size=i_lr.shape[2:], mode='bilinear', align_corners=False
        )
        
        psnr_lr = calculate_psnr(lr_upsampled, i_lr)
        ssim_lr = calculate_ssim(lr_upsampled, i_lr)
        
        print(f"Low-resolution upsampled PSNR: {psnr_lr:.2f} dB")
        print(f"Low-resolution upsampled SSIM: {ssim_lr:.4f}")
        
        # Note: In a real scenario, you would load the super-resolved result
        # and calculate metrics against the original high-resolution image
        print("Note: Full evaluation requires loading the super-resolved DICOM result")
    else:
        print("Super-resolution result not found.")

def main():
    parser = argparse.ArgumentParser(description="Demo script for DICOM super-resolution")
    parser.add_argument("--input_image_path", type=str, default="sample.dcm",
                       help="Path to input DICOM image")
    parser.add_argument("--upscale_factor", type=int, default=2,
                       help="Super-resolution scale factor")
    parser.add_argument("--training_steps", type=int, default=5000,
                       help="Number of training steps (reduced for demo)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (reduced for demo)")
    parser.add_argument("--patch_size", type=int, default=64,
                       help="Patch size (reduced for demo)")
    parser.add_argument("--timesteps", type=int, default=100,
                       help="Number of diffusion timesteps (reduced for demo)")
    parser.add_argument("--lambda_l1", type=float, default=1.0,
                       help="L1 loss weight")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1,
                       help="Perceptual loss weight")
    parser.add_argument("--use_amp", action='store_true',
                       help="Use automatic mixed precision")
    
    args = parser.parse_args()
    
    # Set output directory
    args.output_dir_base = "Demo_Results"
    
    # Run demonstration
    demonstrate_super_resolution(args)

if __name__ == "__main__":
    main()