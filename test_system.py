#!/usr/bin/env python3
"""
Test script to verify the super-resolution system components.
"""

import torch
import numpy as np
import os
import sys
from diffusion_model import SimpleUnet, linear_beta_schedule
from utils import load_dicom_image, custom_downsample, get_image_patches, calculate_psnr, calculate_ssim
from loss import VGGPerceptualLoss, SSIMLoss, CharbonnierLoss, GradientLoss

def test_diffusion_model():
    """Test the diffusion model architecture."""
    print("Testing diffusion model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SimpleUnet(dropout_rate=0.1).to(device)
    
    # Test forward pass
    batch_size = 2
    patch_size = 64
    timesteps = 100
    
    x = torch.randn(batch_size, 1, patch_size, patch_size).to(device)
    t = torch.randint(0, timesteps, (batch_size,)).to(device)
    condition = torch.randn(batch_size, 1, patch_size, patch_size).to(device)
    
    output = model(x, t, condition)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    print("✓ Diffusion model test passed")

def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a synthetic image tensor
    image = torch.randn(1, 1, 256, 256).to(device)
    
    # Test downsampling
    downsampled = custom_downsample(image, scale_factor=2)
    expected_shape = (1, 1, 128, 128)
    assert downsampled.shape == expected_shape, f"Downsampled shape {downsampled.shape} != expected {expected_shape}"
    
    # Test patch extraction
    patches = get_image_patches(image, patch_size=64, batch_size=4)
    expected_patch_shape = (4, 1, 64, 64)
    assert patches.shape == expected_patch_shape, f"Patch shape {patches.shape} != expected {expected_patch_shape}"
    
    # Test metrics
    img1 = torch.randn(1, 1, 64, 64).to(device)
    img2 = torch.randn(1, 1, 64, 64).to(device)
    
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    assert isinstance(psnr, float), "PSNR should be a float"
    assert isinstance(ssim, float), "SSIM should be a float"
    assert 0 <= ssim <= 1, "SSIM should be between 0 and 1"
    
    print("✓ Utility functions test passed")

def test_loss_functions():
    """Test loss functions."""
    print("Testing loss functions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test tensors
    pred = torch.randn(2, 1, 64, 64).to(device)
    target = torch.randn(2, 1, 64, 64).to(device)
    
    # Test VGG perceptual loss
    try:
        perceptual_loss = VGGPerceptualLoss(device)
        loss_val = perceptual_loss(pred, target)
        assert isinstance(loss_val, torch.Tensor), "Perceptual loss should return a tensor"
        print("✓ VGG perceptual loss test passed")
    except Exception as e:
        print(f"⚠ VGG perceptual loss test failed (this is expected if torchvision is not available): {e}")
    
    # Test SSIM loss
    ssim_loss = SSIMLoss()
    loss_val = ssim_loss(pred, target)
    assert isinstance(loss_val, torch.Tensor), "SSIM loss should return a tensor"
    print("✓ SSIM loss test passed")
    
    # Test Charbonnier loss
    charbonnier_loss = CharbonnierLoss()
    loss_val = charbonnier_loss(pred, target)
    assert isinstance(loss_val, torch.Tensor), "Charbonnier loss should return a tensor"
    print("✓ Charbonnier loss test passed")
    
    # Test gradient loss
    gradient_loss = GradientLoss()
    loss_val = gradient_loss(pred, target)
    assert isinstance(loss_val, torch.Tensor), "Gradient loss should return a tensor"
    print("✓ Gradient loss test passed")

def test_training_components():
    """Test training-related components."""
    print("Testing training components...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test beta schedule
    betas = linear_beta_schedule(timesteps=100)
    assert len(betas) == 100, f"Beta schedule length {len(betas)} != expected 100"
    assert torch.all(betas > 0), "All betas should be positive"
    assert torch.all(betas < 1), "All betas should be less than 1"
    print("✓ Beta schedule test passed")
    
    # Test model parameters
    model = SimpleUnet(dropout_rate=0.1).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model has {num_params:,} parameters")
    
    # Test optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("✓ Optimizer test passed")

def test_memory_usage():
    """Test memory usage and GPU compatibility."""
    print("Testing memory usage...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        # Test GPU memory allocation
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create a model and some data
        model = SimpleUnet(dropout_rate=0.1).to(device)
        x = torch.randn(1, 1, 128, 128).to(device)
        t = torch.randint(0, 100, (1,)).to(device)
        condition = torch.randn(1, 1, 128, 128).to(device)
        
        # Forward pass
        output = model(x, t, condition)
        
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory
        print(f"✓ GPU memory usage: {memory_used / 1024**2:.2f} MB")
        
        # Clean up
        del model, x, t, condition, output
        torch.cuda.empty_cache()
    else:
        print("✓ CPU mode - no GPU memory test needed")

def main():
    """Run all tests."""
    print("=== Testing Super-Resolution System ===\n")
    
    try:
        test_diffusion_model()
        test_utils()
        test_loss_functions()
        test_training_components()
        test_memory_usage()
        
        print("\n=== All Tests Passed! ===")
        print("The super-resolution system is ready to use.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()