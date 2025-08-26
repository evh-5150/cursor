#!/usr/bin/env python3
"""
Debug script to check U-Net output sizes.
"""

import torch
import torch.nn.functional as F
from diffusion_model import SimpleUnet

def debug_unet():
    """Debug the U-Net implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SimpleUnet(dropout_rate=0.1).to(device)
    
    # Test with the same input size as quick_demo.py
    print("Testing with input size: 128x128 (same as quick_demo.py)")
    
    # Create input
    x = torch.randn(1, 1, 128, 128).to(device)
    t = torch.zeros(1, dtype=torch.long, device=device)
    condition = x.clone()
    
    # Forward pass
    with torch.no_grad():
        output = model(x, t, condition)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: 512x512")
    
    if output.shape[2:] == (512, 512):
        print("✓ Output size is correct!")
    else:
        print("✗ Output size is incorrect!")
        print(f"Actual output size: {output.shape[2:]}")

if __name__ == "__main__":
    debug_unet()