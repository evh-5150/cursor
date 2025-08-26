#!/usr/bin/env python3
"""
Simple U-Net implementation for testing super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUnet(nn.Module):
    """
    Simple U-Net architecture for diffusion-based super-resolution.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 model_channels: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Conditional low-resolution input processing
        self.condition_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down1 = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, model_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(model_channels, model_channels * 2, 3, padding=1),
            nn.BatchNorm2d(model_channels * 2),
            nn.ReLU(),
            nn.Conv2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(model_channels * 2),
            nn.ReLU(),
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(model_channels * 2, model_channels * 4, 3, padding=1),
            nn.BatchNorm2d(model_channels * 4),
            nn.ReLU(),
            nn.Conv2d(model_channels * 4, model_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(model_channels * 4),
            nn.ReLU(),
        )
        
        # Middle
        self.mid_block1 = nn.Sequential(
            nn.Conv2d(model_channels * 4, model_channels * 4, 3, padding=1),
            nn.BatchNorm2d(model_channels * 4),
            nn.ReLU(),
        )
        
        self.mid_block2 = nn.Sequential(
            nn.Conv2d(model_channels * 4, model_channels * 4, 3, padding=1),
            nn.BatchNorm2d(model_channels * 4),
            nn.ReLU(),
        )
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(model_channels * 4, model_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(model_channels * 2),
            nn.ReLU(),
            nn.Conv2d(model_channels * 2, model_channels * 2, 3, padding=1),
            nn.BatchNorm2d(model_channels * 2),
            nn.ReLU(),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, model_channels, 3, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, model_channels, 3, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
        )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, 3, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Time projection layers
        self.time_proj1 = nn.Linear(time_emb_dim, model_channels)
        self.time_proj2 = nn.Linear(time_emb_dim, model_channels * 2)
        self.time_proj3 = nn.Linear(time_emb_dim, model_channels * 4)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input noisy image tensor of shape (B, 1, H, W)
            timestep: Timestep tensor of shape (B,)
            condition: Low-resolution condition tensor of shape (B, 1, H, W)
        
        Returns:
            Predicted noise tensor of shape (B, 1, H, W)
        """
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Initial convolutions
        x = self.init_conv(x)
        condition = self.condition_conv(condition)
        
        # Combine input with condition
        x = x + condition
        
        # Downsampling path
        x1 = self.down1(x)
        x1 = self.dropout(x1)
        
        x2 = self.down2(x1)
        x2 = self.dropout(x2)
        
        x3 = self.down3(x2)
        x3 = self.dropout(x3)
        
        # Middle
        x3 = self.mid_block1(x3)
        x3 = self.dropout(x3)
        x3 = self.mid_block2(x3)
        x3 = self.dropout(x3)
        
        # Upsampling path
        x = self.up1(x3)
        x = self.dropout(x)
        
        x = self.up2(x)
        x = self.dropout(x)
        
        # Final upsampling to match original size
        x = F.interpolate(x, size=(x3.shape[2] * 8, x3.shape[3] * 8), mode='bilinear', align_corners=False)
        x = self.dropout(x)
        
        # Final convolution
        return self.final_conv(x)

def test_simple_unet():
    """Test the simple U-Net implementation."""
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
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Condition shape: {condition.shape}")
    
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    print("✓ Simple U-Net test passed")
    
    return model

if __name__ == "__main__":
    test_simple_unet()