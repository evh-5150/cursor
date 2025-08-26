import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

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

class Block(nn.Module):
    """Basic convolutional block with optional residual connection."""
    
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up: bool = False, down: bool = False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        elif down:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Identity()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        # Second conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or up sample
        return self.transform(h)

class SimpleUnet(nn.Module):
    """
    Simple U-Net architecture for diffusion-based super-resolution.
    Designed for 16-bit grayscale DICOM images with conditional low-resolution input.
    """
    
    def __init__(self, image_size: int = 256, in_channels: int = 1, out_channels: int = 1, 
                 model_channels: int = 64, num_res_blocks: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        
        self.image_size = image_size
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
        self.down1 = Block(model_channels, model_channels, time_emb_dim, down=True)
        self.down2 = Block(model_channels, model_channels * 2, time_emb_dim, down=True)
        self.down3 = Block(model_channels * 2, model_channels * 4, time_emb_dim, down=True)
        
        # Middle
        mid_channels = model_channels * 4
        self.mid_block1 = Block(mid_channels, mid_channels, time_emb_dim)
        self.mid_block2 = Block(mid_channels, mid_channels, time_emb_dim)
        
        # Upsampling path
        self.up1 = Block(mid_channels, model_channels * 2, time_emb_dim, up=True)
        self.up2 = Block(model_channels * 2, model_channels, time_emb_dim, up=True)
        self.up3 = Block(model_channels, model_channels, time_emb_dim, up=True)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
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
        x1 = self.down1(x, t)
        x1 = self.dropout(x1)
        
        x2 = self.down2(x1, t)
        x2 = self.dropout(x2)
        
        x3 = self.down3(x2, t)
        x3 = self.dropout(x3)
        
        # Middle
        x3 = self.mid_block1(x3, t)
        x3 = self.dropout(x3)
        x3 = self.mid_block2(x3, t)
        x3 = self.dropout(x3)
        
        # Upsampling path with skip connections
        x = self.up1(torch.cat([x3, x2], dim=1), t)
        x = self.dropout(x)
        
        x = self.up2(torch.cat([x, x1], dim=1), t)
        x = self.dropout(x)
        
        x = self.up3(x, t)
        x = self.dropout(x)
        
        # Final convolution
        return self.final_conv(x)

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear beta schedule for the diffusion process.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        Beta schedule tensor
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule for the diffusion process.
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small constant to prevent division by zero
    
    Returns:
        Beta schedule tensor
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionModel:
    """
    Wrapper class for the diffusion process.
    """
    
    def __init__(self, model: SimpleUnet, timesteps: int = 1000, beta_schedule: str = 'linear'):
        self.model = model
        self.timesteps = timesteps
        
        # Setup beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from q(x_t | x_0).
        
        Args:
            x_start: Starting image
            t: Timestep
            noise: Optional noise tensor
        
        Returns:
            Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the loss for the diffusion model.
        
        Args:
            x_start: Starting image
            t: Timestep
            condition: Low-resolution condition
            noise: Optional noise tensor
        
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, condition)
        
        return F.mse_loss(predicted_noise, noise)