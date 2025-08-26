import torch
import numpy as np
import pydicom
from skimage.transform import resize
import torch.nn.functional as F
from typing import Tuple, Optional
import os

def load_dicom_image(dicom_path: str, device: torch.device) -> Tuple[torch.Tensor, Tuple[float, float], pydicom.Dataset]:
    """
    Load a 16-bit grayscale DICOM image and normalize it to [-1, 1] range.
    
    Args:
        dicom_path: Path to the DICOM file
        device: Target device for the tensor
    
    Returns:
        normalized_image: Image tensor in [-1, 1] range, shape (1, 1, H, W)
        original_range: Tuple of (min_val, max_val) from original image
        dicom_dataset: Original DICOM dataset for metadata
    """
    # Load DICOM file
    dicom_dataset = pydicom.dcmread(dicom_path)
    
    # Extract pixel data
    if hasattr(dicom_dataset, 'RescaleSlope') and hasattr(dicom_dataset, 'RescaleIntercept'):
        # Apply rescaling if available
        pixel_array = dicom_dataset.pixel_array * dicom_dataset.RescaleSlope + dicom_dataset.RescaleIntercept
    else:
        pixel_array = dicom_dataset.pixel_array.astype(np.float32)
    
    # Ensure 16-bit range
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    
    # Get original range
    min_val, max_val = float(pixel_array.min()), float(pixel_array.max())
    
    # Normalize to [-1, 1] range
    normalized_array = 2.0 * (pixel_array - min_val) / (max_val - min_val) - 1.0
    
    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(normalized_array).float().unsqueeze(0).unsqueeze(0)
    
    return image_tensor.to(device), (min_val, max_val), dicom_dataset

def custom_downsample(image: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    Create a low-resolution version of the input image by downsampling.
    
    Args:
        image: Input image tensor of shape (1, 1, H, W)
        scale_factor: Downsampling factor
    
    Returns:
        Low-resolution image tensor
    """
    # Apply Gaussian blur before downsampling to prevent aliasing
    kernel_size = scale_factor * 2 + 1
    sigma = scale_factor / 3.0
    
    # Create Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # Apply 2D Gaussian blur
    kernel_2d = kernel.unsqueeze(0) * kernel.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    # Pad the image
    pad_size = kernel_size // 2
    padded_image = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # Apply convolution
    blurred_image = F.conv2d(padded_image, kernel_2d, padding=0)
    
    # Downsample
    _, _, h, w = blurred_image.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    downsampled = F.interpolate(blurred_image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    return downsampled

def get_image_patches(image: torch.Tensor, patch_size: int, batch_size: int) -> torch.Tensor:
    """
    Extract random patches from the input image for training.
    
    Args:
        image: Input image tensor of shape (1, 1, H, W)
        patch_size: Size of patches to extract
        batch_size: Number of patches to extract
    
    Returns:
        Batch of patches of shape (batch_size, 1, patch_size, patch_size)
    """
    _, _, h, w = image.shape
    
    # Ensure we can extract patches
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image size ({h}x{w}) is smaller than patch size ({patch_size}x{patch_size})")
    
    patches = []
    for _ in range(batch_size):
        # Random starting position
        y_start = torch.randint(0, h - patch_size + 1, (1,)).item()
        x_start = torch.randint(0, w - patch_size + 1, (1,)).item()
        
        # Extract patch
        patch = image[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size]
        patches.append(patch)
    
    return torch.cat(patches, dim=0)

def save_16bit_dicom_image(image_array: np.ndarray, original_dicom: pydicom.Dataset, 
                          output_path: str, scale_factor: int = 1) -> None:
    """
    Save a 16-bit image as a DICOM file with appropriate metadata.
    
    Args:
        image_array: 16-bit image array
        original_dicom: Original DICOM dataset for metadata
        output_path: Output file path
        scale_factor: Scale factor for pixel spacing adjustment
    """
    # Create a copy of the original DICOM dataset
    new_dicom = original_dicom.copy()
    
    # Update pixel data
    new_dicom.PixelData = image_array.tobytes()
    new_dicom.Rows, new_dicom.Columns = image_array.shape
    
    # Update pixel spacing if available
    if hasattr(new_dicom, 'PixelSpacing'):
        original_spacing = new_dicom.PixelSpacing
        new_dicom.PixelSpacing = [original_spacing[0] / scale_factor, original_spacing[1] / scale_factor]
    
    # Update image position if available
    if hasattr(new_dicom, 'ImagePositionPatient'):
        # Adjust image position for the new size
        pass  # This would need more complex calculation based on the specific use case
    
    # Update metadata
    new_dicom.SeriesDescription = f"Super-resolved (x{scale_factor})"
    new_dicom.SeriesNumber = getattr(new_dicom, 'SeriesNumber', 1) + 1000
    
    # Save the new DICOM file
    new_dicom.save_as(output_path)
    print(f"Saved super-resolved DICOM to: {output_path}")

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1, img2: Image tensors in [-1, 1] range
    
    Returns:
        PSNR value in dB
    """
    # Convert to [0, 1] range
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1, img2: Image tensors in [-1, 1] range
        window_size: Size of the Gaussian window
    
    Returns:
        SSIM value
    """
    # Convert to [0, 1] range
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    
    # Simple SSIM implementation
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()