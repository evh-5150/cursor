import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for super-resolution tasks.
    Uses pre-trained VGG19 features to compute perceptual similarity.
    """
    
    def __init__(self, device: torch.device, feature_layers: Optional[list] = None):
        super().__init__()
        
        # Load pre-trained VGG19 model
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.device = device
        
        # Default feature layers for perceptual loss
        if feature_layers is None:
            self.feature_layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_2, relu4_2
        else:
            self.feature_layers = feature_layers
        
        # Create feature extractors
        self.feature_extractors = nn.ModuleList()
        layer_idx = 0
        for i in range(len(vgg)):
            if isinstance(vgg[i], nn.ReLU):
                layer_idx += 1
                if layer_idx in self.feature_layers:
                    self.feature_extractors.append(nn.Sequential(*vgg[:i+1]))
        
        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between x and y.
        
        Args:
            x, y: Input tensors in [-1, 1] range, shape (B, 1, H, W)
        
        Returns:
            Perceptual loss value
        """
        # Convert from [-1, 1] to [0, 1] range
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
        
        # Convert grayscale to RGB by repeating channels
        x_rgb = x.repeat(1, 3, 1, 1)
        y_rgb = y.repeat(1, 3, 1, 1)
        
        # Normalize for VGG
        x_rgb = (x_rgb - self.mean) / self.std
        y_rgb = (y_rgb - self.mean) / self.std
        
        # Extract features and compute loss
        loss = 0.0
        for extractor in self.feature_extractors:
            x_feat = extractor(x_rgb)
            y_feat = extractor(y_rgb)
            loss += F.mse_loss(x_feat, y_feat)
        
        return loss / len(self.feature_extractors)

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    """
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, 
              window_size: int, channel: int, size_average: bool = True) -> torch.Tensor:
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss between img1 and img2.
        
        Args:
            img1, img2: Input tensors in [-1, 1] range
        
        Returns:
            SSIM loss value (1 - SSIM)
        """
        # Convert from [-1, 1] to [0, 1] range
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0
        
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1 loss) for robust training.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Charbonnier loss between x and y.
        
        Args:
            x, y: Input tensors
        
        Returns:
            Charbonnier loss value
        """
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.epsilon))

class GradientLoss(nn.Module):
    """
    Gradient loss to preserve edge information.
    """
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss between x and y.
        
        Args:
            x, y: Input tensors
        
        Returns:
            Gradient loss value
        """
        # Compute gradients
        x_grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        x_grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        y_grad_x = y[:, :, :, 1:] - y[:, :, :, :-1]
        y_grad_y = y[:, :, 1:, :] - y[:, :, :-1, :]
        
        # Compute gradient loss
        loss_x = self.l1_loss(x_grad_x, y_grad_x)
        loss_y = self.l1_loss(x_grad_y, y_grad_y)
        
        return loss_x + loss_y

class CombinedLoss(nn.Module):
    """
    Combined loss function for super-resolution training.
    """
    
    def __init__(self, device: torch.device, lambda_l1: float = 1.0, lambda_perceptual: float = 0.1,
                 lambda_ssim: float = 0.1, lambda_gradient: float = 0.05):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_gradient = lambda_gradient
        
        # Initialize loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss(device) if lambda_perceptual > 0 else None
        self.ssim_loss = SSIMLoss() if lambda_ssim > 0 else None
        self.gradient_loss = GradientLoss() if lambda_gradient > 0 else None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image tensor
            target: Target image tensor
        
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # L1 loss
        if self.lambda_l1 > 0:
            l1_loss = self.l1_loss(pred, target)
            total_loss += self.lambda_l1 * l1_loss
        
        # Perceptual loss
        if self.lambda_perceptual > 0 and self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(pred, target)
            total_loss += self.lambda_perceptual * perceptual_loss
        
        # SSIM loss
        if self.lambda_ssim > 0 and self.ssim_loss is not None:
            ssim_loss = self.ssim_loss(pred, target)
            total_loss += self.lambda_ssim * ssim_loss
        
        # Gradient loss
        if self.lambda_gradient > 0 and self.gradient_loss is not None:
            gradient_loss = self.gradient_loss(pred, target)
            total_loss += self.lambda_gradient * gradient_loss
        
        return total_loss