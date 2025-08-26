# Self-Supervised Super-Resolution for 16-bit Grayscale DICOM Images

This repository implements a self-supervised super-resolution system using diffusion models specifically designed for 16-bit grayscale DICOM medical images. The system uses a zero-shot approach that doesn't require paired training data.

## Overview

The system implements a conditional diffusion model that learns to super-resolve low-resolution DICOM images by:
1. **Self-supervised learning**: Uses the same image at different resolutions as training pairs
2. **Diffusion process**: Gradually denoises images while conditioning on low-resolution input
3. **Medical image optimization**: Specifically designed for 16-bit grayscale DICOM format

## Key Features

- **Zero-shot learning**: No paired training data required
- **DICOM support**: Native handling of 16-bit grayscale DICOM images
- **Conditional diffusion**: Uses low-resolution images as conditioning
- **Multiple loss functions**: L1, perceptual, SSIM, and gradient losses
- **Uncertainty estimation**: Provides uncertainty maps for predictions
- **Memory efficient**: Patch-based processing for large images

## Architecture

### Diffusion Model
- **U-Net backbone**: Encoder-decoder architecture with skip connections
- **Time conditioning**: Sinusoidal position embeddings for diffusion timesteps
- **Conditional input**: Low-resolution image as additional input
- **Multi-scale features**: Hierarchical feature extraction

### Loss Functions
- **L1 Loss**: Pixel-wise reconstruction loss
- **Perceptual Loss**: VGG19-based feature matching
- **SSIM Loss**: Structural similarity preservation
- **Gradient Loss**: Edge preservation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dicom-super-resolution
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Full training and inference pipeline
python main.py --mode full --input_image_path path/to/your/dicom.dcm --upscale_factor 2

# Training only
python main.py --mode train --input_image_path path/to/your/dicom.dcm --upscale_factor 2

# Inference only (requires trained model)
python main.py --mode inference --input_image_path path/to/your/dicom.dcm --model_path path/to/model.pth --upscale_factor 2
```

### Advanced Usage

```bash
python main.py \
    --mode full \
    --input_image_path Input/Images/MMG.dcm \
    --upscale_factor 4 \
    --training_steps 30000 \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --patch_size 128 \
    --timesteps 500 \
    --lambda_l1 1.0 \
    --lambda_perceptual 0.1 \
    --use_amp
```

## Parameters

### Training Parameters
- `--training_steps`: Number of training iterations (default: 20000)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size for training (default: 8)
- `--patch_size`: Size of training patches (default: 96)
- `--timesteps`: Number of diffusion timesteps (default: 200)
- `--dropout_rate`: Dropout rate in the model (default: 0.1)

### Loss Weights
- `--lambda_l1`: Weight for L1 loss (default: 1.0)
- `--lambda_perceptual`: Weight for perceptual loss (default: 0.1)

### Inference Parameters
- `--inf_patch_size`: Size of inference patches (default: 256)
- `--inf_overlap`: Overlap between inference patches (default: 128)
- `--n_samples`: Number of samples for uncertainty estimation (default: 3)

### Performance
- `--use_amp`: Enable automatic mixed precision for faster training
- `--interpolation_mode`: Interpolation mode for downsampling (bilinear/nearest-exact)

## Output

The system generates:
1. **Trained model**: Saved as `model.pth`
2. **Super-resolved DICOM**: High-resolution output with updated metadata
3. **Uncertainty map**: PNG visualization of prediction uncertainty
4. **Training logs**: Progress and loss information

## File Structure

```
├── main.py              # Main training and inference script
├── diffusion_model.py   # U-Net diffusion model architecture
├── utils.py            # DICOM utilities and image processing
├── loss.py             # Loss functions (L1, perceptual, SSIM, etc.)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technical Details

### Self-Supervised Learning
The system creates training pairs by:
1. Taking the original high-resolution DICOM image
2. Creating a low-resolution version through downsampling
3. Using the low-resolution image as conditioning for the diffusion model
4. Training the model to reconstruct the original high-resolution image

### Diffusion Process
1. **Forward process**: Gradually adds noise to the high-resolution image
2. **Reverse process**: Learns to denoise while conditioning on low-resolution input
3. **Conditioning**: Low-resolution image is processed and combined with noisy input

### DICOM Handling
- **16-bit support**: Full support for 16-bit grayscale images
- **Metadata preservation**: Maintains DICOM metadata and updates pixel spacing
- **Rescaling**: Handles DICOM rescale slope and intercept
- **Normalization**: Converts to [-1, 1] range for training

## Performance Tips

1. **Memory management**: Use smaller patch sizes for large images
2. **Training speed**: Enable `--use_amp` for faster training on compatible hardware
3. **Quality vs speed**: Increase `--timesteps` for better quality, decrease for faster inference
4. **Uncertainty**: Increase `--n_samples` for more reliable uncertainty estimation

## Limitations

- **Single image training**: Currently designed for single-image training
- **Computational cost**: Diffusion models require significant computational resources
- **Training time**: Full training can take several hours depending on image size and parameters

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dicom_sr_2024,
  title={Self-Supervised Super-Resolution for 16-bit Grayscale DICOM Images using Diffusion Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.