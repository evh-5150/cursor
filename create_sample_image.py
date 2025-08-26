#!/usr/bin/env python3
"""
Create a sample 16-bit grayscale image for testing super-resolution.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data, filters, morphology
import os

def create_sample_image(size=512, save_path="sample_image.png"):
    """Create a complex sample image with various features."""
    
    # Create base image
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Add multiple patterns
    image = np.zeros((size, size))
    
    # 1. Sinusoidal patterns
    image += 0.3 * np.sin(10 * x) * np.cos(8 * y)
    image += 0.2 * np.sin(15 * x + 0.5) * np.sin(12 * y + 0.3)
    
    # 2. Gaussian blobs
    image += 0.4 * np.exp(-((x - 0.3)**2 + (y - 0.7)**2) / 0.02)
    image += 0.3 * np.exp(-((x - 0.8)**2 + (y - 0.2)**2) / 0.01)
    image += 0.2 * np.exp(-((x - 0.6)**2 + (y - 0.6)**2) / 0.015)
    
    # 3. Radial pattern
    r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    image += 0.25 * np.cos(20 * r) * np.exp(-r / 0.3)
    
    # 4. Add some noise
    np.random.seed(42)  # For reproducibility
    image += 0.1 * np.random.randn(size, size)
    
    # 5. Add edges and structures
    # Create some rectangular structures
    image[100:150, 100:200] += 0.3
    image[300:350, 300:400] += 0.2
    image[400:450, 50:150] += 0.25
    
    # Add some diagonal lines
    for i in range(0, size, 50):
        cv2.line(image, (i, 0), (i + 100, 100), 0.2, 2)
    
    # Normalize to [0, 65535] for 16-bit
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 65535).astype(np.uint16)
    
    # Save the image
    cv2.imwrite(save_path, image)
    
    print(f"Sample image created: {save_path}")
    print(f"Image shape: {image.shape}")
    print(f"Value range: {image.min()} - {image.max()}")
    
    return image

def create_dicom_like_image(size=512, save_path="sample_dicom.png"):
    """Create a medical-like image with DICOM characteristics."""
    
    # Use a medical-like pattern
    image = data.camera()  # Use built-in camera image
    
    # Resize to desired size
    image = cv2.resize(image, (size, size))
    
    # Add some medical-like features
    # Simulate blood vessels
    for i in range(5):
        start_x = np.random.randint(0, size)
        start_y = np.random.randint(0, size)
        end_x = np.random.randint(0, size)
        end_y = np.random.randint(0, size)
        cv2.line(image, (start_x, start_y), (end_x, end_y), 100, 3)
    
    # Add some circular structures (like cells or lesions)
    for i in range(8):
        center_x = np.random.randint(50, size-50)
        center_y = np.random.randint(50, size-50)
        radius = np.random.randint(10, 30)
        cv2.circle(image, (center_x, center_y), radius, 150, -1)
    
    # Add noise to simulate medical imaging
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Convert to 16-bit
    image = (image.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
    
    # Save the image
    cv2.imwrite(save_path, image)
    
    print(f"Medical-like image created: {save_path}")
    print(f"Image shape: {image.shape}")
    print(f"Value range: {image.min()} - {image.max()}")
    
    return image

def visualize_image(image, title="Sample Image"):
    """Visualize the created image."""
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"{title} - Full")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    # Show a cropped region
    crop_size = min(128, image.shape[0] // 4)
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    cropped = image[center_y-crop_size//2:center_y+crop_size//2, 
                   center_x-crop_size//2:center_x+crop_size//2]
    plt.imshow(cropped, cmap='gray')
    plt.title(f"{title} - Cropped Region")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.hist(image.flatten(), bins=100, alpha=0.7)
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 2, 4)
    # Show frequency domain
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Frequency Domain")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create sample images."""
    print("Creating sample images for super-resolution testing...")
    
    # Create directory for samples
    os.makedirs("samples", exist_ok=True)
    
    # Create complex pattern image
    print("\n1. Creating complex pattern image...")
    complex_image = create_sample_image(512, "samples/complex_pattern.png")
    visualize_image(complex_image, "Complex Pattern")
    
    # Create medical-like image
    print("\n2. Creating medical-like image...")
    medical_image = create_dicom_like_image(512, "samples/medical_like.png")
    visualize_image(medical_image, "Medical-like")
    
    # Create a simple test image
    print("\n3. Creating simple test image...")
    simple_image = np.zeros((256, 256), dtype=np.uint16)
    
    # Add some simple patterns
    simple_image[50:100, 50:100] = 32768  # Square
    simple_image[150:200, 150:200] = 49152  # Another square
    
    # Add some lines
    cv2.line(simple_image, (0, 0), (255, 255), 16384, 5)
    cv2.line(simple_image, (0, 255), (255, 0), 24576, 5)
    
    cv2.imwrite("samples/simple_test.png", simple_image)
    print("Simple test image created: samples/simple_test.png")
    
    print("\nAll sample images created successfully!")
    print("You can now use these images to test the super-resolution system.")
    print("\nUsage examples:")
    print("1. Test with complex pattern:")
    print("   python test_system.py --input_image_path samples/complex_pattern.png")
    print("2. Test with medical-like image:")
    print("   python test_system.py --input_image_path samples/medical_like.png")
    print("3. Test with simple image:")
    print("   python test_system.py --input_image_path samples/simple_test.png")

if __name__ == "__main__":
    main()