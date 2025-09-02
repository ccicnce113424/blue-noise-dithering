#!/usr/bin/env python3
"""Generate sample blue noise texture and test image for demonstration."""

import numpy as np
from PIL import Image
import os


def generate_blue_noise_texture(size: int = 64) -> np.ndarray:
    """Generate a simple blue noise-like texture.
    
    This is a simplified implementation for demonstration.
    Real blue noise would require more sophisticated algorithms.
    
    Args:
        size: Texture size (will be size x size)
        
    Returns:
        Blue noise texture as numpy array
    """
    # Start with white noise
    np.random.seed(42)
    noise = np.random.random((size, size))
    
    # Apply some filtering to approximate blue noise characteristics
    # This is a very simplified approach
    for _ in range(3):
        # Slight smoothing to reduce high frequency noise
        kernel = np.array([[0.1, 0.1, 0.1],
                          [0.1, 0.2, 0.1], 
                          [0.1, 0.1, 0.1]])
        
        # Apply convolution manually
        smoothed = np.zeros_like(noise)
        for i in range(1, size-1):
            for j in range(1, size-1):
                region = noise[i-1:i+2, j-1:j+2]
                smoothed[i, j] = np.sum(region * kernel)
        
        # Keep edges as original
        smoothed[0, :] = noise[0, :]
        smoothed[-1, :] = noise[-1, :]
        smoothed[:, 0] = noise[:, 0]
        smoothed[:, -1] = noise[:, -1]
        
        # Combine with original to maintain some high frequency content
        noise = 0.7 * noise + 0.3 * smoothed
    
    # Normalize to 0-1 range
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    return noise


def generate_test_image(width: int = 256, height: int = 256) -> Image.Image:
    """Generate a test image with various features.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Test image
    """
    # Create image with gradients and patterns
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Horizontal gradient
    for x in range(width):
        image[:height//3, x, 0] = int(255 * x / width)  # Red gradient
    
    # Vertical gradient  
    for y in range(height//3, 2*height//3):
        image[y, :, 1] = int(255 * (y - height//3) / (height//3))  # Green gradient
    
    # Checkerboard pattern
    checker_size = 16
    for y in range(2*height//3, height):
        for x in range(width):
            if ((x // checker_size) + (y // checker_size)) % 2:
                image[y, x] = [255, 255, 255]  # White
            else:
                image[y, x] = [0, 0, 255]      # Blue
    
    return Image.fromarray(image, 'RGB')


def main():
    """Generate sample files for demonstration."""
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating sample blue noise texture...")
    blue_noise = generate_blue_noise_texture(64)
    blue_noise_image = Image.fromarray((blue_noise * 255).astype(np.uint8), 'L')
    blue_noise_path = os.path.join(examples_dir, 'blue_noise.png')
    blue_noise_image.save(blue_noise_path)
    print(f"Saved blue noise texture to {blue_noise_path}")
    
    print("Generating test image...")
    test_image = generate_test_image(256, 256)
    test_image_path = os.path.join(examples_dir, 'test_image.png')
    test_image.save(test_image_path)
    print(f"Saved test image to {test_image_path}")
    
    print("\nExample usage:")
    print(f"python -m blue_noise_dithering.cli {test_image_path} dithered_output.png \\")
    print(f"    --palette {os.path.join(examples_dir, 'sample_palette.txt')} \\")
    print(f"    --blue-noise {blue_noise_path} \\")
    print("    --color-distance ciede2000_fast --adaptive-noise --adaptive-strategy gradient_edge")
    
    print("\nFor maximum quality:")
    print(f"python -m blue_noise_dithering.cli {test_image_path} dithered_max_quality.png \\")
    print(f"    --palette {os.path.join(examples_dir, 'sample_palette.txt')} \\")
    print(f"    --blue-noise {blue_noise_path} \\")
    print("    --color-distance ciede2000 --adaptive-noise --adaptive-strategy all")
    
    print("\nFor maximum speed:")
    print(f"python -m blue_noise_dithering.cli {test_image_path} dithered_fast.png \\")
    print(f"    --palette {os.path.join(examples_dir, 'sample_palette.txt')} \\")
    print(f"    --blue-noise {blue_noise_path} \\")
    print("    --color-distance weighted_rgb")


if __name__ == '__main__':
    main()