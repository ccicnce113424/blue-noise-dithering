#!/usr/bin/env python3
"""Performance comparison between different color distance methods."""

import time
import os
import sys
from PIL import Image
import numpy as np

# Add parent directory to path for importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blue_noise_dithering.core import BlueNoiseDitherer
from blue_noise_dithering.palette import PaletteLoader


def load_test_assets():
    """Load test image, palette, and blue noise texture."""
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load test image
    test_image_path = os.path.join(examples_dir, 'test_image.png')
    if not os.path.exists(test_image_path):
        print("Test image not found. Run generate_samples.py first.")
        return None, None, None
    
    test_image = Image.open(test_image_path)
    
    # Load palette
    palette_path = os.path.join(examples_dir, 'sample_palette.txt')
    palette_loader = PaletteLoader()
    palette = palette_loader.load_from_file(palette_path)
    
    # Load blue noise
    blue_noise_path = os.path.join(examples_dir, 'blue_noise.png')
    blue_noise = Image.open(blue_noise_path)
    
    return test_image, palette, blue_noise


def benchmark_method(method_name: str, test_image: Image.Image, palette: list, blue_noise: Image.Image):
    """Benchmark a specific color distance method."""
    print(f"\nTesting {method_name}...")
    
    # Create ditherer
    ditherer = BlueNoiseDitherer(
        color_distance=method_name,
        noise_strength=0.5,
        adaptive_noise=False  # Disable for pure method comparison
    )
    
    # Set assets
    ditherer.set_palette(palette)
    ditherer.set_blue_noise(blue_noise)
    
    # Measure time
    start_time = time.time()
    
    try:
        result = ditherer.dither(test_image)
        end_time = time.time()
        
        # Calculate performance
        processing_time = end_time - start_time
        pixel_count = test_image.width * test_image.height
        pixels_per_second = pixel_count / processing_time
        
        print(f"  Time: {processing_time:.3f}s")
        print(f"  Speed: {pixels_per_second:,.0f} px/s")
        
        # Save result
        output_path = f"benchmark_{method_name}.png"
        result.save(output_path)
        print(f"  Output: {output_path}")
        
        return processing_time, pixels_per_second
        
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


def main():
    """Run performance comparison for all color distance methods."""
    print("Blue Noise Dithering Performance Comparison")
    print("=" * 50)
    
    # Load test assets
    test_image, palette, blue_noise = load_test_assets()
    if test_image is None:
        return
    
    print(f"Test image: {test_image.width}x{test_image.height} pixels ({test_image.width * test_image.height:,} total)")
    print(f"Palette: {len(palette)} colors")
    print(f"Blue noise: {blue_noise.width}x{blue_noise.height}")
    
    # Methods to test (in order of expected speed)
    methods = [
        'rgb',
        'weighted_rgb', 
        'hsv',
        'oklab',
        'ciede2000_fast',
        'cie76',
        'cie94',
        'ciede2000'
    ]
    
    results = {}
    
    # Benchmark each method
    for method in methods:
        time_taken, speed = benchmark_method(method, test_image, palette, blue_noise)
        if time_taken is not None:
            results[method] = {'time': time_taken, 'speed': speed}
    
    # Print summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"{'Method':<15} {'Time (s)':<10} {'Speed (px/s)':<15} {'Relative Speed':<15}")
    print("-" * 65)
    
    # Sort by speed (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['speed'], reverse=True)
    fastest_speed = sorted_results[0][1]['speed'] if sorted_results else 1
    
    for method, data in sorted_results:
        relative_speed = data['speed'] / fastest_speed
        print(f"{method:<15} {data['time']:<10.3f} {data['speed']:<15,.0f} {relative_speed:<15.2f}x")
    
    print("\nRecommendations:")
    print("- For maximum speed: weighted_rgb or rgb")
    print("- For balanced quality/speed: ciede2000_fast")  
    print("- For maximum quality: ciede2000")
    print("- For modern perceptual accuracy: oklab")
    
    print(f"\nBenchmark images saved as benchmark_*.png")


if __name__ == '__main__':
    main()