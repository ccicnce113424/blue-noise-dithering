#!/usr/bin/env python3
"""Comprehensive demonstration of all blue noise dithering features."""

import os
import sys
from PIL import Image

# Add parent directory to path for importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blue_noise_dithering.core import BlueNoiseDitherer
from blue_noise_dithering.palette import PaletteLoader


def ensure_sample_files():
    """Ensure sample files exist."""
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = ['test_image.png', 'blue_noise.png', 'sample_palette.txt']
    missing_files = []
    
    for filename in required_files:
        filepath = os.path.join(examples_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        print(f"Missing sample files: {', '.join(missing_files)}")
        print("Please run 'python examples/generate_samples.py' first.")
        return False
    
    return True


def demo_color_distance_methods():
    """Demonstrate different color distance methods."""
    print("\n" + "="*60)
    print("DEMO 1: Color Distance Methods Comparison")
    print("="*60)
    
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load assets
    test_image = Image.open(os.path.join(examples_dir, 'test_image.png'))
    blue_noise = Image.open(os.path.join(examples_dir, 'blue_noise.png'))
    
    palette_loader = PaletteLoader()
    palette = palette_loader.load_from_file(os.path.join(examples_dir, 'sample_palette.txt'))
    
    methods = [
        ('weighted_rgb', 'Fast, good quality (recommended default)'),
        ('compuphase', 'Excellent quality with good performance'),
        ('ciede2000', 'Maximum quality (slower)'),
        ('cam16_ucs', 'Perceptually uniform color space'),
        ('oklab', 'Modern perceptual accuracy')
    ]
    
    for method, description in methods:
        print(f"\nProcessing with {method}: {description}")
        
        ditherer = BlueNoiseDitherer(color_distance=method, noise_strength=0.5)
        ditherer.set_palette(palette)
        ditherer.set_blue_noise(blue_noise)
        
        result = ditherer.dither(test_image)
        output_path = f"demo_method_{method}.png"
        result.save(output_path)
        print(f"  Saved: {output_path}")


def demo_adaptive_strategies():
    """Demonstrate adaptive noise strategies."""
    print("\n" + "="*60)
    print("DEMO 2: Adaptive Noise Strategies")
    print("="*60)
    
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load assets
    test_image = Image.open(os.path.join(examples_dir, 'test_image.png'))
    blue_noise = Image.open(os.path.join(examples_dir, 'blue_noise.png'))
    
    palette_loader = PaletteLoader()
    palette = palette_loader.load_from_file(os.path.join(examples_dir, 'sample_palette.txt'))
    
    strategies = [
        ('uniform', 'No adaptation (consistent noise)'),
        ('gradient', 'Adapts to gradients (preserves smooth transitions)'), 
        ('edge', 'Adapts to edges (preserves sharp details)'),
        ('contrast', 'Adapts to contrast (preserves texture)'),
        ('gradient_edge', 'Combination: gradient + edge detection'),
        ('gradient_contrast', 'Combination: gradient + contrast analysis'),
        ('edge_contrast', 'Combination: edge + contrast detection'),
        ('all', 'Combination: all strategies (maximum detail preservation)')
    ]
    
    for strategy, description in strategies:
        print(f"\nProcessing with adaptive strategy '{strategy}': {description}")
        
        ditherer = BlueNoiseDitherer(
            color_distance='weighted_rgb',
            noise_strength=0.5,
            adaptive_noise=True,
            adaptive_strategy=strategy
        )
        ditherer.set_palette(palette)
        ditherer.set_blue_noise(blue_noise)
        
        result = ditherer.dither(test_image)
        output_path = f"demo_adaptive_{strategy}.png"
        result.save(output_path)
        print(f"  Saved: {output_path}")


def demo_noise_map_visualization():
    """Demonstrate noise strength map output."""
    print("\n" + "="*60)
    print("DEMO 3: Noise Strength Map Visualization")
    print("="*60)
    
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load assets
    test_image = Image.open(os.path.join(examples_dir, 'test_image.png'))
    blue_noise = Image.open(os.path.join(examples_dir, 'blue_noise.png'))
    
    palette_loader = PaletteLoader()
    palette = palette_loader.load_from_file(os.path.join(examples_dir, 'sample_palette.txt'))
    
    strategies_for_map = [
        ('gradient', 'Gradient-based adaptation'),
        ('edge', 'Edge-based adaptation'),
        ('all', 'All strategies combined')
    ]
    
    for strategy, description in strategies_for_map:
        print(f"\nGenerating noise map for '{strategy}': {description}")
        
        ditherer = BlueNoiseDitherer(
            color_distance='weighted_rgb',
            noise_strength=0.5,
            adaptive_noise=True,
            adaptive_strategy=strategy
        )
        ditherer.set_palette(palette)
        ditherer.set_blue_noise(blue_noise)
        
        # Generate noise map
        noise_map = ditherer.generate_adaptive_noise_map(test_image)
        map_image = Image.fromarray((noise_map * 255).astype('uint8'), 'L')
        map_path = f"demo_noise_map_{strategy}.png"
        map_image.save(map_path)
        
        # Generate dithered image
        result = ditherer.dither(test_image)
        result_path = f"demo_with_map_{strategy}.png"
        result.save(result_path)
        
        print(f"  Noise map: {map_path}")
        print(f"  Dithered: {result_path}")
        print(f"  In noise map: White=high noise, Black=low noise, Gray=medium noise")


def demo_configuration_workflow():
    """Demonstrate configuration file workflow."""
    print("\n" + "="*60)
    print("DEMO 4: Configuration File Workflow")
    print("="*60)
    
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a comprehensive configuration
    config_content = """# Blue Noise Dithering Configuration
# Generated by comprehensive demo

# Color distance method
color_distance: weighted_rgb

# Noise settings
noise_strength: 0.6
adaptive_noise: true
adaptive_strategy: gradient_edge

# Alpha channel handling
alpha_method: dithering
alpha_threshold: 0.4

# Output options
output_noise_map: demo_config_noise_map.png
"""
    
    config_path = "demo_comprehensive_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created configuration file: {config_path}")
    print("Contents:")
    print(config_content)
    
    print(f"\nTo use this configuration:")
    print(f"python -m blue_noise_dithering.cli input.png output.png \\")
    print(f"    --config {config_path} \\")
    print(f"    --palette examples/sample_palette.txt \\")
    print(f"    --blue-noise examples/blue_noise.png")


def main():
    """Run comprehensive demonstration."""
    print("Blue Noise Dithering - Comprehensive Feature Demonstration")
    print("="*70)
    
    if not ensure_sample_files():
        return
    
    print("\nThis demo will showcase all major features:")
    print("1. Different color distance methods")
    print("2. Adaptive noise strategies") 
    print("3. Noise strength map visualization")
    print("4. Configuration file workflow")
    
    input("\nPress Enter to continue...")
    
    try:
        demo_color_distance_methods()
        demo_adaptive_strategies()
        demo_noise_map_visualization()
        demo_configuration_workflow()
        
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("- demo_method_*.png: Different color distance methods")
        print("- demo_adaptive_*.png: Different adaptive strategies")
        print("- demo_noise_map_*.png: Noise strength visualizations")
        print("- demo_with_map_*.png: Dithered images with adaptive noise")
        print("- demo_comprehensive_config.yaml: Sample configuration")
        
        print("\nRecommended settings by use case:")
        print("- General use: --color-distance weighted_rgb --adaptive-noise --adaptive-strategy gradient")
        print("- Maximum quality: --color-distance ciede2000 --adaptive-strategy all")
        print("- Maximum speed: --color-distance weighted_rgb")
        print("- Complex images: --adaptive-strategy gradient_edge or all")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Make sure all dependencies are installed and sample files are generated.")


if __name__ == '__main__':
    main()