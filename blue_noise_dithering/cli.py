"""Command-line interface for blue noise dithering."""

import argparse
import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any

from .core import BlueNoiseDitherer
from .palette import PaletteLoader
from .color_distance import ColorDistanceCalculator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config if config else {}
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        config_dir = os.path.dirname(config_path)
        if config_dir:  # Only create directory if there is a directory component
            os.makedirs(config_dir, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Warning: Could not save config file {config_path}: {e}")


def create_sample_config(config_path: str) -> None:
    """Create a sample configuration file.
    
    Args:
        config_path: Path to save sample configuration
    """
    sample_config = {
        'color_distance_method': 'ciede2000',
        'noise_strength': 0.5,
        'adaptive_strategy': 'gradient_edge',
        'alpha_method': 'dithering',
        'alpha_threshold': 0.5,
        'output_noise_map': 'noise_strength_map.png'
    }
    
    save_config(sample_config, config_path)


def progress_callback(progress: float) -> None:
    """Progress callback for dithering process.
    
    Args:
        progress: Progress value between 0.0 and 1.0
    """
    # This is handled by tqdm in the core module
    pass


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Blue noise dithering tool with multiple color distance methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  blue-noise-dither input.png output.png --palette colors.txt --blue-noise noise.png
  
  # Advanced usage with configuration
  blue-noise-dither input.png output.png --config settings.yaml
  
  # Create sample configuration
  blue-noise-dither --create-config sample.yaml
  
  # Use uniform strategy to disable adaptive noise
  blue-noise-dither input.png output.png --palette colors.txt --blue-noise noise.png --adaptive-strategy uniform
  
Color distance methods:
  rgb, weighted_rgb, cie76, cie94, ciede2000, oklab, hsv
  
Alpha methods:
  threshold, dithering
  
Adaptive strategies:
  uniform (no adaptation - constant noise strength)
  gradient, edge, contrast (individual structural strategies)
  luminance, saturation (perceptual strategies)
  gradient_edge, gradient_contrast, edge_contrast, all (structural combinations)
  luminance_saturation, gradient_luminance, gradient_saturation (perceptual combinations)
  all_perceptual (combines all structural and perceptual strategies)
        """
    )
    
    # Input/Output
    parser.add_argument('input', nargs='?', help='Input image file')
    parser.add_argument('output', nargs='?', help='Output image file')
    
    # Required parameters
    parser.add_argument('--palette', '-p', type=str,
                       help='Palette file in Paint.net TXT format')
    parser.add_argument('--blue-noise', '-n', type=str,
                       help='Blue noise texture image file')
    
    # Color distance
    parser.add_argument('--color-distance', '-c', type=str, 
                       choices=ColorDistanceCalculator.METHODS,
                       help='Color distance calculation method (default: weighted_rgb)')
    
    # Noise settings
    parser.add_argument('--noise-strength', '-s', type=float,
                       help='Noise strength (0.0 to 1.0, default: 0.5)')
    parser.add_argument('--adaptive-strategy', type=str,
                       choices=BlueNoiseDitherer.ADAPTIVE_STRATEGIES,
                       help='Adaptive noise strategy (default: gradient, use "uniform" to disable adaptive noise)')
    parser.add_argument('--output-noise-map', type=str,
                       help='Save noise strength map as grayscale image to specified path')
    
    # Alpha handling
    parser.add_argument('--alpha-method', type=str,
                       choices=BlueNoiseDitherer.ALPHA_METHODS,
                       help='Alpha channel handling method (default: threshold)')
    parser.add_argument('--alpha-threshold', type=float,
                       help='Alpha threshold value (0.0 to 1.0, default: 0.5)')
    
    # Configuration
    parser.add_argument('--config', type=str,
                       help='Load settings from YAML configuration file')
    parser.add_argument('--save-config', type=str,
                       help='Save current settings to YAML configuration file')
    parser.add_argument('--create-config', type=str,
                       help='Create sample configuration file and exit')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    # Handle special actions
    if args.create_config:
        create_sample_config(args.create_config)
        print(f"Sample configuration created at {args.create_config}")
        return
    
    # Validate required arguments
    if not args.input or not args.output:
        parser.error("Input and output files are required")
    
    # Load configuration if specified
    config = {}
    if args.config:
        config = load_config(args.config)
        if args.verbose:
            print(f"Loaded configuration from {args.config}")
    
    # Override config with command line arguments (only if explicitly provided)
    if args.color_distance is not None:
        config['color_distance_method'] = args.color_distance
    if args.noise_strength is not None:
        config['noise_strength'] = args.noise_strength
    if args.adaptive_strategy is not None:
        config['adaptive_strategy'] = args.adaptive_strategy
    if args.output_noise_map is not None:
        config['output_noise_map'] = args.output_noise_map
    if args.alpha_method is not None:
        config['alpha_method'] = args.alpha_method
    if args.alpha_threshold is not None:
        config['alpha_threshold'] = args.alpha_threshold
    
    # Set defaults for missing config values
    config.setdefault('color_distance_method', 'weighted_rgb')
    config.setdefault('noise_strength', 0.5)
    config.setdefault('adaptive_strategy', 'gradient')
    config.setdefault('alpha_method', 'threshold')
    config.setdefault('alpha_threshold', 0.5)
    
    try:
        # Validate input files
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        palette_file = args.palette or config.get('palette_file')
        if not palette_file:
            raise ValueError("Palette file must be specified")
        if not os.path.exists(palette_file):
            raise FileNotFoundError(f"Palette file not found: {palette_file}")
        
        blue_noise_file = args.blue_noise or config.get('blue_noise_file')
        if not blue_noise_file:
            raise ValueError("Blue noise texture file must be specified")
        if not os.path.exists(blue_noise_file):
            raise FileNotFoundError(f"Blue noise file not found: {blue_noise_file}")
        
        # Initialize components
        if args.verbose:
            print("Initializing blue noise ditherer...")
        
        ditherer = BlueNoiseDitherer(
            color_distance_method=config['color_distance_method'],
            noise_strength=config['noise_strength'],
            adaptive_strategy=config['adaptive_strategy'],
            alpha_method=config['alpha_method'],
            alpha_threshold=config['alpha_threshold'],
            output_noise_map=config.get('output_noise_map')
        )
        
        # Load palette
        if args.verbose:
            print(f"Loading palette from {palette_file}...")
        
        palette_loader = PaletteLoader()
        palette_loader.load_from_file(palette_file)
        ditherer.load_palette(palette_loader)
        
        # Load blue noise texture
        if args.verbose:
            print(f"Loading blue noise texture from {blue_noise_file}...")
        
        ditherer.load_blue_noise_texture(blue_noise_file)
        
        # Process image
        if args.verbose:
            print(f"Processing {args.input}...")
            print(f"Settings: {ditherer.get_config()}")
        
        result_image = ditherer.dither_image(args.input, progress_callback)
        
        # Save result
        if args.verbose:
            print(f"Saving result to {args.output}...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        result_image.save(args.output)
        
        if args.verbose:
            print("Processing complete!")
        
        # Save configuration if requested
        if args.save_config:
            current_config = ditherer.get_config()
            current_config['palette_file'] = palette_file
            current_config['blue_noise_file'] = blue_noise_file
            save_config(current_config, args.save_config)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()