# Blue Noise Dithering

A comprehensive blue noise dithering tool that converts images to specified palettes using various color distance methods and noise strategies.

## Features

- **Multiple Color Distance Methods**: RGB, Weighted RGB, CIE 76, CIE 94, CIEDE2000, Oklab, HSV
- **Paint.net TXT Palette Support**: Load palettes in Paint.net TXT format
- **Blue Noise Texture**: Use custom blue noise textures with tiling support
- **Alpha Channel Handling**: Threshold and dithering methods for transparency
- **Adaptive Noise Strength**: Multiple strategies for adaptive noise application
- **Performance Optimized**: Efficient memory usage and processing speed
- **Progress Display**: Real-time progress indicators
- **Configuration Support**: Save and load settings from configuration files

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python -m blue_noise_dithering.cli input.png output.png --palette palette.txt --blue-noise noise.png
```

### Advanced Options

```bash
python -m blue_noise_dithering.cli input.png output.png \
    --palette palette.txt \
    --blue-noise noise.png \
    --color-distance ciede2000 \
    --noise-strength 0.5 \
    --alpha-method dithering \
    --config config.yaml
```

## Color Distance Methods

- `rgb`: Standard RGB Euclidean distance (fast)
- `weighted_rgb`: Weighted RGB with perceptual weights (fast, recommended)
- `cie76`: CIE76 Delta E (moderate speed)
- `cie94`: CIE94 Delta E (slower)
- `ciede2000`: CIEDE2000 Delta E (most accurate, very slow)
- `oklab`: Oklab perceptual color space (moderate speed)
- `hsv`: HSV color space distance (fast)

## Configuration File

Example `config.yaml`:

```yaml
color_distance: weighted_rgb
noise_strength: 0.5
adaptive_noise: true
adaptive_strategy: gradient
alpha_method: dithering
alpha_threshold: 0.5
```

## Performance Notes

- For best speed: Use `rgb` or `weighted_rgb` color distance methods
- For best quality: Use `ciede2000` (much slower but most perceptually accurate)
- For balanced performance: Use `cie76` or `oklab`
- CIEDE2000 calculation can be very slow on large images due to complex color space conversions