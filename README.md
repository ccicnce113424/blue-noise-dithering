# Blue Noise Dithering

A comprehensive blue noise dithering tool that converts images to specified palettes using various color distance methods and noise strategies.

## Features

- **Multiple Color Distance Methods**: RGB, Weighted RGB, CIE 76, CIE 94, CIEDE2000 (full standard), CIEDE2000 Fast (optimized), Oklab, HSV
- **Performance Optimized**: Vectorized implementations with significant speed improvements
- **Paint.net TXT Palette Support**: Load palettes in Paint.net TXT format  
- **Blue Noise Texture**: Use custom blue noise textures with tiling support
- **Alpha Channel Handling**: Threshold and dithering methods for transparency
- **Adaptive Noise Strength**: Multiple strategies for adaptive noise application including combination approaches
- **Noise Strength Visualization**: Export adaptive noise maps as grayscale images
- **High Performance**: Efficient memory usage and processing speed (130K-2M+ px/s depending on method)
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
    --color-distance ciede2000_fast \
    --noise-strength 0.5 \
    --adaptive-strategy gradient_edge \
    --alpha-method dithering \
    --output-noise-map noise_map.png \
    --config config.yaml
```

## Color Distance Methods

- `rgb`: Standard RGB Euclidean distance (fastest)
- `weighted_rgb`: Weighted RGB with perceptual weights (fast, recommended for general use)
- `cie76`: CIE76 Delta E (moderate speed, good accuracy)
- `cie94`: CIE94 Delta E (moderate speed, better accuracy)
- `ciede2000`: Full standard CIEDE2000 Delta E (best accuracy, moderate speed)
- `ciede2000_fast`: Optimized CIEDE2000 implementation (fast, excellent accuracy)
- `oklab`: Oklab perceptual color space (fast, modern accuracy)
- `hsv`: HSV color space distance (fast, for artistic effects)

## Adaptive Noise Strategies

The adaptive noise system adjusts noise strength based on image content for better dithering quality:

### Individual Strategies
- `uniform`: Consistent noise across the entire image (no adaptation)
- `gradient`: Reduces noise in high-gradient areas (preserves detail)
- `edge`: Reduces noise on detected edges (preserves sharp transitions) 
- `contrast`: Reduces noise in high-contrast areas (preserves texture detail)
- `luminance`: Emphasizes mid-tones for better perceptual dithering quality
- `saturation`: Increases noise in low-saturation (grayscale-like) areas

### Combination Strategies
- `gradient_edge`: Combines gradient and edge detection for comprehensive detail preservation
- `gradient_contrast`: Combines gradient and contrast for balanced detail and texture preservation
- `edge_contrast`: Combines edge and contrast detection for sharp detail preservation
- `luminance_saturation`: Combines perceptual luminance and saturation analysis
- `gradient_luminance`: Balances structural detail with perceptual importance
- `gradient_saturation`: Balances structural detail with color saturation
- `all`: Combines all three original structural strategies for maximum detail preservation
- `all_perceptual`: Combines all structural and perceptual strategies for optimal results

### Usage Examples

Enable adaptive noise with a specific strategy:
```bash
python -m blue_noise_dithering.cli input.png output.png \
    --palette palette.txt \
    --blue-noise noise.png \
    --adaptive-strategy gradient_edge
```

Disable adaptive noise (use uniform strategy):
```bash
python -m blue_noise_dithering.cli input.png output.png \
    --palette palette.txt \
    --blue-noise noise.png \
    --adaptive-strategy uniform
```

Use perceptual strategies for better luminance handling:
```bash
python -m blue_noise_dithering.cli input.png output.png \
    --palette palette.txt \
    --blue-noise noise.png \
    --adaptive-strategy luminance
```

Combine all perceptual strategies for optimal results:
```bash
python -m blue_noise_dithering.cli input.png output.png \
    --palette palette.txt \
    --blue-noise noise.png \
    --adaptive-strategy all_perceptual
```

## Noise Strength Map Output

You can save the noise strength map as a grayscale image to visualize how adaptive noise is being applied:

```bash
python -m blue_noise_dithering.cli input.png output.png \
    --palette palette.txt \
    --blue-noise noise.png \
    --adaptive-strategy all \
    --output-noise-map noise_visualization.png
```

The noise map shows:
- **White areas**: High noise strength (smooth areas that benefit from dithering)
- **Dark areas**: Low noise strength (detailed areas where noise is reduced)
- **Gray areas**: Medium noise strength (balanced dithering)

## Configuration File

Example `config.yaml`:

```yaml
color_distance_method: weighted_rgb
noise_strength: 0.5
adaptive_strategy: gradient_edge
alpha_method: dithering
alpha_threshold: 0.5
output_noise_map: noise_strength_map.png
```

## Performance Notes

**Recommended configurations by use case:**

- **Fast processing**: Use `weighted_rgb` or `ciede2000_fast` color distance methods
- **High quality**: Use `ciede2000` (full standard algorithm) for maximum perceptual accuracy  
- **Balanced performance**: Use `ciede2000_fast` - provides excellent color accuracy with good speed
- **Large images**: Use `ciede2000_fast` or `weighted_rgb` to maintain reasonable processing times
- **Small images/highest quality**: Use `ciede2000` (full standard) for maximum accuracy

**Performance comparison** (approximate processing speeds):
- `rgb`: ~2M+ px/s
- `weighted_rgb`: ~1.5M+ px/s  
- `ciede2000_fast`: ~780K-1.3M px/s (4-6x faster than standard)
- `ciede2000`: ~130K-240K px/s (full accuracy)
- `cie94`: ~900K+ px/s
- `oklab`: ~400K+ px/s

All methods benefit from recent vectorized optimizations and provide excellent results for their respective use cases.