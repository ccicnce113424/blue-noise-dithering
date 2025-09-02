# Blue Noise Dithering - Usage Guide

This guide demonstrates the usage of the blue noise dithering tool with practical examples.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate sample files:**
   ```bash
   python examples/generate_samples.py
   ```

3. **Basic dithering:**
   ```bash
   python -m blue_noise_dithering.cli examples/test_image.png output.png \
     --palette examples/sample_palette.txt \
     --blue-noise examples/blue_noise.png
   ```

## Examples

### Example 1: Fast RGB Dithering
```bash
python -m blue_noise_dithering.cli examples/test_image.png dithered_fast.png \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png \
  --color-distance rgb
```

### Example 2: High Quality with Adaptive Noise
```bash
python -m blue_noise_dithering.cli examples/test_image.png dithered_quality.png \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png \
  --color-distance weighted_rgb \
  --adaptive-noise \
  --adaptive-strategy gradient
```

### Example 3: Alpha Channel Processing
```bash
python -m blue_noise_dithering.cli transparent_input.png dithered_alpha.png \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png \
  --alpha-method dithering \
  --alpha-threshold 0.3
```

### Example 4: Using Configuration File
```bash
# Create a configuration file
python -m blue_noise_dithering.cli --create-config my_settings.yaml

# Edit my_settings.yaml as needed, then use it
python -m blue_noise_dithering.cli input.png output.png \
  --config my_settings.yaml \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png
```

### Example 5: Save Settings for Reuse
```bash
python -m blue_noise_dithering.cli examples/test_image.png output.png \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png \
  --color-distance weighted_rgb \
  --adaptive-noise \
  --save-config reusable_settings.yaml
```

### Example 6: Combination Adaptive Strategies
```bash
# Use gradient and edge detection combination for detailed image preservation
python -m blue_noise_dithering.cli examples/test_image.png dithered_combo.png \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png \
  --adaptive-noise \
  --adaptive-strategy gradient_edge \
  --color-distance weighted_rgb
```

### Example 7: Maximum Detail Preservation
```bash
# Use all strategies combined for maximum detail preservation
python -m blue_noise_dithering.cli examples/test_image.png dithered_max_detail.png \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png \
  --adaptive-noise \
  --adaptive-strategy all \
  --color-distance ciede2000
```

### Example 8: Noise Strength Map Visualization
```bash
# Generate both dithered image and noise strength visualization
python -m blue_noise_dithering.cli examples/test_image.png dithered_with_map.png \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png \
  --adaptive-noise \
  --adaptive-strategy gradient_contrast \
  --output-noise-map noise_visualization.png \
  --color-distance weighted_rgb
```

### Example 9: Complete Advanced Configuration
```bash
# Full featured configuration showing all new options
python -m blue_noise_dithering.cli examples/test_image.png final_output.png \
  --palette examples/sample_palette.txt \
  --blue-noise examples/blue_noise.png \
  --color-distance ciede2000 \
  --noise-strength 0.6 \
  --adaptive-noise \
  --adaptive-strategy all \
  --alpha-method dithering \
  --alpha-threshold 0.4 \
  --output-noise-map complete_noise_map.png \
  --save-config complete_settings.yaml
```

## Color Distance Methods Comparison

| Method | Speed | Quality | Best Use Case |
|--------|-------|---------|---------------|
| `rgb` | Fastest | Basic | Quick tests, draft processing |
| `weighted_rgb` | Fast | Good | General purpose, recommended default |
| `cie76` | Medium | Better | When color accuracy matters |
| `cie94` | Slow | Better | Professional color work |
| `ciede2000` | Very Slow | Best | Highest quality, small images |
| `oklab` | Medium | Good | Modern perceptual accuracy |
| `hsv` | Fast | Fair | Artistic effects |

## Adaptive Noise Strategies

### Individual Strategies
- **uniform**: Consistent noise across the image (fastest, no adaptation)
- **gradient**: Less noise in detailed areas (recommended for general use)
- **edge**: Preserves sharp edges and transitions
- **contrast**: Adapts to local contrast variations

### Combination Strategies
- **gradient_edge**: Combines gradient and edge detection for comprehensive detail preservation
- **gradient_contrast**: Combines gradient and contrast for balanced detail/texture preservation
- **edge_contrast**: Combines edge and contrast detection for sharp detail preservation  
- **all**: Combines all three strategies for maximum detail preservation (slowest)

### Strategy Selection Guide
| Strategy | Speed | Detail Preservation | Best Use Case |
|----------|-------|-------------------|---------------|
| `uniform` | Fastest | None | Simple images, speed priority |
| `gradient` | Fast | Good | General purpose (recommended) |
| `edge` | Fast | Sharp details | Line art, technical drawings |
| `contrast` | Fast | Texture | Photos with varied textures |
| `gradient_edge` | Medium | Excellent | Detailed artwork, illustrations |
| `gradient_contrast` | Medium | Excellent | Complex photos |
| `edge_contrast` | Medium | Sharp + Texture | Mixed content |
| `all` | Slowest | Maximum | Highest quality requirements |

### Noise Map Visualization
When using `--output-noise-map`, the generated grayscale image shows:
- **White pixels**: High noise strength (smooth areas)
- **Black pixels**: Low noise strength (detailed areas)  
- **Gray pixels**: Medium noise strength (moderate detail areas)

This visualization helps understand how the adaptive algorithm is working and can guide strategy selection.

## Performance Tips

1. **Use `weighted_rgb` for best speed/quality balance**
2. **For large images, avoid `ciede2000` unless quality is critical**
3. **Use smaller blue noise textures (32x32 to 128x128) for better performance**
4. **Enable adaptive noise for better visual quality**
5. **Process images in batches with the same settings using config files**

## Palette Creation

### Paint.net TXT Format
```
; Comment lines start with semicolon
#FF0000  ; Red
#00FF00  ; Green  
#0000FF  ; Blue
```

### Programmatic Palette Creation
```python
from blue_noise_dithering.palette import PaletteLoader

# Create grayscale palette
palette = PaletteLoader()
palette.create_grayscale_palette(16)
palette.save_to_file("grayscale_16.txt")

# Create web-safe palette
palette.create_web_safe_palette()
palette.save_to_file("websafe.txt")
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Install dependencies with `pip install -r requirements.txt`
2. **Very slow processing**: Use `rgb` or `weighted_rgb` instead of `ciede2000`
3. **Memory errors**: Process smaller images or reduce chunk size in code
4. **Poor results**: Try different adaptive noise strategies or adjust noise strength

### Performance Optimization

For large images or batch processing:
1. Use `rgb` or `weighted_rgb` color distance
2. Disable adaptive noise for maximum speed
3. Use smaller blue noise textures
4. Process multiple files with the same configuration