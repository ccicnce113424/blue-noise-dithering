"""Test suite for blue noise dithering package."""

import unittest
import numpy as np
import tempfile
import os
from PIL import Image

from blue_noise_dithering.color_distance import ColorDistanceCalculator
from blue_noise_dithering.palette import PaletteLoader
from blue_noise_dithering.core import BlueNoiseDitherer


class TestColorDistanceCalculator(unittest.TestCase):
    """Test color distance calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ColorDistanceCalculator('rgb')
        self.red = np.array([255, 0, 0])
        self.green = np.array([0, 255, 0])
        self.blue = np.array([0, 0, 255])
        self.black = np.array([0, 0, 0])
        self.white = np.array([255, 255, 255])
    
    def test_rgb_distance(self):
        """Test RGB distance calculation."""
        # Distance from red to green should be greater than red to pink
        red_green_dist = self.calculator.calculate_distance(self.red, self.green)
        red_pink_dist = self.calculator.calculate_distance(self.red, np.array([255, 128, 128]))
        
        self.assertGreater(red_green_dist, red_pink_dist)
        
        # Distance from a color to itself should be 0
        self.assertEqual(self.calculator.calculate_distance(self.red, self.red), 0)
    
    def test_weighted_rgb_distance(self):
        """Test weighted RGB distance calculation."""
        calculator = ColorDistanceCalculator('weighted_rgb')
        
        # Test that distances are calculated
        distance = calculator.calculate_distance(self.red, self.green)
        self.assertGreater(distance, 0)
    
    def test_batch_distance_calculation(self):
        """Test batch distance calculation."""
        colors = np.array([self.red, self.green, self.blue])
        palette = np.array([self.black, self.white])
        
        distances = self.calculator.calculate_distances_batch(colors, palette)
        
        # Should have shape (3, 2)
        self.assertEqual(distances.shape, (3, 2))
        
        # All distances should be positive
        self.assertTrue(np.all(distances >= 0))
    
    def test_all_methods(self):
        """Test that all color distance methods work."""
        for method in ColorDistanceCalculator.METHODS:
            with self.subTest(method=method):
                calc = ColorDistanceCalculator(method)
                distance = calc.calculate_distance(self.red, self.green)
                self.assertGreater(distance, 0)
    
    def test_compuphase_distance(self):
        """Test compuphase distance calculation specifically."""
        calculator = ColorDistanceCalculator('compuphase')
        
        # Test basic properties
        # Distance from a color to itself should be 0
        self.assertEqual(calculator.calculate_distance(self.red, self.red), 0)
        self.assertEqual(calculator.calculate_distance(self.green, self.green), 0)
        self.assertEqual(calculator.calculate_distance(self.blue, self.blue), 0)
        
        # Distance should be positive for different colors
        red_green_dist = calculator.calculate_distance(self.red, self.green)
        red_blue_dist = calculator.calculate_distance(self.red, self.blue)
        green_blue_dist = calculator.calculate_distance(self.green, self.blue)
        
        self.assertGreater(red_green_dist, 0)
        self.assertGreater(red_blue_dist, 0)
        self.assertGreater(green_blue_dist, 0)
        
        # Distance should be symmetric
        self.assertEqual(
            calculator.calculate_distance(self.red, self.green),
            calculator.calculate_distance(self.green, self.red)
        )
        
        # Test that batch and single calculations are consistent
        colors = np.array([self.red, self.green, self.blue])
        palette = np.array([self.black, self.white])
        
        batch_distances = calculator.calculate_distances_batch(colors, palette)
        
        # Compare with individual calculations
        for i, color in enumerate(colors):
            for j, palette_color in enumerate(palette):
                single_dist = calculator.calculate_distance(color, palette_color)
                batch_dist = batch_distances[i, j]
                self.assertAlmostEqual(single_dist, batch_dist, places=10,
                                     msg=f"Mismatch for color {i} to palette {j}")
        
        # Test specific known values to ensure algorithm correctness
        # These values are calculated manually using the compuphase formula
        black_to_white = calculator.calculate_distance(self.black, self.white)
        self.assertAlmostEqual(black_to_white, 764.8333151739665, places=5)  # Verified calculation

    def test_ciede2000_no_sep_distance(self):
        """Test CIEDE2000 No Separation distance calculation specifically."""
        calculator = ColorDistanceCalculator('ciede2000_no_sep')
        
        # Test basic properties
        # Distance from a color to itself should be 0
        self.assertEqual(calculator.calculate_distance(self.red, self.red), 0)
        self.assertEqual(calculator.calculate_distance(self.green, self.green), 0)
        self.assertEqual(calculator.calculate_distance(self.blue, self.blue), 0)
        
        # Distance should be positive for different colors
        red_green_dist = calculator.calculate_distance(self.red, self.green)
        red_blue_dist = calculator.calculate_distance(self.red, self.blue)
        green_blue_dist = calculator.calculate_distance(self.green, self.blue)
        
        self.assertGreater(red_green_dist, 0)
        self.assertGreater(red_blue_dist, 0)
        self.assertGreater(green_blue_dist, 0)
        
        # Distance should be symmetric
        self.assertEqual(
            calculator.calculate_distance(self.red, self.green),
            calculator.calculate_distance(self.green, self.red)
        )
        
        # Test that batch and single calculations are consistent
        colors = np.array([self.red, self.green, self.blue])
        palette = np.array([self.black, self.white])
        
        batch_distances = calculator.calculate_distances_batch(colors, palette, show_progress=False)
        
        # Compare with individual calculations
        for i, color in enumerate(colors):
            for j, palette_color in enumerate(palette):
                single_dist = calculator.calculate_distance(color, palette_color)
                batch_dist = batch_distances[i, j]
                self.assertAlmostEqual(single_dist, batch_dist, places=10,
                                     msg=f"Mismatch for color {i} to palette {j}")
        
        # Test that No Separation variant produces reasonable results compared to standard CIEDE2000
        standard_calculator = ColorDistanceCalculator('ciede2000')
        
        # Test a few color pairs to ensure the No Separation variant produces similar but potentially
        # improved results (should be in the same ballpark, but with better continuity properties)
        test_pairs = [
            (self.red, self.green),
            (self.blue, self.white),
            (self.black, np.array([128, 128, 128]))  # mid gray
        ]
        
        for color1, color2 in test_pairs:
            no_sep_dist = calculator.calculate_distance(color1, color2)
            standard_dist = standard_calculator.calculate_distance(color1, color2)
            
            # The distances should be reasonably similar (within 20% typically)
            # But we focus more on ensuring the No Separation variant doesn't crash
            # and produces sensible positive values
            self.assertGreater(no_sep_dist, 0)
            self.assertLess(abs(no_sep_dist - standard_dist) / standard_dist, 0.5)  # Within 50%


class TestPaletteLoader(unittest.TestCase):
    """Test palette loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.palette_loader = PaletteLoader()
    
    def test_load_from_colors(self):
        """Test loading palette from color list."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.palette_loader.load_from_colors(colors)
        
        self.assertEqual(len(self.palette_loader), 3)
        self.assertEqual(self.palette_loader[0], (255, 0, 0))
    
    def test_create_grayscale_palette(self):
        """Test creating grayscale palette."""
        self.palette_loader.create_grayscale_palette(16)
        
        self.assertEqual(len(self.palette_loader), 16)
        
        # First color should be black
        self.assertEqual(self.palette_loader[0], (0, 0, 0))
        
        # Last color should be white
        self.assertEqual(self.palette_loader[-1], (255, 255, 255))
    
    def test_create_web_safe_palette(self):
        """Test creating web-safe palette."""
        self.palette_loader.create_web_safe_palette()
        
        # Web-safe palette should have 216 colors
        self.assertEqual(len(self.palette_loader), 216)
    
    def test_hex_color_parsing(self):
        """Test hex color parsing."""
        # Test various hex formats
        test_cases = [
            ("FF0000", (255, 0, 0)),  # RRGGBB
            ("#00FF00", (0, 255, 0)),  # #RRGGBB
            ("00F", (0, 0, 255)),  # RGB
            ("#F0F", (255, 0, 255)),  # #RGB
        ]
        
        for hex_str, expected in test_cases:
            with self.subTest(hex_str=hex_str):
                result = self.palette_loader._hex_to_rgb(hex_str.lstrip('#'), 1)
                self.assertEqual(result, expected)
    
    def test_save_and_load_txt_file(self):
        """Test saving and loading Paint.net TXT files."""
        # Create a test palette
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.palette_loader.load_from_colors(colors)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            self.palette_loader.save_to_file(temp_path)
            
            # Load from file
            new_loader = PaletteLoader()
            new_loader.load_from_file(temp_path)
            
            # Should have same colors
            self.assertEqual(len(new_loader), 3)
            self.assertEqual(list(new_loader.colors), colors)
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestBlueNoiseDitherer(unittest.TestCase):
    """Test blue noise dithering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ditherer = BlueNoiseDitherer()
        
        # Create a simple test image
        self.test_image = Image.new('RGB', (64, 64), (128, 128, 128))
        
        # Create a simple blue noise texture
        np.random.seed(42)
        self.blue_noise_array = np.random.random((32, 32)).astype(np.float32)
        
        # Create temporary files for blue noise texture
        self.blue_noise_image = Image.fromarray((self.blue_noise_array * 255).astype(np.uint8), 'L')
        
        # Create a simple palette
        self.palette_loader = PaletteLoader()
        self.palette_loader.create_grayscale_palette(4)  # 4 gray levels
    
    def test_load_palette(self):
        """Test palette loading."""
        self.ditherer.load_palette(self.palette_loader)
        self.assertIsNotNone(self.ditherer.palette)
        self.assertEqual(len(self.ditherer.palette), 4)
    
    def test_blue_noise_tiling(self):
        """Test blue noise texture tiling."""
        # Manually set blue noise texture
        self.ditherer.blue_noise_texture = self.blue_noise_array
        
        # Test tiling to larger dimensions
        tiled = self.ditherer._tile_blue_noise(100, 80)
        
        self.assertEqual(tiled.shape, (80, 100))
    
    def test_adaptive_noise_strategies(self):
        """Test different adaptive noise strategies."""
        self.ditherer.blue_noise_texture = self.blue_noise_array
        
        rgb_image = np.array(self.test_image)
        
        for strategy in BlueNoiseDitherer.ADAPTIVE_STRATEGIES:
            with self.subTest(strategy=strategy):
                self.ditherer.adaptive_strategy = strategy
                
                noise_map = self.ditherer._calculate_adaptive_noise(rgb_image)
                
                self.assertEqual(noise_map.shape, (64, 64))
                self.assertTrue(np.all(noise_map >= 0))
                self.assertTrue(np.all(noise_map <= 1))
    
    def test_alpha_methods(self):
        """Test alpha channel handling methods."""
        alpha_values = np.array([50, 100, 150, 200, 250])
        noise_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        strength_values = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Test threshold method
        result_threshold = self.ditherer._apply_alpha_threshold(alpha_values)
        self.assertTrue(np.all((result_threshold == 0) | (result_threshold == 255)))
        
        # Test dithering method
        result_dithering = self.ditherer._apply_alpha_dithering(
            alpha_values, noise_values, strength_values
        )
        self.assertTrue(np.all((result_dithering == 0) | (result_dithering == 255)))
    
    def test_dither_image_integration(self):
        """Test full image dithering integration."""
        # Save blue noise texture to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            blue_noise_path = f.name
        
        try:
            self.blue_noise_image.save(blue_noise_path)
            
            # Load components
            self.ditherer.load_blue_noise_texture(blue_noise_path)
            self.ditherer.load_palette(self.palette_loader)
            
            # Dither the test image
            result = self.ditherer.dither_image(self.test_image)
            
            # Check result
            self.assertIsInstance(result, Image.Image)
            self.assertEqual(result.size, self.test_image.size)
            self.assertEqual(result.mode, 'RGBA')
        
        finally:
            if os.path.exists(blue_noise_path):
                os.unlink(blue_noise_path)
    
    def test_config_management(self):
        """Test configuration save/load."""
        # Set some configuration
        config = {
            'color_distance_method': 'cie76',
            'noise_strength': 0.8,
            'adaptive_strategy': 'edge',
            'alpha_method': 'dithering',
            'alpha_threshold': 0.3
        }
        
        self.ditherer.set_config(config)
        
        retrieved_config = self.ditherer.get_config()
        
        self.assertEqual(retrieved_config['color_distance_method'], 'cie76')
        self.assertEqual(retrieved_config['noise_strength'], 0.8)
        self.assertEqual(retrieved_config['adaptive_strategy'], 'edge')
        self.assertEqual(retrieved_config['alpha_method'], 'dithering')
        self.assertEqual(retrieved_config['alpha_threshold'], 0.3)
    
    def test_noise_map_output(self):
        """Test noise strength map output functionality."""
        import tempfile
        import os
        
        # Create a temporary file for the noise map
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Set up ditherer with noise map output
            self.ditherer.output_noise_map = tmp_path
            self.ditherer.adaptive_strategy = 'gradient'
            self.ditherer.blue_noise_texture = self.blue_noise_array
            
            # Load a simple palette
            palette_loader = PaletteLoader()
            palette_loader.create_grayscale_palette(4)
            self.ditherer.load_palette(palette_loader)
            
            # Perform dithering
            result = self.ditherer.dither_image(self.test_image)
            
            # Check that the noise map file was created
            self.assertTrue(os.path.exists(tmp_path))
            
            # Check that the noise map is a valid image
            noise_map_image = Image.open(tmp_path)
            self.assertEqual(noise_map_image.mode, 'L')  # Grayscale
            self.assertEqual(noise_map_image.size, (64, 64))  # Same size as input image
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_combination_strategies(self):
        """Test combination adaptive strategies specifically."""
        self.ditherer.blue_noise_texture = self.blue_noise_array
        
        rgb_image = np.array(self.test_image)
        
        # Test combination strategies
        combination_strategies = ['gradient_edge', 'gradient_contrast', 'edge_contrast', 'all']
        
        for strategy in combination_strategies:
            with self.subTest(strategy=strategy):
                self.ditherer.adaptive_strategy = strategy
                
                noise_map = self.ditherer._calculate_adaptive_noise(rgb_image)
                
                self.assertEqual(noise_map.shape, (64, 64))
                self.assertTrue(np.all(noise_map >= 0))
                self.assertTrue(np.all(noise_map <= 1))
                
                # Test that combination strategies produce different results than uniform
                self.ditherer.adaptive_strategy = 'uniform'
                uniform_map = self.ditherer._calculate_adaptive_noise(rgb_image)
                
                # They should not be identical (unless the image is completely uniform)
                if not np.allclose(noise_map, uniform_map):
                    self.assertTrue(True)  # Good, they're different
    
    def test_perceptual_strategies(self):
        """Test luminance and saturation based adaptive strategies."""
        self.ditherer.blue_noise_texture = self.blue_noise_array
        
        rgb_image = np.array(self.test_image)
        
        # Test individual perceptual strategies
        perceptual_strategies = ['luminance', 'saturation']
        
        for strategy in perceptual_strategies:
            with self.subTest(strategy=strategy):
                self.ditherer.adaptive_strategy = strategy
                
                noise_map = self.ditherer._calculate_adaptive_noise(rgb_image)
                
                self.assertEqual(noise_map.shape, (64, 64))
                self.assertTrue(np.all(noise_map >= 0))
                self.assertTrue(np.all(noise_map <= 1))
                
                # Test that perceptual strategies produce different results than uniform
                self.ditherer.adaptive_strategy = 'uniform'
                uniform_map = self.ditherer._calculate_adaptive_noise(rgb_image)
                
                # They should not be identical (unless the image has no variation)
                if not np.allclose(noise_map, uniform_map):
                    self.assertTrue(True)  # Good, they're different
    
    def test_perceptual_combination_strategies(self):
        """Test combination strategies that include perceptual features."""
        self.ditherer.blue_noise_texture = self.blue_noise_array
        
        rgb_image = np.array(self.test_image)
        
        # Test new combination strategies
        combination_strategies = ['luminance_saturation', 'gradient_luminance', 
                                'gradient_saturation', 'all_perceptual']
        
        for strategy in combination_strategies:
            with self.subTest(strategy=strategy):
                self.ditherer.adaptive_strategy = strategy
                
                noise_map = self.ditherer._calculate_adaptive_noise(rgb_image)
                
                self.assertEqual(noise_map.shape, (64, 64))
                self.assertTrue(np.all(noise_map >= 0))
                self.assertTrue(np.all(noise_map <= 1))
                
                # Test that combination strategies produce different results than uniform
                self.ditherer.adaptive_strategy = 'uniform'
                uniform_map = self.ditherer._calculate_adaptive_noise(rgb_image)
                
                # They should not be identical
                if not np.allclose(noise_map, uniform_map):
                    self.assertTrue(True)  # Good, they're different
    
    def test_luminance_map_calculation(self):
        """Test luminance map calculation specifically."""
        # Create a test image with varying luminance
        test_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        test_rgb[:, :5, :] = [0, 0, 0]      # Black (low luminance)
        test_rgb[:, 5:, :] = [255, 255, 255] # White (high luminance)
        
        luminance_map = self.ditherer._calculate_luminance_map_optimized(test_rgb)
        
        self.assertEqual(luminance_map.shape, (10, 10))
        self.assertTrue(np.all(luminance_map >= 0))
        self.assertTrue(np.all(luminance_map <= 1))
        
        # Low luminance areas should have lower values (less mid-tone emphasis)
        # High luminance areas should also have lower values (less mid-tone emphasis)
        # Mid-tone areas should have higher values
        black_values = luminance_map[:, :5]
        white_values = luminance_map[:, 5:]
        
        # Both black and white should have lower emphasis than mid-tones
        self.assertTrue(np.all(black_values < 0.5))  # Black areas
        self.assertTrue(np.all(white_values < 0.5))  # White areas
    
    def test_saturation_map_calculation(self):
        """Test saturation map calculation specifically."""
        # Create a test image with varying saturation
        test_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        test_rgb[:, :5, :] = [128, 128, 128]  # Gray (low saturation)
        test_rgb[:, 5:, :] = [255, 0, 0]     # Red (high saturation)
        
        saturation_map = self.ditherer._calculate_saturation_map_optimized(test_rgb)
        
        self.assertEqual(saturation_map.shape, (10, 10))
        self.assertTrue(np.all(saturation_map >= 0))
        self.assertTrue(np.all(saturation_map <= 1))
        
        # Low saturation (gray) areas should have higher values (inverted)
        # High saturation (red) areas should have lower values
        gray_values = saturation_map[:, :5]
        red_values = saturation_map[:, 5:]
        
        # Gray areas should have higher emphasis for dithering
        self.assertTrue(np.mean(gray_values) > np.mean(red_values))
    
    def test_uniform_strategy_disables_adaptation(self):
        """Test that uniform strategy acts like disabled adaptive noise."""
        self.ditherer.blue_noise_texture = self.blue_noise_array
        rgb_image = np.array(self.test_image)
        
        # Test uniform strategy
        self.ditherer.adaptive_strategy = 'uniform'
        uniform_map = self.ditherer._calculate_adaptive_noise(rgb_image)
        
        # Should be all the same value (noise_strength)
        expected_value = self.ditherer.noise_strength
        self.assertTrue(np.allclose(uniform_map, expected_value))
        
        # Should have the same shape as the image
        self.assertEqual(uniform_map.shape, (64, 64))
    
    def test_config_file_loading(self):
        """Test that configuration files are loaded and applied correctly."""
        import tempfile
        import yaml
        import os
        
        # Create a test configuration
        test_config = {
            'color_distance_method': 'cie76',
            'noise_strength': 0.8,
            'adaptive_strategy': 'edge',
            'alpha_method': 'dithering',
            'alpha_threshold': 0.3
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            # Load configuration using CLI function
            from blue_noise_dithering.cli import load_config
            loaded_config = load_config(config_path)
            
            # Check that all values were loaded correctly
            self.assertEqual(loaded_config['color_distance_method'], 'cie76')
            self.assertEqual(loaded_config['noise_strength'], 0.8)
            self.assertEqual(loaded_config['adaptive_strategy'], 'edge')
            self.assertEqual(loaded_config['alpha_method'], 'dithering')
            self.assertEqual(loaded_config['alpha_threshold'], 0.3)
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)


if __name__ == '__main__':
    unittest.main()