"""Test validation of color distance algorithms for correctness and consistency."""

import unittest
import numpy as np
from blue_noise_dithering.color_distance import ColorDistanceCalculator


class TestColorDistanceValidation(unittest.TestCase):
    """Validate color distance algorithms for correctness and consistency."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test colors for validation
        self.test_colors = [
            np.array([255, 0, 0]),    # Red
            np.array([0, 255, 0]),    # Green  
            np.array([0, 0, 255]),    # Blue
            np.array([255, 255, 255]), # White
            np.array([0, 0, 0]),       # Black
            np.array([128, 128, 128]), # Gray
            np.array([255, 128, 128]), # Light red (pink)
            np.array([128, 255, 128]), # Light green
            np.array([128, 128, 255]), # Light blue
        ]
        
        # Tolerance for floating point comparisons
        self.tolerance = 1e-10
        
    def test_distance_properties(self):
        """Test basic mathematical properties of distance functions."""
        methods_to_test = ['cie76', 'cie94', 'ciede2000', 'oklab', 'hsv']
        
        for method in methods_to_test:
            with self.subTest(method=method):
                calculator = ColorDistanceCalculator(method)
                
                for color1 in self.test_colors:
                    for color2 in self.test_colors:
                        # Distance from a color to itself should be 0
                        self_distance = calculator.calculate_distance(color1, color1)
                        self.assertAlmostEqual(self_distance, 0.0, places=10,
                                             msg=f"Self-distance should be 0 for {method}")
                        
                        # Distance should be non-negative
                        distance = calculator.calculate_distance(color1, color2)
                        self.assertGreaterEqual(distance, 0.0,
                                              msg=f"Distance should be non-negative for {method}")
                        
                        # Distance should be symmetric (except for CIE94 which is asymmetric by standard)
                        if method != 'cie94':
                            reverse_distance = calculator.calculate_distance(color2, color1)
                            self.assertAlmostEqual(distance, reverse_distance, places=10,
                                                 msg=f"Distance should be symmetric for {method}")
    
    def test_cie94_asymmetric_behavior(self):
        """Test that CIE94 exhibits correct asymmetric behavior."""
        calculator = ColorDistanceCalculator('cie94')
        
        # Test with colors that should show clear asymmetric behavior
        red = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        
        distance_12 = calculator.calculate_distance(red, green)
        distance_21 = calculator.calculate_distance(green, red)
        
        # CIE94 should be asymmetric
        self.assertNotAlmostEqual(distance_12, distance_21, places=6,
                                msg="CIE94 should be asymmetric (standard behavior)")
        
        # Both distances should still be positive
        self.assertGreater(distance_12, 0.0, msg="CIE94 distance should be positive")
        self.assertGreater(distance_21, 0.0, msg="CIE94 reverse distance should be positive")
    
    def test_batch_consistency(self):
        """Test that batch and single calculations produce identical results."""
        methods_to_test = ['cie76', 'cie94', 'ciede2000', 'oklab', 'hsv']
        
        colors = np.array(self.test_colors[:4])  # Use subset for batch test
        palette = np.array(self.test_colors[4:7])  # Use another subset as palette
        
        for method in methods_to_test:
            with self.subTest(method=method):
                calculator = ColorDistanceCalculator(method)
                
                # Calculate batch distances
                batch_distances = calculator.calculate_distances_batch(colors, palette, show_progress=False)
                
                # Calculate individual distances and compare
                for i, color in enumerate(colors):
                    for j, palette_color in enumerate(palette):
                        single_distance = calculator.calculate_distance(color, palette_color)
                        batch_distance = batch_distances[i, j]
                        
                        self.assertAlmostEqual(
                            single_distance, batch_distance, places=10,
                            msg=f"Batch vs single mismatch in {method} for "
                                f"color {i} -> palette {j}: "
                                f"single={single_distance:.10f}, batch={batch_distance:.10f}"
                        )
    
    def test_color_relationships(self):
        """Test expected relationships between specific colors."""
        methods_to_test = ['cie76', 'cie94', 'ciede2000', 'oklab', 'hsv']
        
        red = np.array([255, 0, 0])
        green = np.array([0, 255, 0]) 
        blue = np.array([0, 0, 255])
        pink = np.array([255, 128, 128])  # Light red
        black = np.array([0, 0, 0])
        white = np.array([255, 255, 255])
        
        for method in methods_to_test:
            with self.subTest(method=method):
                calculator = ColorDistanceCalculator(method)
                
                # Red should be closer to pink than to green
                red_pink_dist = calculator.calculate_distance(red, pink)
                red_green_dist = calculator.calculate_distance(red, green)
                self.assertLess(red_pink_dist, red_green_dist,
                               f"{method}: Red should be closer to pink than green")
                
                # Complementary colors should have larger distances than similar colors
                red_green_dist = calculator.calculate_distance(red, green)
                red_pink_dist = calculator.calculate_distance(red, pink)
                self.assertGreater(red_green_dist, red_pink_dist,
                                 f"{method}: Complementary colors should be more distant")
                
                # Black and white should be far apart for most methods
                if method != 'hsv':  # HSV might have different behavior for grayscale
                    black_white_dist = calculator.calculate_distance(black, white)
                    red_pink_dist = calculator.calculate_distance(red, pink)
                    self.assertGreater(black_white_dist, red_pink_dist,
                                     f"{method}: Black-white should be far apart")
    
    def test_cie_algorithms_lab_consistency(self):
        """Test that CIE algorithms produce reasonable LAB-space results."""
        # Test that CIE76 gives expected results for known LAB differences
        calculator = ColorDistanceCalculator('cie76')
        
        # Test case: Two colors that should have a specific LAB distance
        # Using red and a slightly different red
        red = np.array([255, 0, 0])
        dark_red = np.array([200, 0, 0])
        
        distance = calculator.calculate_distance(red, dark_red)
        
        # Should be a reasonable LAB distance (not zero, not huge)
        self.assertGreater(distance, 0.1, "CIE76 should detect difference between red shades")
        self.assertLess(distance, 100.0, "CIE76 distance should be reasonable for similar reds")
        
        # Test CIE94 vs CIE76 - they should give different results but same ordering
        calc76 = ColorDistanceCalculator('cie76')
        calc94 = ColorDistanceCalculator('cie94')
        
        colors_pairs = [
            (np.array([255, 0, 0]), np.array([0, 255, 0])),   # Red to green
            (np.array([255, 0, 0]), np.array([255, 128, 128])), # Red to pink
        ]
        
        for color1, color2 in colors_pairs:
            dist76 = calc76.calculate_distance(color1, color2)
            dist94 = calc94.calculate_distance(color1, color2)
            
            # Both should be positive
            self.assertGreater(dist76, 0)
            self.assertGreater(dist94, 0)
            
        # Test ordering consistency - red to pink should be less than red to green for both
        red_pink_76 = calc76.calculate_distance(colors_pairs[1][0], colors_pairs[1][1])
        red_green_76 = calc76.calculate_distance(colors_pairs[0][0], colors_pairs[0][1])
        red_pink_94 = calc94.calculate_distance(colors_pairs[1][0], colors_pairs[1][1])
        red_green_94 = calc94.calculate_distance(colors_pairs[0][0], colors_pairs[0][1])
        
        self.assertLess(red_pink_76, red_green_76, "CIE76: Red-pink should be less than red-green")
        self.assertLess(red_pink_94, red_green_94, "CIE94: Red-pink should be less than red-green")
    
    def test_ciede2000_reasonableness(self):
        """Test CIEDE2000 produces reasonable results."""
        calculator = ColorDistanceCalculator('ciede2000')
        
        # Test with some known challenging cases for CIEDE2000
        colors_to_test = [
            (np.array([255, 0, 0]), np.array([255, 0, 0])),     # Identical - should be 0
            (np.array([255, 0, 0]), np.array([0, 255, 0])),     # Red to green - should be large
            (np.array([128, 128, 128]), np.array([129, 129, 129])), # Nearly identical grays
        ]
        
        for color1, color2 in colors_to_test:
            distance = calculator.calculate_distance(color1, color2)
            self.assertGreaterEqual(distance, 0.0, "CIEDE2000 distance should be non-negative")
            
            if np.array_equal(color1, color2):
                self.assertAlmostEqual(distance, 0.0, places=10, 
                                     msg="CIEDE2000 distance should be 0 for identical colors")
    
    def test_oklab_properties(self):
        """Test Oklab implementation properties."""
        calculator = ColorDistanceCalculator('oklab')
        
        # Test some basic properties
        red = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        blue = np.array([0, 0, 255])
        
        # All distances should be positive
        rg_dist = calculator.calculate_distance(red, green)
        rb_dist = calculator.calculate_distance(red, blue) 
        gb_dist = calculator.calculate_distance(green, blue)
        
        self.assertGreater(rg_dist, 0, "Red-green Oklab distance should be positive")
        self.assertGreater(rb_dist, 0, "Red-blue Oklab distance should be positive") 
        self.assertGreater(gb_dist, 0, "Green-blue Oklab distance should be positive")
        
        # Test that Oklab handles grayscale properly
        black = np.array([0, 0, 0])
        white = np.array([255, 255, 255])
        gray = np.array([128, 128, 128])
        
        bw_dist = calculator.calculate_distance(black, white)
        bg_dist = calculator.calculate_distance(black, gray)
        gw_dist = calculator.calculate_distance(gray, white)
        
        # Gray should be between black and white in distance relationships
        self.assertGreater(bw_dist, bg_dist, "Black-white should be further than black-gray")
        self.assertGreater(bw_dist, gw_dist, "Black-white should be further than gray-white")
    
    def test_hsv_properties(self):
        """Test HSV implementation properties.""" 
        calculator = ColorDistanceCalculator('hsv')
        
        # Test hue circularity is handled properly
        red1 = np.array([255, 0, 0])      # Hue ~0
        red2 = np.array([255, 0, 50])     # Slightly different red
        
        # Should have small distance since both are red-ish
        distance = calculator.calculate_distance(red1, red2)
        self.assertGreater(distance, 0, "HSV should detect small differences")
        self.assertLess(distance, 2.0, "HSV distance for similar reds should be reasonable")
        
        # Test saturation differences
        saturated_red = np.array([255, 0, 0])
        desaturated_red = np.array([255, 128, 128])  # Pink
        
        sat_dist = calculator.calculate_distance(saturated_red, desaturated_red)
        self.assertGreater(sat_dist, 0, "HSV should detect saturation differences")
    
    def test_excluded_algorithms_work(self):
        """Test that RGB, weighted_rgb, and compuphase still work correctly."""
        # These are excluded from external validation but should still work
        methods = ['rgb', 'weighted_rgb', 'compuphase']
        
        red = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        
        for method in methods:
            with self.subTest(method=method):
                calculator = ColorDistanceCalculator(method)
                
                # Basic functionality test
                distance = calculator.calculate_distance(red, green)
                self.assertGreater(distance, 0, f"{method} should detect red-green difference")
                
                # Self-distance should be zero
                self_dist = calculator.calculate_distance(red, red)
                self.assertAlmostEqual(self_dist, 0.0, places=10, 
                                     msg=f"{method} self-distance should be zero")
                
                # Batch consistency
                colors = np.array([red, green])
                palette = np.array([red])
                batch_distances = calculator.calculate_distances_batch(colors, palette, show_progress=False)
                
                single_distances = [calculator.calculate_distance(c, palette[0]) for c in colors]
                
                for i, single_dist in enumerate(single_distances):
                    self.assertAlmostEqual(single_dist, batch_distances[i, 0], places=10,
                                         msg=f"{method} batch inconsistency")
    
    def test_cam16_ucs_properties(self):
        """Test CAM16-UCS specific properties."""
        calculator = ColorDistanceCalculator('cam16_ucs')
        
        # Test basic functionality
        red = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        blue = np.array([0, 0, 255])
        
        # Distance should be non-negative
        distance = calculator.calculate_distance(red, green)
        self.assertGreaterEqual(distance, 0)
        
        # Distance to self should be zero  
        self.assertAlmostEqual(calculator.calculate_distance(red, red), 0, places=10)
        
        # Triangle inequality (approximately - may not be exact due to CAM16-UCS properties)
        d_rg = calculator.calculate_distance(red, green)
        d_rb = calculator.calculate_distance(red, blue)
        d_gb = calculator.calculate_distance(green, blue)
        
        # Check triangle inequality holds (with some tolerance for numerical precision)
        self.assertLessEqual(d_rg, d_rb + d_gb + 1e-10)
        self.assertLessEqual(d_rb, d_rg + d_gb + 1e-10)
        self.assertLessEqual(d_gb, d_rg + d_rb + 1e-10)
        
        # Test batch consistency
        colors = np.array([red, green, blue])
        palette = np.array([red])
        batch_distances = calculator.calculate_distances_batch(colors, palette, show_progress=False)
        
        for i, color in enumerate(colors):
            single_distance = calculator.calculate_distance(color, palette[0])
            self.assertAlmostEqual(single_distance, batch_distances[i, 0], places=10)


if __name__ == '__main__':
    unittest.main()