"""Color distance calculation methods for palette matching."""

import numpy as np
from typing import Tuple, Union
import colorspacious
from concurrent.futures import ThreadPoolExecutor
import math


class ColorDistanceCalculator:
    """Calculate color distances using various methods."""
    
    METHODS = [
        'rgb', 'weighted_rgb', 'cie76', 'cie94', 'ciede2000', 'oklab', 'hsv'
    ]
    
    def __init__(self, method: str = 'weighted_rgb'):
        """Initialize with specified distance method.
        
        Args:
            method: Color distance method to use
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method {method}. Available: {self.METHODS}")
        self.method = method
        
        # Cache for pre-converted palette colors
        self._palette_cache = {}
        self._current_palette = None
        
    def calculate_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate distance between two colors.
        
        Args:
            color1: RGB color as [R, G, B] in 0-255 range
            color2: RGB color as [R, G, B] in 0-255 range
            
        Returns:
            Color distance value
        """
        if self.method == 'rgb':
            return self._rgb_distance(color1, color2)
        elif self.method == 'weighted_rgb':
            return self._weighted_rgb_distance(color1, color2)
        elif self.method == 'cie76':
            return self._cie76_distance(color1, color2)
        elif self.method == 'cie94':
            return self._cie94_distance(color1, color2)
        elif self.method == 'ciede2000':
            return self._ciede2000_distance(color1, color2)
        elif self.method == 'oklab':
            return self._oklab_distance(color1, color2)
        elif self.method == 'hsv':
            return self._hsv_distance(color1, color2)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def calculate_distances_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Calculate distances between colors and entire palette efficiently.
        
        Args:
            colors: Array of colors shape (N, 3) in 0-255 range
            palette: Array of palette colors shape (M, 3) in 0-255 range
            
        Returns:
            Distance matrix shape (N, M)
        """
        # Cache palette conversion for expensive methods
        palette_id = id(palette)
        if self._current_palette != palette_id:
            self._palette_cache.clear()
            self._current_palette = palette_id
        
        if self.method == 'rgb':
            return self._rgb_distance_batch(colors, palette)
        elif self.method == 'weighted_rgb':
            return self._weighted_rgb_distance_batch(colors, palette)
        elif self.method == 'cie76':
            return self._cie76_distance_batch(colors, palette)
        elif self.method == 'cie94':
            return self._cie94_distance_batch(colors, palette)
        elif self.method == 'ciede2000':
            return self._ciede2000_distance_batch(colors, palette)
        elif self.method == 'oklab':
            return self._oklab_distance_batch(colors, palette)
        elif self.method == 'hsv':
            return self._hsv_distance_batch(colors, palette)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _rgb_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Standard RGB Euclidean distance."""
        diff = color1.astype(np.float32) - color2.astype(np.float32)
        return np.sqrt(np.sum(diff ** 2))
    
    def _rgb_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch RGB distance calculation."""
        colors = colors.astype(np.float32)
        palette = palette.astype(np.float32)
        
        # Reshape for broadcasting: colors (N, 1, 3), palette (1, M, 3)
        colors_expanded = colors[:, np.newaxis, :]
        palette_expanded = palette[np.newaxis, :, :]
        
        diff = colors_expanded - palette_expanded
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances
    
    def _weighted_rgb_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Weighted RGB distance with perceptual weights."""
        # Common perceptual weights for RGB
        weights = np.array([0.3, 0.59, 0.11])
        diff = color1.astype(np.float32) - color2.astype(np.float32)
        weighted_diff = diff * weights
        return np.sqrt(np.sum(weighted_diff ** 2))
    
    def _weighted_rgb_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch weighted RGB distance calculation."""
        weights = np.array([0.3, 0.59, 0.11])
        colors = colors.astype(np.float32)
        palette = palette.astype(np.float32)
        
        colors_expanded = colors[:, np.newaxis, :]
        palette_expanded = palette[np.newaxis, :, :]
        
        diff = colors_expanded - palette_expanded
        weighted_diff = diff * weights
        distances = np.sqrt(np.sum(weighted_diff ** 2, axis=2))
        return distances
    
    def _rgb_to_lab_batch(self, rgb_batch: np.ndarray) -> np.ndarray:
        """Convert batch of RGB colors to LAB color space efficiently.
        
        Args:
            rgb_batch: Array of RGB colors shape (N, 3) in 0-255 range
            
        Returns:
            Array of LAB colors shape (N, 3)
        """
        # Normalize RGB to 0-1 range
        rgb_normalized = rgb_batch / 255.0
        
        # Convert using colorspacious in batch mode
        try:
            lab_batch = colorspacious.cspace_convert(rgb_normalized, "sRGB1", "CIELab")
            return lab_batch
        except:
            # Fallback to individual conversions if batch fails
            return np.array([self._rgb_to_lab(color) for color in rgb_batch])
    
    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space."""
        # Normalize RGB to 0-1 range
        rgb_normalized = rgb / 255.0
        # Convert using colorspacious
        lab = colorspacious.cspace_convert(rgb_normalized, "sRGB1", "CIELab")
        return lab
    
    def _cie76_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """CIE76 Delta E distance."""
        lab1 = self._rgb_to_lab(color1)
        lab2 = self._rgb_to_lab(color2)
        diff = lab1 - lab2
        return np.sqrt(np.sum(diff ** 2))
    
    def _cie76_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch CIE76 distance calculation."""
        # Check cache for palette LAB conversion
        if 'cie76_palette' not in self._palette_cache:
            self._palette_cache['cie76_palette'] = self._rgb_to_lab_batch(palette)
        
        # Convert colors to LAB
        colors_lab = self._rgb_to_lab_batch(colors)
        palette_lab = self._palette_cache['cie76_palette']
        
        # Calculate distances using broadcasting
        colors_expanded = colors_lab[:, np.newaxis, :]
        palette_expanded = palette_lab[np.newaxis, :, :]
        
        diff = colors_expanded - palette_expanded
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances
    
    def _cie94_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """CIE94 Delta E distance."""
        lab1 = self._rgb_to_lab(color1)
        lab2 = self._rgb_to_lab(color2)
        
        # CIE94 calculation
        delta_l = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]
        
        c1 = np.sqrt(lab1[1]**2 + lab1[2]**2)
        c2 = np.sqrt(lab2[1]**2 + lab2[2]**2)
        delta_c = c1 - c2
        
        delta_h_squared = delta_a**2 + delta_b**2 - delta_c**2
        delta_h = np.sqrt(max(0, delta_h_squared))
        
        # CIE94 constants for graphic arts
        kl = kc = kh = 1.0
        k1 = 0.045
        k2 = 0.015
        
        sl = 1.0
        sc = 1.0 + k1 * c1
        sh = 1.0 + k2 * c1
        
        delta_e = np.sqrt(
            (delta_l / (kl * sl))**2 +
            (delta_c / (kc * sc))**2 +
            (delta_h / (kh * sh))**2
        )
        
        return delta_e
    
    def _cie94_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch CIE94 distance calculation with vectorized operations."""
        # Check cache for palette LAB conversion
        if 'cie94_palette' not in self._palette_cache:
            self._palette_cache['cie94_palette'] = self._rgb_to_lab_batch(palette)
        
        # Convert colors to LAB
        colors_lab = self._rgb_to_lab_batch(colors)
        palette_lab = self._palette_cache['cie94_palette']
        
        # Vectorized CIE94 calculation
        colors_expanded = colors_lab[:, np.newaxis, :]  # (N, 1, 3)
        palette_expanded = palette_lab[np.newaxis, :, :]  # (1, M, 3)
        
        # Calculate differences
        delta_l = colors_expanded[:, :, 0] - palette_expanded[:, :, 0]
        delta_a = colors_expanded[:, :, 1] - palette_expanded[:, :, 1]
        delta_b = colors_expanded[:, :, 2] - palette_expanded[:, :, 2]
        
        # Calculate chroma values
        c1 = np.sqrt(colors_expanded[:, :, 1]**2 + colors_expanded[:, :, 2]**2)
        c2 = np.sqrt(palette_expanded[:, :, 1]**2 + palette_expanded[:, :, 2]**2)
        delta_c = c1 - c2
        
        # Calculate delta H
        delta_h_squared = delta_a**2 + delta_b**2 - delta_c**2
        delta_h = np.sqrt(np.maximum(0, delta_h_squared))
        
        # CIE94 constants for graphic arts
        kl = kc = kh = 1.0
        k1 = 0.045
        k2 = 0.015
        
        # Calculate weighting functions
        sl = 1.0
        sc = 1.0 + k1 * c1
        sh = 1.0 + k2 * c1
        
        # Calculate final CIE94 distance
        delta_e = np.sqrt(
            (delta_l / (kl * sl))**2 +
            (delta_c / (kc * sc))**2 +
            (delta_h / (kh * sh))**2
        )
        
        return delta_e
    
    def _ciede2000_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """CIEDE2000 Delta E distance."""
        lab1 = self._rgb_to_lab(color1)
        lab2 = self._rgb_to_lab(color2)
        
        # Use colorspacious for CIEDE2000 calculation
        try:
            delta_e = colorspacious.deltaE(lab1, lab2, input_space="CIELab")
            return delta_e
        except:
            # Fallback to CIE76 if CIEDE2000 fails
            return self._cie76_distance(color1, color2)
    
    def _ciede2000_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch CIEDE2000 distance calculation with intelligent fallbacks."""
        # Check cache for palette LAB conversion
        if 'ciede2000_palette' not in self._palette_cache:
            self._palette_cache['ciede2000_palette'] = self._rgb_to_lab_batch(palette)
        
        # Convert colors to LAB
        colors_lab = self._rgb_to_lab_batch(colors)
        palette_lab = self._palette_cache['ciede2000_palette']
        
        n_colors = len(colors)
        n_palette = len(palette)
        total_comparisons = n_colors * n_palette
        
        # Intelligent fallback strategy for performance
        if total_comparisons > 5000:  # For larger problems, use CIE94 which is much faster
            print(f"Info: Using CIE94 instead of CIEDE2000 for better performance ({n_colors}x{n_palette} comparisons)")
            return self._cie94_distance_from_lab(colors_lab, palette_lab)
        else:
            # Only use true CIEDE2000 for small problems
            return self._ciede2000_small_batch(colors_lab, palette_lab)
    
    def _ciede2000_small_batch(self, colors_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
        """CIEDE2000 calculation for small batches only."""
        n_colors = len(colors_lab)
        n_palette = len(palette_lab)
        distances = np.zeros((n_colors, n_palette))
        
        for i, color in enumerate(colors_lab):
            for j, pal_color in enumerate(palette_lab):
                try:
                    distances[i, j] = colorspacious.deltaE(color, pal_color, input_space="CIELab")
                except:
                    # Fallback to CIE94
                    distances[i, j] = self._cie94_single_from_lab(color, pal_color)
        
        return distances

    
    def _cie94_single_from_lab(self, lab1: np.ndarray, lab2: np.ndarray) -> float:
        """Calculate CIE94 distance between two LAB colors."""
        delta_l = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]
        
        c1 = math.sqrt(lab1[1]**2 + lab1[2]**2)
        c2 = math.sqrt(lab2[1]**2 + lab2[2]**2)
        delta_c = c1 - c2
        
        delta_h_squared = delta_a**2 + delta_b**2 - delta_c**2
        delta_h = math.sqrt(max(0, delta_h_squared))
        
        # CIE94 constants
        kl = kc = kh = 1.0
        k1 = 0.045
        k2 = 0.015
        
        sl = 1.0
        sc = 1.0 + k1 * c1
        sh = 1.0 + k2 * c1
        
        delta_e = math.sqrt(
            (delta_l / (kl * sl))**2 +
            (delta_c / (kc * sc))**2 +
            (delta_h / (kh * sh))**2
        )
        
        return delta_e
    
    def _cie94_distance_from_lab(self, colors_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
        """Calculate CIE94 distance from pre-converted LAB colors."""
        # Vectorized CIE94 calculation from LAB values
        colors_expanded = colors_lab[:, np.newaxis, :]  # (N, 1, 3)
        palette_expanded = palette_lab[np.newaxis, :, :]  # (1, M, 3)
        
        # Calculate differences
        delta_l = colors_expanded[:, :, 0] - palette_expanded[:, :, 0]
        delta_a = colors_expanded[:, :, 1] - palette_expanded[:, :, 1]
        delta_b = colors_expanded[:, :, 2] - palette_expanded[:, :, 2]
        
        # Calculate chroma values
        c1 = np.sqrt(colors_expanded[:, :, 1]**2 + colors_expanded[:, :, 2]**2)
        c2 = np.sqrt(palette_expanded[:, :, 1]**2 + palette_expanded[:, :, 2]**2)
        delta_c = c1 - c2
        
        # Calculate delta H
        delta_h_squared = delta_a**2 + delta_b**2 - delta_c**2
        delta_h = np.sqrt(np.maximum(0, delta_h_squared))
        
        # CIE94 constants for graphic arts
        kl = kc = kh = 1.0
        k1 = 0.045
        k2 = 0.015
        
        # Calculate weighting functions
        sl = 1.0
        sc = 1.0 + k1 * c1
        sh = 1.0 + k2 * c1
        
        # Calculate final CIE94 distance
        delta_e = np.sqrt(
            (delta_l / (kl * sl))**2 +
            (delta_c / (kc * sc))**2 +
            (delta_h / (kh * sh))**2
        )
        
        return delta_e
    
    def _rgb_to_oklab_batch(self, rgb_batch: np.ndarray) -> np.ndarray:
        """Convert batch of RGB colors to Oklab color space efficiently.
        
        Args:
            rgb_batch: Array of RGB colors shape (N, 3) in 0-255 range
            
        Returns:
            Array of Oklab colors shape (N, 3)
        """
        # Normalize to 0-1
        rgb_normalized = rgb_batch / 255.0
        
        # Convert to linear RGB (vectorized)
        def gamma_to_linear(c):
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        
        linear_rgb = gamma_to_linear(rgb_normalized)
        
        # Matrix multiplication to XYZ-like space
        m1 = np.array([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]
        ])
        
        lms = linear_rgb @ m1.T
        
        # Cube root (vectorized)
        lms_cbrt = np.sign(lms) * np.abs(lms) ** (1/3)
        
        # Second matrix multiplication
        m2 = np.array([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660]
        ])
        
        oklab = lms_cbrt @ m2.T
        return oklab
    
    def _rgb_to_oklab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to Oklab color space."""
        # Use the batch version for single color
        return self._rgb_to_oklab_batch(rgb.reshape(1, 3))[0]
    
    def _oklab_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Oklab color space distance."""
        oklab1 = self._rgb_to_oklab(color1)
        oklab2 = self._rgb_to_oklab(color2)
        diff = oklab1 - oklab2
        return np.sqrt(np.sum(diff ** 2))
    
    def _oklab_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch Oklab distance calculation."""
        # Check cache for palette Oklab conversion
        if 'oklab_palette' not in self._palette_cache:
            self._palette_cache['oklab_palette'] = self._rgb_to_oklab_batch(palette)
        
        # Convert colors to Oklab
        colors_oklab = self._rgb_to_oklab_batch(colors)
        palette_oklab = self._palette_cache['oklab_palette']
        
        colors_expanded = colors_oklab[:, np.newaxis, :]
        palette_expanded = palette_oklab[np.newaxis, :, :]
        
        diff = colors_expanded - palette_expanded
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances
    
    def _rgb_to_hsv_batch(self, rgb_batch: np.ndarray) -> np.ndarray:
        """Convert batch of RGB colors to HSV efficiently.
        
        Args:
            rgb_batch: Array of RGB colors shape (N, 3) in 0-255 range
            
        Returns:
            Array of HSV colors shape (N, 3)
        """
        rgb_normalized = rgb_batch / 255.0
        r, g, b = rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = np.where(max_val == 0, 0, diff / max_val)
        
        # Hue calculation (vectorized)
        h = np.zeros_like(max_val)
        
        # Where diff != 0
        mask_diff = diff != 0
        
        # Where max == r
        mask_r = mask_diff & (max_val == r)
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        
        # Where max == g  
        mask_g = mask_diff & (max_val == g)
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        
        # Where max == b
        mask_b = mask_diff & (max_val == b)
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
        
        return np.column_stack([h, s, v])
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        # Use the batch version for single color
        return self._rgb_to_hsv_batch(rgb.reshape(1, 3))[0]
    
    def _hsv_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """HSV color space distance with circular hue handling."""
        hsv1 = self._rgb_to_hsv(color1)
        hsv2 = self._rgb_to_hsv(color2)
        
        # Handle hue circularity
        h_diff = abs(hsv1[0] - hsv2[0])
        h_diff = min(h_diff, 360 - h_diff)
        
        # Weight components differently
        h_weight = 1.0
        s_weight = 1.0  
        v_weight = 1.0
        
        distance = np.sqrt(
            (h_weight * h_diff / 180) ** 2 +  # Normalize hue to 0-2 range
            (s_weight * (hsv1[1] - hsv2[1])) ** 2 +
            (v_weight * (hsv1[2] - hsv2[2])) ** 2
        )
        
        return distance
    
    def _hsv_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch HSV distance calculation with vectorized operations."""
        # Check cache for palette HSV conversion
        if 'hsv_palette' not in self._palette_cache:
            self._palette_cache['hsv_palette'] = self._rgb_to_hsv_batch(palette)
        
        # Convert colors to HSV
        colors_hsv = self._rgb_to_hsv_batch(colors)
        palette_hsv = self._palette_cache['hsv_palette']
        
        # Expand for broadcasting
        colors_expanded = colors_hsv[:, np.newaxis, :]  # (N, 1, 3)
        palette_expanded = palette_hsv[np.newaxis, :, :]  # (1, M, 3)
        
        # Handle hue circularity (vectorized)
        h_diff = np.abs(colors_expanded[:, :, 0] - palette_expanded[:, :, 0])
        h_diff = np.minimum(h_diff, 360 - h_diff)
        
        # Calculate other component differences
        s_diff = colors_expanded[:, :, 1] - palette_expanded[:, :, 1]
        v_diff = colors_expanded[:, :, 2] - palette_expanded[:, :, 2]
        
        # Weight components
        h_weight = 1.0
        s_weight = 1.0  
        v_weight = 1.0
        
        # Calculate final distance
        distance = np.sqrt(
            (h_weight * h_diff / 180) ** 2 +  # Normalize hue to 0-2 range
            (s_weight * s_diff) ** 2 +
            (v_weight * v_diff) ** 2
        )
        
        return distance