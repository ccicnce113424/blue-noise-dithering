"""Color distance calculation methods for palette matching."""

import numpy as np
from typing import Tuple, Union
import colorspacious
import scipy.ndimage
from tqdm import tqdm


class ColorDistanceCalculator:
    """Calculate color distances using various methods."""
    
    METHODS = [
        'rgb', 'weighted_rgb', 'cie76', 'cie94', 'ciede2000', 'cam16_ucs', 'oklab', 'hsv', 'compuphase'
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
        elif self.method == 'cam16_ucs':
            return self._cam16_ucs_distance(color1, color2)
        elif self.method == 'oklab':
            return self._oklab_distance(color1, color2)
        elif self.method == 'hsv':
            return self._hsv_distance(color1, color2)
        elif self.method == 'compuphase':
            return self._compuphase_distance(color1, color2)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def calculate_distances_batch(self, colors: np.ndarray, palette: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """Calculate distances between colors and entire palette efficiently.
        
        Args:
            colors: Array of colors shape (N, 3) in 0-255 range
            palette: Array of palette colors shape (M, 3) in 0-255 range
            show_progress: Whether to show progress bar for expensive methods
            
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
            return self._ciede2000_distance_batch(colors, palette, show_progress)
        elif self.method == 'cam16_ucs':
            return self._cam16_ucs_distance_batch(colors, palette)
        elif self.method == 'oklab':
            return self._oklab_distance_batch(colors, palette)
        elif self.method == 'hsv':
            return self._hsv_distance_batch(colors, palette)
        elif self.method == 'compuphase':
            return self._compuphase_distance_batch(colors, palette)
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
        """Standard CIEDE2000 Delta E distance following CIE specification."""
        lab1 = self._rgb_to_lab(color1)
        lab2 = self._rgb_to_lab(color2)
        
        # Use standard CIEDE2000 implementation
        return self._ciede2000_standard_single(lab1, lab2)
    
    def _ciede2000_distance_batch(self, colors: np.ndarray, palette: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """Batch standard CIEDE2000 distance calculation following CIE specification."""
        # Check cache for palette LAB conversion
        if 'ciede2000_palette' not in self._palette_cache:
            self._palette_cache['ciede2000_palette'] = self._rgb_to_lab_batch(palette)
        
        # Convert colors to LAB
        colors_lab = self._rgb_to_lab_batch(colors)
        palette_lab = self._palette_cache['ciede2000_palette']
        
        n_colors = len(colors)
        n_palette = len(palette)
        
        if show_progress:
            print(f"Computing standard CIEDE2000 for {n_colors} colors vs {n_palette} palette colors...")
        
        # Use standard CIEDE2000 implementation
        return self._ciede2000_standard_vectorized(colors_lab, palette_lab, show_progress)
    

    

    

    
    def _ciede2000_standard_vectorized(self, colors_lab: np.ndarray, palette_lab: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """Standard CIEDE2000 implementation following CIE specification."""
        n_colors = len(colors_lab)
        n_palette = len(palette_lab)
        
        # Process in chunks for memory efficiency
        chunk_size = min(500, n_colors)  # Smaller chunks for more complex calculation
        distances = np.zeros((n_colors, n_palette))
        
        # Only show progress bar if requested (avoid conflicts with main progress bar)
        with tqdm(total=n_colors, desc="CIEDE2000 Standard", unit="px", disable=not show_progress) as pbar:
            for start_idx in range(0, n_colors, chunk_size):
                end_idx = min(start_idx + chunk_size, n_colors)
                chunk_colors = colors_lab[start_idx:end_idx]
                
                # Standard CIEDE2000 calculation for this chunk
                chunk_distances = self._ciede2000_standard_chunk_vectorized(chunk_colors, palette_lab)
                distances[start_idx:end_idx] = chunk_distances
                
                if show_progress:
                    pbar.update(end_idx - start_idx)
        
        return distances
    
    def _ciede2000_standard_chunk_vectorized(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Full standard CIEDE2000 calculation following CIE specification."""
        # Reshape for broadcasting
        colors_exp = colors[:, np.newaxis, :]  # (n_colors, 1, 3)
        palette_exp = palette[np.newaxis, :, :]  # (1, n_palette, 3)
        
        # Extract L*, a*, b* values
        L1 = colors_exp[:, :, 0]
        a1 = colors_exp[:, :, 1]
        b1 = colors_exp[:, :, 2]
        
        L2 = palette_exp[:, :, 0]
        a2 = palette_exp[:, :, 1]
        b2 = palette_exp[:, :, 2]
        
        # Step 1: Calculate chroma values
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_avg = (C1 + C2) / 2.0
        
        # Step 2: Calculate G factor for chroma adjustment
        G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
        
        # Step 3: Adjust a* values
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        
        # Step 4: Calculate adjusted chroma and hue
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        
        # Hue calculation with proper handling of atan2 edge cases
        h1_prime = np.arctan2(b1, a1_prime) * 180 / np.pi
        h2_prime = np.arctan2(b2, a2_prime) * 180 / np.pi
        
        # Ensure hue values are in [0, 360) range
        h1_prime = np.where(h1_prime < 0, h1_prime + 360, h1_prime)
        h2_prime = np.where(h2_prime < 0, h2_prime + 360, h2_prime)
        
        # Step 5: Calculate differences
        dL_prime = L2 - L1
        dC_prime = C2_prime - C1_prime
        
        # Hue difference calculation following CIE specification
        dh_prime = np.zeros_like(h1_prime)
        
        # Case 1: Either C1' or C2' is zero
        zero_chroma = (C1_prime * C2_prime) == 0
        dh_prime = np.where(zero_chroma, 0, dh_prime)
        
        # Case 2: |h2' - h1'| <= 180
        abs_diff = np.abs(h2_prime - h1_prime)
        case2 = (~zero_chroma) & (abs_diff <= 180)
        dh_prime = np.where(case2, h2_prime - h1_prime, dh_prime)
        
        # Case 3: h2' - h1' > 180
        case3 = (~zero_chroma) & (~case2) & ((h2_prime - h1_prime) > 180)
        dh_prime = np.where(case3, h2_prime - h1_prime - 360, dh_prime)
        
        # Case 4: h2' - h1' < -180
        case4 = (~zero_chroma) & (~case2) & (~case3)
        dh_prime = np.where(case4, h2_prime - h1_prime + 360, dh_prime)
        
        # Calculate dH'
        dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))
        
        # Step 6: Calculate average values
        L_avg = (L1 + L2) / 2.0
        C_prime_avg = (C1_prime + C2_prime) / 2.0
        
        # Average hue calculation
        h_prime_avg = np.zeros_like(h1_prime)
        
        # Case 1: Either C1' or C2' is zero
        h_prime_avg = np.where(zero_chroma, h1_prime + h2_prime, h_prime_avg)
        
        # Case 2: |h1' - h2'| <= 180
        abs_h_diff = np.abs(h1_prime - h2_prime)
        case2_h = (~zero_chroma) & (abs_h_diff <= 180)
        h_prime_avg = np.where(case2_h, (h1_prime + h2_prime) / 2.0, h_prime_avg)
        
        # Case 3: |h1' - h2'| > 180 and h1' + h2' < 360
        case3_h = (~zero_chroma) & (~case2_h) & ((h1_prime + h2_prime) < 360)
        h_prime_avg = np.where(case3_h, (h1_prime + h2_prime + 360) / 2.0, h_prime_avg)
        
        # Case 4: |h1' - h2'| > 180 and h1' + h2' >= 360
        case4_h = (~zero_chroma) & (~case2_h) & (~case3_h)
        h_prime_avg = np.where(case4_h, (h1_prime + h2_prime - 360) / 2.0, h_prime_avg)
        
        # Step 7: Calculate weighting functions
        T = (1 - 0.17 * np.cos(np.radians(h_prime_avg - 30)) +
             0.24 * np.cos(np.radians(2 * h_prime_avg)) +
             0.32 * np.cos(np.radians(3 * h_prime_avg + 6)) -
             0.20 * np.cos(np.radians(4 * h_prime_avg - 63)))
        
        delta_theta = 30 * np.exp(-((h_prime_avg - 275) / 25)**2)
        
        R_C = 2 * np.sqrt(C_prime_avg**7 / (C_prime_avg**7 + 25**7))
        
        S_L = 1 + (0.015 * (L_avg - 50)**2) / np.sqrt(20 + (L_avg - 50)**2)
        S_C = 1 + 0.045 * C_prime_avg
        S_H = 1 + 0.015 * C_prime_avg * T
        
        R_T = -np.sin(2 * np.radians(delta_theta)) * R_C
        
        # Step 8: Calculate final CIEDE2000 distance
        # Parametric factors (usually kL = kC = kH = 1 for graphic arts)
        kL = kC = kH = 1.0
        
        delta_E00 = np.sqrt(
            (dL_prime / (kL * S_L))**2 +
            (dC_prime / (kC * S_C))**2 +
            (dH_prime / (kH * S_H))**2 +
            R_T * (dC_prime / (kC * S_C)) * (dH_prime / (kH * S_H))
        )
        
        return delta_E00
    
    def _ciede2000_standard_single(self, lab1: np.ndarray, lab2: np.ndarray) -> float:
        """Standard CIEDE2000 calculation for single color pair."""
        # Use the vectorized version for single pair
        lab1_batch = lab1.reshape(1, 3)
        lab2_batch = lab2.reshape(1, 3)
        result = self._ciede2000_standard_chunk_vectorized(lab1_batch, lab2_batch)
        return float(result[0, 0])
    
    def _rgb_to_xyz_batch(self, rgb_batch: np.ndarray) -> np.ndarray:
        """Convert batch of RGB colors to XYZ color space efficiently.
        
        Args:
            rgb_batch: Array of RGB colors shape (N, 3) in 0-255 range
            
        Returns:
            Array of XYZ colors shape (N, 3)
        """
        # Normalize to 0-1 range
        rgb_normalized = rgb_batch / 255.0
        
        # Convert to linear RGB (vectorized)
        def gamma_to_linear(c):
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        
        linear_rgb = gamma_to_linear(rgb_normalized)
        
        # sRGB to XYZ conversion matrix (D65 illuminant)
        m = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = linear_rgb @ m.T
        return xyz
    
    def _rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to XYZ color space."""
        # Use the batch version for single color
        return self._rgb_to_xyz_batch(rgb.reshape(1, 3))[0]
    
    def _xyz_to_cam16_batch(self, xyz_batch: np.ndarray) -> np.ndarray:
        """Convert batch of XYZ colors to CAM16 color appearance coordinates.
        
        Args:
            xyz_batch: Array of XYZ colors shape (N, 3)
            
        Returns:
            Array of CAM16 coordinates shape (N, 3) as [J, C, h]
        """
        # CAM16 viewing conditions (typical indoor conditions)
        # D65 white point
        Xw, Yw, Zw = 95.047, 100.000, 108.883
        # Luminance of white point (cd/m²)
        Lw = 100.0
        # Luminance of background (cd/m²)  
        Yb = 20.0
        # Surround conditions: 1=average, 0.9=dim, 0.8=dark
        c = 0.69  # impact of surround
        Nc = 1.0  # chromatic induction factor
        
        # Viewing flare
        F = 1.0  # maximum flare
        
        # Calculate adapting luminance
        LA = Lw * Yb / 100.0
        
        # Calculate degree of adaptation
        k = 1.0 / (5.0 * LA + 1.0)
        FL = 0.2 * (k ** 4) * (5.0 * LA) + 0.1 * ((1.0 - k ** 4) ** 2) * ((5.0 * LA) ** (1.0/3.0))
        
        # Ensure FL is positive
        FL = np.maximum(FL, 0.01)
        
        n = Yb / Yw
        z = 1.48 + np.sqrt(n)
        Nbb = 0.725 * (1.0 / n) ** 0.2
        Ncb = Nbb
        
        # CAM16 forward transformation
        # Step 1: Convert to cone responses
        # CAT16 transformation matrix
        MCAT16 = np.array([
            [ 0.401288, 0.650173, -0.051461],
            [-0.250268, 1.204414,  0.045854], 
            [-0.002079, 0.048952,  0.953127]
        ])
        
        # Convert XYZ to CAT16 space
        # Normalize XYZ (assuming input is already in 0-100 range for Y)
        xyz_normalized = xyz_batch * np.array([Xw/100.0, Yw/100.0, Zw/100.0])
        rgb_cat16 = xyz_normalized @ MCAT16.T
        
        # Step 2: Calculate degree of adaptation
        # White point in CAT16
        rgb_w = np.array([Xw, Yw, Zw]) @ MCAT16.T
        
        # Adapted cone responses
        D = F * (1.0 - (1.0/3.6) * np.exp((-LA - 42.0) / 92.0))
        D = np.clip(D, 0.0, 1.0)
        
        # Adapted white point
        Rw_c = D * (Yw / rgb_w[0]) + (1.0 - D)
        Gw_c = D * (Yw / rgb_w[1]) + (1.0 - D)  
        Bw_c = D * (Yw / rgb_w[2]) + (1.0 - D)
        
        # Adapted cone responses for sample
        R_c = D * (Yw / rgb_w[0]) * rgb_cat16[:, 0] / rgb_w[0] + (1.0 - D) * rgb_cat16[:, 0] / rgb_w[0]
        G_c = D * (Yw / rgb_w[1]) * rgb_cat16[:, 1] / rgb_w[1] + (1.0 - D) * rgb_cat16[:, 1] / rgb_w[1]
        B_c = D * (Yw / rgb_w[2]) * rgb_cat16[:, 2] / rgb_w[2] + (1.0 - D) * rgb_cat16[:, 2] / rgb_w[2]
        
        # Step 3: Convert to Hunt-Pointer-Estevez space
        MHPE = np.array([
            [ 0.38971, 0.68898, -0.07868],
            [-0.22981, 1.18340,  0.04641],
            [ 0.00000, 0.00000,  1.00000]
        ])
        
        # Transform adapted cone responses
        adapted_rgb = np.column_stack([R_c, G_c, B_c])
        rgb_prime = adapted_rgb @ MHPE.T
        
        # Step 4: Apply nonlinearity
        def f_nonlinear(x):
            return np.where(x >= 0, 
                           np.sign(x) * (400.0 * (FL * np.abs(x) / 100.0) ** 0.42) / (27.13 + (FL * np.abs(x) / 100.0) ** 0.42) + 0.1,
                           np.sign(x) * (400.0 * (FL * np.abs(x) / 100.0) ** 0.42) / (27.13 + (FL * np.abs(x) / 100.0) ** 0.42) + 0.1)
        
        R_a = f_nonlinear(rgb_prime[:, 0])
        G_a = f_nonlinear(rgb_prime[:, 1]) 
        B_a = f_nonlinear(rgb_prime[:, 2])
        
        # Step 5: Calculate achromatic response
        A = (2.0 * R_a + G_a + (1.0/20.0) * B_a - 0.305) * Nbb
        
        # Step 6: Calculate lightness
        J = 100.0 * (A / (1.64 - 0.29 ** n)) ** (c * z)
        
        # Step 7: Calculate chroma
        # Red-green and yellow-blue opponent responses
        a = R_a - 12.0 * G_a / 11.0 + B_a / 11.0
        b = (1.0/9.0) * (R_a + G_a - 2.0 * B_a)
        
        # Hue angle
        h = np.arctan2(b, a) * 180.0 / np.pi
        h = np.where(h < 0, h + 360.0, h)
        
        # Chroma
        C = 50.0 * np.sqrt(a**2 + b**2) * (1.64 - 0.29 ** n) ** 0.73 / (n ** 0.2 * (1.64 - 0.29 ** n) ** 0.73)
        
        return np.column_stack([J, C, h])
    
    def _xyz_to_cam16(self, xyz: np.ndarray) -> np.ndarray:
        """Convert XYZ to CAM16 color appearance coordinates."""
        # Use the batch version for single color
        return self._xyz_to_cam16_batch(xyz.reshape(1, 3))[0]
    
    def _cam16_to_ucs_batch(self, cam16_batch: np.ndarray) -> np.ndarray:
        """Convert batch of CAM16 coordinates to CAM16-UCS coordinates.
        
        Args:
            cam16_batch: Array of CAM16 coordinates shape (N, 3) as [J, C, h]
            
        Returns:
            Array of CAM16-UCS coordinates shape (N, 3) as [Jab_L, Jab_a, Jab_b]
        """
        J = cam16_batch[:, 0]
        C = cam16_batch[:, 1] 
        h = cam16_batch[:, 2]
        
        # Convert to radians
        h_rad = h * np.pi / 180.0
        
        # CAM16-UCS transformation
        # Lightness
        Jab_L = 1.7 * J / (1.0 + 0.007 * J)
        
        # Chroma components
        Jab_C = (1.0 + 0.0228 * C) * C / (1.0 + 0.0228 * C)
        
        # Cartesian coordinates
        Jab_a = Jab_C * np.cos(h_rad)
        Jab_b = Jab_C * np.sin(h_rad)
        
        return np.column_stack([Jab_L, Jab_a, Jab_b])
    
    def _cam16_to_ucs(self, cam16: np.ndarray) -> np.ndarray:
        """Convert CAM16 coordinates to CAM16-UCS coordinates."""
        # Use the batch version for single color
        return self._cam16_to_ucs_batch(cam16.reshape(1, 3))[0]
    
    def _cam16_ucs_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """CAM16-UCS color difference calculation."""
        # Convert RGB to XYZ
        xyz1 = self._rgb_to_xyz(color1)
        xyz2 = self._rgb_to_xyz(color2)
        
        # Convert XYZ to CAM16
        cam16_1 = self._xyz_to_cam16(xyz1)
        cam16_2 = self._xyz_to_cam16(xyz2)
        
        # Convert CAM16 to UCS
        ucs1 = self._cam16_to_ucs(cam16_1)
        ucs2 = self._cam16_to_ucs(cam16_2)
        
        # Calculate Euclidean distance in UCS space
        diff = ucs1 - ucs2
        return np.sqrt(np.sum(diff ** 2))
    
    def _cam16_ucs_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch CAM16-UCS distance calculation."""
        # Check cache for palette UCS conversion
        if 'cam16_ucs_palette' not in self._palette_cache:
            # Convert palette to UCS
            xyz_palette = self._rgb_to_xyz_batch(palette)
            cam16_palette = self._xyz_to_cam16_batch(xyz_palette)
            self._palette_cache['cam16_ucs_palette'] = self._cam16_to_ucs_batch(cam16_palette)
        
        # Convert colors to UCS
        xyz_colors = self._rgb_to_xyz_batch(colors)
        cam16_colors = self._xyz_to_cam16_batch(xyz_colors)
        ucs_colors = self._cam16_to_ucs_batch(cam16_colors)
        ucs_palette = self._palette_cache['cam16_ucs_palette']
        
        # Calculate distances using broadcasting
        colors_expanded = ucs_colors[:, np.newaxis, :]
        palette_expanded = ucs_palette[np.newaxis, :, :]
        
        diff = colors_expanded - palette_expanded
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances
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
        s = np.divide(diff, max_val, out=np.zeros_like(max_val), where=max_val != 0)
        
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
    
    def _compuphase_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Compuphase low-cost color distance approximation.
        
        Based on the algorithm from https://www.compuphase.com/cmetric.htm
        This is a fast approximation that takes into account human color perception
        by weighting colors based on the average red value.
        """
        # Convert to int to match the original C implementation
        r1, g1, b1 = color1.astype(np.int32)
        r2, g2, b2 = color2.astype(np.int32)
        
        # Calculate mean red value
        rmean = (r1 + r2) // 2
        
        # Calculate color differences
        dr = r1 - r2
        dg = g1 - g2
        db = b1 - b2
        
        # Apply the compuphase formula with bit shifts
        # (512+rmean)*dr*dr >> 8 + 4*dg*dg + (767-rmean)*db*db >> 8
        red_term = ((512 + rmean) * dr * dr) >> 8
        green_term = 4 * dg * dg
        blue_term = ((767 - rmean) * db * db) >> 8
        
        distance_squared = red_term + green_term + blue_term
        return np.sqrt(distance_squared)
    
    def _compuphase_distance_batch(self, colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Batch compuphase distance calculation with vectorized operations."""
        # Convert to int32 for calculations
        colors = colors.astype(np.int32)
        palette = palette.astype(np.int32)
        
        # Reshape for broadcasting: colors (N, 1, 3), palette (1, M, 3)
        colors_expanded = colors[:, np.newaxis, :]  # (N, 1, 3)
        palette_expanded = palette[np.newaxis, :, :]  # (1, M, 3)
        
        # Extract color components
        r1 = colors_expanded[:, :, 0]
        g1 = colors_expanded[:, :, 1] 
        b1 = colors_expanded[:, :, 2]
        
        r2 = palette_expanded[:, :, 0]
        g2 = palette_expanded[:, :, 1]
        b2 = palette_expanded[:, :, 2]
        
        # Calculate mean red values (vectorized)
        rmean = (r1 + r2) // 2
        
        # Calculate color differences (vectorized)
        dr = r1 - r2
        dg = g1 - g2
        db = b1 - b2
        
        # Apply the compuphase formula with bit shifts (vectorized)
        red_term = ((512 + rmean) * dr * dr) >> 8
        green_term = 4 * dg * dg
        blue_term = ((767 - rmean) * db * db) >> 8
        
        distance_squared = red_term + green_term + blue_term
        distances = np.sqrt(distance_squared.astype(np.float64))
        
        return distances