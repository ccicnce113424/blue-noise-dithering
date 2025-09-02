"""Core blue noise dithering implementation."""

import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional, Callable
from tqdm import tqdm
import scipy.ndimage
from skimage import filters, feature

from .color_distance import ColorDistanceCalculator
from .palette import PaletteLoader


class BlueNoiseDitherer:
    """Main blue noise dithering processor."""
    
    ALPHA_METHODS = ['threshold', 'dithering']
    ADAPTIVE_STRATEGIES = ['uniform', 'gradient', 'edge', 'contrast', 
                          'gradient_edge', 'gradient_contrast', 'edge_contrast', 'all']
    
    def __init__(self, 
                 color_distance_method: str = 'weighted_rgb',
                 noise_strength: float = 0.5,
                 adaptive_noise: bool = False,
                 adaptive_strategy: str = 'gradient',
                 alpha_method: str = 'threshold',
                 alpha_threshold: float = 0.5,
                 output_noise_map: Optional[str] = None):
        """Initialize the ditherer.
        
        Args:
            color_distance_method: Method for color distance calculation
            noise_strength: Base noise strength (0.0 to 1.0)
            adaptive_noise: Whether to use adaptive noise strength
            adaptive_strategy: Strategy for adaptive noise ('uniform', 'gradient', 'edge', 'contrast', 
                             'gradient_edge', 'gradient_contrast', 'edge_contrast', 'all')
            alpha_method: Method for alpha handling ('threshold' or 'dithering')
            alpha_threshold: Threshold for alpha processing (0.0 to 1.0)
            output_noise_map: Optional path to save noise strength map as image
        """
        self.color_calculator = ColorDistanceCalculator(color_distance_method)
        self.noise_strength = np.clip(noise_strength, 0.0, 1.0)
        self.adaptive_noise = adaptive_noise
        self.adaptive_strategy = adaptive_strategy
        self.alpha_method = alpha_method
        self.alpha_threshold = np.clip(alpha_threshold, 0.0, 1.0)
        self.output_noise_map = output_noise_map
        
        if alpha_method not in self.ALPHA_METHODS:
            raise ValueError(f"Unknown alpha method: {alpha_method}. Available: {self.ALPHA_METHODS}")
        
        if adaptive_strategy not in self.ADAPTIVE_STRATEGIES:
            raise ValueError(f"Unknown adaptive strategy: {adaptive_strategy}. Available: {self.ADAPTIVE_STRATEGIES}")
        
        self.blue_noise_texture = None
        self.palette = None
    
    def load_blue_noise_texture(self, filepath: str) -> None:
        """Load blue noise texture from file.
        
        Args:
            filepath: Path to grayscale blue noise texture image
        """
        try:
            image = Image.open(filepath).convert('L')  # Convert to grayscale
            self.blue_noise_texture = np.array(image, dtype=np.float32) / 255.0
            print(f"Loaded blue noise texture: {self.blue_noise_texture.shape}")
        except Exception as e:
            raise ValueError(f"Error loading blue noise texture: {e}")
    
    def load_palette(self, palette_loader: PaletteLoader) -> None:
        """Load color palette.
        
        Args:
            palette_loader: Loaded palette instance
        """
        self.palette = palette_loader.get_colors()
        print(f"Loaded palette with {len(self.palette)} colors")
    
    def dither_image(self, 
                    input_image: Union[str, Image.Image, np.ndarray],
                    progress_callback: Optional[Callable[[float], None]] = None) -> Image.Image:
        """Dither an image using blue noise and the loaded palette.
        
        Args:
            input_image: Input image (file path, PIL Image, or numpy array)
            progress_callback: Optional callback for progress reporting
            
        Returns:
            Dithered PIL Image
        """
        if self.blue_noise_texture is None:
            raise ValueError("Blue noise texture not loaded")
        if self.palette is None:
            raise ValueError("Palette not loaded")
        
        # Load and prepare input image
        if isinstance(input_image, str):
            image = Image.open(input_image).convert('RGBA')
        elif isinstance(input_image, Image.Image):
            image = input_image.convert('RGBA')
        elif isinstance(input_image, np.ndarray):
            if input_image.shape[2] == 3:
                # Add alpha channel
                alpha = np.full(input_image.shape[:2] + (1,), 255, dtype=input_image.dtype)
                input_image = np.concatenate([input_image, alpha], axis=2)
            image = Image.fromarray(input_image, 'RGBA')
        else:
            raise ValueError("Invalid input image type")
        
        image_array = np.array(image, dtype=np.float32)
        height, width = image_array.shape[:2]
        
        print(f"Processing image: {width}x{height}")
        
        # Prepare output image
        output_array = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Tile blue noise texture to match image size
        noise_texture = self._tile_blue_noise(width, height)
        
        # Calculate adaptive noise strength if enabled
        if self.adaptive_noise:
            noise_strength_map = self._calculate_adaptive_noise(image_array[:, :, :3])
        else:
            noise_strength_map = np.full((height, width), self.noise_strength)
        
        # Save noise strength map if requested
        if self.output_noise_map:
            self._save_noise_strength_map(noise_strength_map, self.output_noise_map)
        
        # Process pixels in chunks for memory efficiency
        chunk_size = 10000  # Increased from 1000 to 10000 pixels at a time for better performance
        total_pixels = height * width
        processed_pixels = 0
        
        # Process pixels in chunks for memory efficiency and better performance
        chunk_size = 10000  # Process 10000 pixels at a time for better performance
        total_pixels = height * width
        processed_pixels = 0
        
        # Flatten arrays for chunk processing
        flat_rgb = image_array[:, :, :3].reshape(-1, 3)
        flat_alpha = image_array[:, :, 3].reshape(-1)
        flat_noise = noise_texture.reshape(-1)
        flat_strength = noise_strength_map.reshape(-1)
        
        output_rgb = np.zeros((total_pixels, 3), dtype=np.uint8)
        output_alpha = np.zeros(total_pixels, dtype=np.uint8)
        
        with tqdm(total=total_pixels, desc="Dithering", unit="px", 
                 disable=progress_callback is None) as pbar:
            
            for start_idx in range(0, total_pixels, chunk_size):
                end_idx = min(start_idx + chunk_size, total_pixels)
                chunk_size_actual = end_idx - start_idx
                
                # Extract chunk data
                chunk_rgb = flat_rgb[start_idx:end_idx]
                chunk_alpha = flat_alpha[start_idx:end_idx]
                chunk_noise = flat_noise[start_idx:end_idx]
                chunk_strength = flat_strength[start_idx:end_idx]
                
                # Apply blue noise to RGB
                noise_offset = (chunk_noise - 0.5) * 2.0  # Convert to -1 to 1 range
                noise_scale = chunk_strength.reshape(-1, 1) * 50.0  # Scale noise
                
                noisy_rgb = chunk_rgb + noise_offset.reshape(-1, 1) * noise_scale
                noisy_rgb = np.clip(noisy_rgb, 0, 255)
                
                # Find closest palette colors
                palette_indices = self._find_closest_colors(noisy_rgb, show_progress=False)
                chunk_output_rgb = self.palette[palette_indices]
                
                # Handle alpha channel
                if self.alpha_method == 'threshold':
                    chunk_output_alpha = self._apply_alpha_threshold(chunk_alpha)
                else:  # dithering
                    chunk_output_alpha = self._apply_alpha_dithering(
                        chunk_alpha, chunk_noise, chunk_strength
                    )
                
                # Store results
                output_rgb[start_idx:end_idx] = chunk_output_rgb
                output_alpha[start_idx:end_idx] = chunk_output_alpha
                
                processed_pixels += chunk_size_actual
                
                if progress_callback:
                    progress_callback(processed_pixels / total_pixels)
                
                pbar.update(chunk_size_actual)
        
        # Reshape back to image dimensions
        output_array[:, :, :3] = output_rgb.reshape(height, width, 3)
        output_array[:, :, 3] = output_alpha.reshape(height, width)
        
        return Image.fromarray(output_array, 'RGBA')
    
    def _tile_blue_noise(self, width: int, height: int) -> np.ndarray:
        """Tile blue noise texture to cover the image dimensions.
        
        Args:
            width: Target width
            height: Target height
            
        Returns:
            Tiled noise texture array
        """
        noise_h, noise_w = self.blue_noise_texture.shape
        
        # Calculate how many tiles we need in each dimension
        tiles_x = (width + noise_w - 1) // noise_w
        tiles_y = (height + noise_h - 1) // noise_h
        
        # Tile the texture
        tiled = np.tile(self.blue_noise_texture, (tiles_y, tiles_x))
        
        # Crop to exact dimensions
        return tiled[:height, :width]
    
    def _calculate_adaptive_noise(self, rgb_image: np.ndarray) -> np.ndarray:
        """Calculate adaptive noise strength map with performance optimizations.
        
        Args:
            rgb_image: RGB image array of shape (H, W, 3)
            
        Returns:
            Noise strength map of shape (H, W)
        """
        height, width = rgb_image.shape[:2]
        
        if self.adaptive_strategy == 'uniform':
            return np.full((height, width), self.noise_strength)
        
        print("Calculating adaptive noise strength map...")
        
        with tqdm(total=100, desc="Adaptive noise", unit="%") as pbar:
            if self.adaptive_strategy == 'gradient':
                # Use gradient magnitude for adaptive noise
                gradient_map = self._calculate_gradient_map_optimized(rgb_image)
                pbar.update(100)
                noise_map = self.noise_strength * (1.0 - gradient_map * 0.7)
                return np.clip(noise_map, 0.1 * self.noise_strength, self.noise_strength)
            
            elif self.adaptive_strategy == 'edge':
                # Use edge detection for adaptive noise
                edge_map = self._calculate_edge_map_optimized(rgb_image)
                pbar.update(100)
                noise_map = self.noise_strength * (1.0 - edge_map * 0.8)
                return np.clip(noise_map, 0.1 * self.noise_strength, self.noise_strength)
            
            elif self.adaptive_strategy == 'contrast':
                # Use local contrast for adaptive noise
                contrast_map = self._calculate_contrast_map_optimized(rgb_image)
                pbar.update(100)
                noise_map = self.noise_strength * (1.0 - contrast_map * 0.6)
                return np.clip(noise_map, 0.2 * self.noise_strength, self.noise_strength)
            
            elif self.adaptive_strategy == 'gradient_edge':
                # Combine gradient and edge detection
                gradient_map = self._calculate_gradient_map_optimized(rgb_image)
                pbar.update(50)
                edge_map = self._calculate_edge_map_optimized(rgb_image)
                pbar.update(50)
                combined_map = (gradient_map + edge_map) / 2.0
                noise_map = self.noise_strength * (1.0 - combined_map * 0.75)
                return np.clip(noise_map, 0.1 * self.noise_strength, self.noise_strength)
            
            elif self.adaptive_strategy == 'gradient_contrast':
                # Combine gradient and contrast
                gradient_map = self._calculate_gradient_map_optimized(rgb_image)
                pbar.update(50)
                contrast_map = self._calculate_contrast_map_optimized(rgb_image)
                pbar.update(50)
                combined_map = (gradient_map + contrast_map) / 2.0
                noise_map = self.noise_strength * (1.0 - combined_map * 0.65)
                return np.clip(noise_map, 0.15 * self.noise_strength, self.noise_strength)
            
            elif self.adaptive_strategy == 'edge_contrast':
                # Combine edge and contrast
                edge_map = self._calculate_edge_map_optimized(rgb_image)
                pbar.update(50)
                contrast_map = self._calculate_contrast_map_optimized(rgb_image)
                pbar.update(50)
                combined_map = (edge_map + contrast_map) / 2.0
                noise_map = self.noise_strength * (1.0 - combined_map * 0.7)
                return np.clip(noise_map, 0.15 * self.noise_strength, self.noise_strength)
            
            elif self.adaptive_strategy == 'all':
                # Combine all three strategies
                gradient_map = self._calculate_gradient_map_optimized(rgb_image)
                pbar.update(33)
                edge_map = self._calculate_edge_map_optimized(rgb_image)
                pbar.update(33)
                contrast_map = self._calculate_contrast_map_optimized(rgb_image)
                pbar.update(34)
                combined_map = (gradient_map + edge_map + contrast_map) / 3.0
                noise_map = self.noise_strength * (1.0 - combined_map * 0.6)
                return np.clip(noise_map, 0.2 * self.noise_strength, self.noise_strength)
            
            else:
                pbar.update(100)
                return np.full((height, width), self.noise_strength)
    
    def _calculate_gradient_map_optimized(self, rgb_image: np.ndarray) -> np.ndarray:
        """Calculate gradient magnitude map using optimized scipy operations.
        
        Args:
            rgb_image: RGB image array of shape (H, W, 3)
            
        Returns:
            Normalized gradient magnitude map (0-1)
        """
        gray = np.mean(rgb_image, axis=2)
        
        # Use scipy's optimized gradient calculation
        grad_x = scipy.ndimage.sobel(gray, axis=1)
        grad_y = scipy.ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        return gradient_magnitude
    
    def _calculate_edge_map_optimized(self, rgb_image: np.ndarray) -> np.ndarray:
        """Calculate edge detection map using optimized skimage operations.
        
        Args:
            rgb_image: RGB image array of shape (H, W, 3)
            
        Returns:
            Normalized edge map (0-1)
        """
        gray = np.mean(rgb_image, axis=2)
        
        # Use skimage's optimized Canny edge detection
        edges = feature.canny(gray / 255.0, low_threshold=0.1, high_threshold=0.2)
        
        # Convert boolean to float and optionally dilate for smoother transitions
        edge_map = edges.astype(np.float32)
        
        # Apply Gaussian blur for smoother edge transitions
        edge_map = scipy.ndimage.gaussian_filter(edge_map, sigma=1.0)
        
        # Normalize
        if edge_map.max() > 0:
            edge_map = edge_map / edge_map.max()
        
        return edge_map
    
    def _calculate_contrast_map_optimized(self, rgb_image: np.ndarray) -> np.ndarray:
        """Calculate local contrast map using optimized scipy operations.
        
        Args:
            rgb_image: RGB image array of shape (H, W, 3)
            
        Returns:
            Normalized contrast map (0-1)
        """
        gray = np.mean(rgb_image, axis=2)
        
        # Use scipy's optimized uniform filter for local statistics
        kernel_size = 5
        
        # Calculate local mean and local variance efficiently
        local_mean = scipy.ndimage.uniform_filter(gray, size=kernel_size)
        local_mean_sq = scipy.ndimage.uniform_filter(gray**2, size=kernel_size)
        local_variance = local_mean_sq - local_mean**2
        
        # Local standard deviation as contrast measure
        contrast = np.sqrt(np.maximum(local_variance, 0))
        
        # Normalize
        if contrast.max() > 0:
            contrast = contrast / contrast.max()
        
        return contrast
    
    # Keep original methods as fallbacks for compatibility
    def _calculate_gradient_map(self, rgb_image: np.ndarray) -> np.ndarray:
        """Fallback gradient calculation (for compatibility)."""
        return self._calculate_gradient_map_optimized(rgb_image)
    
    def _calculate_edge_map(self, rgb_image: np.ndarray) -> np.ndarray:
        """Fallback edge calculation (for compatibility)."""
        return self._calculate_edge_map_optimized(rgb_image)
    
    def _calculate_contrast_map(self, rgb_image: np.ndarray) -> np.ndarray:
        """Fallback contrast calculation (for compatibility)."""
        return self._calculate_contrast_map_optimized(rgb_image)
    
    def _save_noise_strength_map(self, noise_map: np.ndarray, filepath: str) -> None:
        """Save noise strength map as a grayscale image.
        
        Args:
            noise_map: Noise strength map array (0-1 values)
            filepath: Path to save the image
        """
        try:
            # Normalize to 0-255 range
            normalized_map = np.clip(noise_map * 255 / self.noise_strength, 0, 255).astype(np.uint8)
            
            # Create PIL Image and save
            noise_image = Image.fromarray(normalized_map, mode='L')
            noise_image.save(filepath)
            
            print(f"Noise strength map saved to: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save noise strength map to {filepath}: {e}")
    
    def _find_closest_colors(self, colors: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """Find closest palette colors for a batch of colors.
        
        Args:
            colors: Array of colors to match, shape (N, 3)
            show_progress: Whether to show progress for expensive color distance methods
            
        Returns:
            Array of palette indices, shape (N,)
        """
        # Calculate distances to all palette colors
        distances = self.color_calculator.calculate_distances_batch(colors, self.palette, show_progress)
        
        # Find indices of closest colors
        closest_indices = np.argmin(distances, axis=1)
        
        return closest_indices
    
    def _apply_alpha_threshold(self, alpha_values: np.ndarray) -> np.ndarray:
        """Apply threshold method to alpha values.
        
        Args:
            alpha_values: Alpha values (0-255)
            
        Returns:
            Thresholded alpha values (0 or 255)
        """
        threshold = self.alpha_threshold * 255
        return np.where(alpha_values >= threshold, 255, 0).astype(np.uint8)
    
    def _apply_alpha_dithering(self, 
                              alpha_values: np.ndarray,
                              noise_values: np.ndarray,
                              strength_values: np.ndarray) -> np.ndarray:
        """Apply dithering method to alpha values.
        
        Args:
            alpha_values: Alpha values (0-255)
            noise_values: Blue noise values (0-1)
            strength_values: Noise strength values (0-1)
            
        Returns:
            Dithered alpha values (0 or 255)
        """
        # Normalize alpha to 0-1 range
        alpha_normalized = alpha_values / 255.0
        
        # Apply noise
        noise_offset = (noise_values - 0.5) * strength_values * 0.5
        noisy_alpha = alpha_normalized + noise_offset
        
        # Apply threshold with noise
        threshold = self.alpha_threshold
        result = np.where(noisy_alpha >= threshold, 255, 0).astype(np.uint8)
        
        return result
    
    def get_config(self) -> dict:
        """Get current configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'color_distance_method': self.color_calculator.method,
            'noise_strength': float(self.noise_strength),
            'adaptive_noise': self.adaptive_noise,
            'adaptive_strategy': self.adaptive_strategy,
            'alpha_method': self.alpha_method,
            'alpha_threshold': float(self.alpha_threshold),
            'output_noise_map': self.output_noise_map
        }
    
    def set_config(self, config: dict) -> None:
        """Set configuration from dictionary.
        
        Args:
            config: Configuration dictionary
        """
        if 'color_distance_method' in config:
            self.color_calculator = ColorDistanceCalculator(config['color_distance_method'])
        
        if 'noise_strength' in config:
            self.noise_strength = np.clip(config['noise_strength'], 0.0, 1.0)
        
        if 'adaptive_noise' in config:
            self.adaptive_noise = config['adaptive_noise']
        
        if 'adaptive_strategy' in config:
            if config['adaptive_strategy'] in self.ADAPTIVE_STRATEGIES:
                self.adaptive_strategy = config['adaptive_strategy']
        
        if 'alpha_method' in config:
            if config['alpha_method'] in self.ALPHA_METHODS:
                self.alpha_method = config['alpha_method']
        
        if 'alpha_threshold' in config:
            self.alpha_threshold = np.clip(config['alpha_threshold'], 0.0, 1.0)
        
        if 'output_noise_map' in config:
            self.output_noise_map = config['output_noise_map']