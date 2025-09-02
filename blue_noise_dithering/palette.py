"""Palette loading functionality for Paint.net TXT format."""

import numpy as np
from typing import List, Tuple
import re


class PaletteLoader:
    """Load and manage color palettes in Paint.net TXT format."""
    
    def __init__(self):
        """Initialize palette loader."""
        self.colors = []
        self.color_array = None
    
    def load_from_file(self, filepath: str) -> None:
        """Load palette from Paint.net TXT file.
        
        Paint.net TXT format:
        - Each line contains a color in hex format (#RRGGBB or #AARRGGBB)
        - Lines starting with ; are comments
        - Empty lines are ignored
        
        Args:
            filepath: Path to the palette file
        """
        self.colors = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith(';'):
                        continue
                    
                    # Parse hex color
                    color = self._parse_hex_color(line, line_num)
                    if color is not None:
                        self.colors.append(color)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Palette file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading palette: {e}")
        
        if not self.colors:
            raise ValueError("No valid colors found in palette file")
        
        # Convert to numpy array for efficient processing
        self.color_array = np.array(self.colors, dtype=np.uint8)
    
    def _parse_hex_color(self, line: str, line_num: int) -> Tuple[int, int, int]:
        """Parse a hex color string.
        
        Args:
            line: Line containing hex color
            line_num: Line number for error reporting
            
        Returns:
            RGB tuple (R, G, B) or None if invalid
        """
        # Remove any whitespace and comments
        line = line.split(';')[0].strip()
        
        # Match hex color patterns
        hex_patterns = [
            r'^#?([A-Fa-f0-9]{6})$',          # #RRGGBB or RRGGBB
            r'^#?([A-Fa-f0-9]{8})$',          # #AARRGGBB or AARRGGBB
            r'^#?([A-Fa-f0-9]{3})$'           # #RGB or RGB
        ]
        
        for pattern in hex_patterns:
            match = re.match(pattern, line)
            if match:
                hex_str = match.group(1)
                return self._hex_to_rgb(hex_str, line_num)
        
        # Try parsing as decimal RGB values (R,G,B) or (R G B)
        rgb_match = re.match(r'^(\d+)[,\s]+(\d+)[,\s]+(\d+)$', line)
        if rgb_match:
            try:
                r, g, b = map(int, rgb_match.groups())
                if all(0 <= c <= 255 for c in [r, g, b]):
                    return (r, g, b)
            except ValueError:
                pass
        
        print(f"Warning: Invalid color format on line {line_num}: {line}")
        return None
    
    def _hex_to_rgb(self, hex_str: str, line_num: int) -> Tuple[int, int, int]:
        """Convert hex string to RGB tuple.
        
        Args:
            hex_str: Hex string without # prefix
            line_num: Line number for error reporting
            
        Returns:
            RGB tuple (R, G, B)
        """
        try:
            if len(hex_str) == 3:
                # #RGB -> #RRGGBB
                r = int(hex_str[0] * 2, 16)
                g = int(hex_str[1] * 2, 16)
                b = int(hex_str[2] * 2, 16)
            elif len(hex_str) == 6:
                # #RRGGBB
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
            elif len(hex_str) == 8:
                # #AARRGGBB - ignore alpha channel
                r = int(hex_str[2:4], 16)
                g = int(hex_str[4:6], 16)
                b = int(hex_str[6:8], 16)
            else:
                raise ValueError(f"Invalid hex length: {len(hex_str)}")
            
            return (r, g, b)
        
        except ValueError as e:
            print(f"Warning: Invalid hex color on line {line_num}: {hex_str}")
            return None
    
    def load_from_colors(self, colors: List[Tuple[int, int, int]]) -> None:
        """Load palette from a list of RGB colors.
        
        Args:
            colors: List of (R, G, B) tuples
        """
        self.colors = list(colors)
        self.color_array = np.array(self.colors, dtype=np.uint8)
    
    def get_colors(self) -> np.ndarray:
        """Get the palette colors as numpy array.
        
        Returns:
            Array of shape (N, 3) with RGB colors
        """
        if self.color_array is None:
            raise ValueError("No palette loaded")
        return self.color_array.copy()
    
    def get_color_count(self) -> int:
        """Get the number of colors in the palette.
        
        Returns:
            Number of colors
        """
        return len(self.colors) if self.colors else 0
    
    def save_to_file(self, filepath: str) -> None:
        """Save current palette to Paint.net TXT file.
        
        Args:
            filepath: Output file path
        """
        if not self.colors:
            raise ValueError("No palette to save")
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write("; Paint.NET Palette File\n")
            file.write(f"; Colors: {len(self.colors)}\n")
            file.write(";\n")
            
            for r, g, b in self.colors:
                file.write(f"#{r:02X}{g:02X}{b:02X}\n")
    
    def add_color(self, r: int, g: int, b: int) -> None:
        """Add a color to the palette.
        
        Args:
            r, g, b: RGB values (0-255)
        """
        if not all(0 <= c <= 255 for c in [r, g, b]):
            raise ValueError("RGB values must be in range 0-255")
        
        self.colors.append((r, g, b))
        self.color_array = np.array(self.colors, dtype=np.uint8)
    
    def remove_duplicates(self) -> int:
        """Remove duplicate colors from palette.
        
        Returns:
            Number of duplicates removed
        """
        original_count = len(self.colors)
        
        # Convert to set to remove duplicates, then back to list
        unique_colors = list(set(self.colors))
        self.colors = unique_colors
        self.color_array = np.array(self.colors, dtype=np.uint8)
        
        return original_count - len(self.colors)
    
    def create_web_safe_palette(self) -> None:
        """Create a web-safe 216-color palette."""
        self.colors = []
        
        # Web-safe colors use values 0, 51, 102, 153, 204, 255
        web_safe_values = [0, 51, 102, 153, 204, 255]
        
        for r in web_safe_values:
            for g in web_safe_values:
                for b in web_safe_values:
                    self.colors.append((r, g, b))
        
        self.color_array = np.array(self.colors, dtype=np.uint8)
    
    def create_grayscale_palette(self, levels: int = 256) -> None:
        """Create a grayscale palette.
        
        Args:
            levels: Number of gray levels (default 256)
        """
        if levels < 2 or levels > 256:
            raise ValueError("Levels must be between 2 and 256")
        
        self.colors = []
        
        for i in range(levels):
            gray_value = int(i * 255 / (levels - 1))
            self.colors.append((gray_value, gray_value, gray_value))
        
        self.color_array = np.array(self.colors, dtype=np.uint8)
    
    def __len__(self) -> int:
        """Get the number of colors in the palette."""
        return len(self.colors)
    
    def __getitem__(self, index: int) -> Tuple[int, int, int]:
        """Get a color by index."""
        return self.colors[index]
    
    def __iter__(self):
        """Iterate over colors in the palette."""
        return iter(self.colors)