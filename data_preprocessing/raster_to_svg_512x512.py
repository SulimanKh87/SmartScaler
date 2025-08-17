"""
This module converts raster images (e.g., PNG, JPG) into per-pixel SVG vector graphics,
while preserving the original aspect ratio with transparent padding.

Key Features:
-------------
- Maintains original aspect ratio with transparent padding
- Outputs consistent 512x512 SVG files with transparency
- Supports both GPU (CuPy) and CPU processing
- Preserves transparency from source images
- Generates detailed metadata for each conversion
- Processes images in parallel for better performance

Input:
-------
- A directory of raster images (PNG/JPG/JPEG) stored in: `output/input_raster_512x512`

Output:
--------
- 512x512 SVG files with transparent padding saved to: `output/svgs_512x512`
- Metadata CSV summarizing each conversion
"""

import os
import sys
import cv2
import numpy as np
import svgwrite
import uuid
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, Dict, List


def check_gpu_support() -> tuple:
    """Check for GPU support with comprehensive error handling"""
    gpu_available = False
    gpu_libs = {}
    cupy = None

    try:
        import cupy as cp
        cupy = cp
        gpu_libs['cupy'] = True
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            gpu_libs['cuda_runtime'] = True
            gpu_available = device_count > 0
            if gpu_available:
                print(f"✅ Found {device_count} CUDA-capable GPU(s)")
                print(f"CuPy version: {cp.__version__}")
                print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
                device_id = cp.cuda.Device().id
                device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name']
                print(f"Current device: {device_name}")

            else:
                print("⚠️ CUDA runtime found but no GPUs detected")
        except Exception as e:
            print(f"⚠️ CUDA runtime error: {str(e)}")
            gpu_libs['cuda_runtime'] = False
    except ImportError:
        gpu_libs['cupy'] = False
        print("❌ CuPy not installed")
    except Exception as e:
        gpu_libs['cupy'] = False
        print(f"⚠️ CuPy import error: {str(e)}")

    return gpu_available, gpu_libs, cupy


# Check GPU support and get cupy module reference if available
GPU_AVAILABLE, GPU_LIBS, cp = check_gpu_support()


class ImageProcessor:
    def __init__(self):
        self.metadata = []
        self.skipped_images = []
        self.target_size = 512  # Output will be square 512x512
        self.MIN_FILE_SIZE_KB = 1  # Minimum file size in KB

    def validate_image(self, image_path: str) -> Optional[Union[np.ndarray, 'cp.ndarray']]:
        """Validate and load image with support for transparency"""
        try:
            # Read image with alpha channel if present
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"Warning: Failed to load image '{image_path}'. Skipping.")
                return None

            # Convert to BGRA if needed (3-channel images)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
            elif image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

            # Check dimensions
            if image.shape[0] == 0 or image.shape[1] == 0:
                print(f"Warning: Image '{image_path}' has invalid dimensions. Skipping.")
                return None

            # Convert to GPU if available
            if GPU_AVAILABLE and cp is not None:
                try:
                    return cp.asarray(image)
                except Exception as e:
                    print(f"GPU transfer failed, using CPU: {str(e)}")
                    return image
            return image

        except Exception as e:
            print(f"Error validating image {image_path}: {str(e)}")
            return None

    def _resize_with_padding(self, img: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Resize image while maintaining aspect ratio with transparent padding"""
        if GPU_AVAILABLE and cp is not None and isinstance(img, cp.ndarray):
            try:
                return self._gpu_resize_with_padding(img)
            except Exception as e:
                print(f"GPU resize failed, falling back to CPU: {str(e)}")
                return self._cpu_resize_with_padding(cp.asnumpy(img))
        return self._cpu_resize_with_padding(img)

    def _gpu_resize_with_padding(self, img: 'cp.ndarray') -> np.ndarray:
        """GPU-accelerated resize with transparent padding"""
        h, w = img.shape[:2]
        scale = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Create coordinate grids
        y = cp.linspace(0, h-1, new_h)
        x = cp.linspace(0, w-1, new_w)
        y_idx = cp.round(y).astype(cp.int32)
        x_idx = cp.round(x).astype(cp.int32)

        # Resize using advanced indexing
        resized = img[y_idx[:, None], x_idx[None, :]]

        # Create transparent canvas
        canvas = cp.zeros((self.target_size, self.target_size, 4), dtype=img.dtype)
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return cp.asnumpy(canvas)

    def _cpu_resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        """CPU implementation of resize with transparent padding"""
        h, w = img.shape[:2]
        scale = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Choose appropriate interpolation
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Create transparent canvas
        canvas = np.zeros((self.target_size, self.target_size, 4), dtype=np.uint8)
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def collect_metadata(self, image_path: str, image: Union[np.ndarray, 'cp.ndarray']) -> Dict:
        """Collect comprehensive metadata about the image"""
        original_h, original_w = image.shape[:2]
        file_size = os.path.getsize(image_path) / 1024  # KB
        has_alpha = image.shape[2] == 4
        img_format = os.path.splitext(image_path)[1].upper()[1:]

        return {
            "filename": os.path.basename(image_path),
            "original_resolution": f"{original_w}x{original_h}",
            "file_size_kb": round(file_size, 2),
            "format": img_format,
            "has_alpha": has_alpha,
            "processing_mode": "GPU" if GPU_AVAILABLE and isinstance(image, cp.ndarray) else "CPU",
            "target_size": self.target_size
        }

    def convert_to_svg(self, input_image_path: str, output_dir: str) -> Optional[str]:
        """Convert raster image to SVG with transparent padding"""
        try:
            # Validate and load image
            image = self.validate_image(input_image_path)
            if image is None:
                self.skipped_images.append(input_image_path)
                return None

            # Collect metadata
            self.metadata.append(self.collect_metadata(input_image_path, image))

            # Resize with aspect ratio preservation and transparent padding
            resized_image = self._resize_with_padding(image)

            # Create output directory if needed
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(input_image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.svg")

            # Generate SVG
            self._create_svg(resized_image, output_path)

            return output_path
        except Exception as e:
            print(f"Error converting {input_image_path}: {str(e)}")
            self.skipped_images.append(input_image_path)
            return None

    def _create_svg(self, image: np.ndarray, output_path: str):
        """Create optimized SVG with transparent support"""
        dwg = svgwrite.Drawing(output_path, size=(self.target_size, self.target_size), profile='tiny')

        # Process in chunks for memory efficiency
        chunk_size = 64
        for y_start in range(0, image.shape[0], chunk_size):
            y_end = min(y_start + chunk_size, image.shape[0])
            for y in range(y_start, y_end):
                for x in range(image.shape[1]):
                    pixel = image[y, x]
                    if len(pixel) == 4:  # RGBA
                        b, g, r, a = pixel  # OpenCV uses BGRA order
                        if a > 0:  # Only add non-transparent pixels
                            color = f"rgb({r},{g},{b})"
                            opacity = a / 255.0
                            dwg.add(dwg.rect(
                                insert=(x, y),
                                size=(1, 1),
                                fill=color,
                                stroke='none',
                                opacity=opacity
                            ))
                    else:  # Shouldn't happen as we convert to RGBA earlier
                        b, g, r = pixel
                        dwg.add(dwg.rect(
                            insert=(x, y),
                            size=(1, 1),
                            fill=f"rgb({r},{g},{b})",
                            stroke='none'
                        ))

        dwg.save()
        print(f"SVG saved: {output_path}")


def main():
    input_dir = "output/input_raster_512x512"
    output_dir = os.path.join("output", "svgs_512x512")

    # Verify input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Get image files
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_files:
        print(f"No valid images found in {input_dir}")
        sys.exit(0)

    # Initialize processor
    processor = ImageProcessor()

    # Process images
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(
            lambda f: processor.convert_to_svg(
                os.path.join(input_dir, f),
                output_dir
            ),
            image_files
        ))

    # Save metadata
    if processor.metadata:
        metadata_df = pd.DataFrame(processor.metadata)
        metadata_path = os.path.join(output_dir, "image_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"\nMetadata saved to: {metadata_path}")

    # Report skipped files
    if processor.skipped_images:
        print("\nSkipped files:")
        for img in processor.skipped_images:
            print(f"- {img}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()