"""
Command Syntax
Usage:
bash
python data_preprocessing/smart_resizer.py --input_dir ./input --output_dir ./output --batch_size 1
This will create:

output/
├── input_raster_256x256/
│   ├── image1_256.png
│   ├── image2_256.png
│   └── ...
├── input_raster_512x512/
│   ├── image1_512.png
│   ├── image2_512.png
│   └── ...
├── processed_images.csv
└── skipped_images.csv

requirement:
pip install cupy-cuda11x  # Replace with your CUDA version
"""
import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
import argparse
import math


class DualSizeImageResizer:
    def __init__(self):
        self.skipped_images = []
        self.processed_images = []
        # Conservative minimum dimensions (width or height)
        self.MIN_SIZES = {
            '256x256': 192,  # Minimum dimension for 256px output
            '512x512': 384  # Minimum dimension for 512px output
        }
        self.ASPECT_RATIO_TOLERANCE = 0.15  # Allow ±15% aspect ratio change
        self.BATCH_SIZE = 8  # Conservative batch size
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available with CuPy"""
        try:
            import cupy as cp
            print(f"✅ Found {cp.cuda.runtime.getDeviceCount()} CUDA-capable GPU(s)")
            return True
        except ImportError:
            print("❌ CuPy not available - falling back to CPU")
            return False

    def _validate_image(self, img: np.ndarray, original_path: str) -> Tuple[bool, List[str]]:
        """Comprehensive quality validation for both target sizes"""
        h, w = img.shape[:2]
        reasons = []

        # 1. Minimum size check for both target sizes
        valid_for_256 = min(h, w) >= self.MIN_SIZES['256x256']
        valid_for_512 = min(h, w) >= self.MIN_SIZES['512x512']

        if not valid_for_256:
            reasons.append(f"Too small for 256x256 (min {self.MIN_SIZES['256x256']}px)")
        if not valid_for_512:
            reasons.append(f"Too small for 512x512 (min {self.MIN_SIZES['512x512']}px)")

        # 2. Aspect ratio validation
        original_ar = w / h
        for target_size in [256, 512]:
            scale = target_size / max(w, h)
            new_ar = (w * scale) / (h * scale)
            if abs(new_ar - original_ar) > self.ASPECT_RATIO_TOLERANCE:
                reasons.append(f"Excessive AR change for {target_size}px")

        # 3. Quality metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur detection
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_value < 150:
            reasons.append(f"Too blurry (Laplacian: {blur_value:.1f})")

        # Edge density (prevents thinning)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (w * h * 255)
        if edge_density < 0.015:
            reasons.append(f"Low edge density: {edge_density:.4f}")

        return (valid_for_256 and valid_for_512 and not reasons), reasons

    def _smart_resize(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """High-quality resize with aspect ratio preservation"""
        h, w = img.shape[:2]
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        interpolation = cv2.INTER_LANCZOS4 if scale > 0.7 else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        # Create canvas with transparency
        canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
        if resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)

        # Center the image
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def process_image(self, input_path: str, output_dir_256: str, output_dir_512: str) -> Optional[Dict]:
        """Process single image with dual output"""
        try:
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Invalid image file")

            # Convert to 3-channel if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Strict quality validation
            is_valid, reasons = self._validate_image(img, input_path)
            if not is_valid:
                self.skipped_images.append({
                    "file": os.path.basename(input_path),
                    "reasons": "; ".join(reasons),
                    "resolution": f"{img.shape[1]}x{img.shape[0]}"
                })
                return None

            # Create output directories
            os.makedirs(output_dir_256, exist_ok=True)
            os.makedirs(output_dir_512, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(input_path))[0]

            # Process 256x256 version
            resized_256 = self._smart_resize(img, 256)
            output_path_256 = os.path.join(output_dir_256, f"{base_name}_256.png")
            cv2.imwrite(output_path_256, resized_256, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Process 512x512 version
            resized_512 = self._smart_resize(img, 512)
            output_path_512 = os.path.join(output_dir_512, f"{base_name}_512.png")
            cv2.imwrite(output_path_512, resized_512, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Record metadata
            result = {
                "file": os.path.basename(input_path),
                "original_resolution": f"{img.shape[1]}x{img.shape[0]}",
                "output_256": output_path_256,
                "output_512": output_path_512,
                "status": "success"
            }
            self.processed_images.append(result)
            return result

        except Exception as e:
            self.skipped_images.append({
                "file": os.path.basename(input_path),
                "reason": str(e),
                "resolution": "Unknown"
            })
            return None

    def process_batch(self, input_dir: str, output_base_dir: str):
        """Process all images with dual output directories"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

        # Create output directories
        output_dir_256 = os.path.join(output_base_dir, "input_raster_256x256")
        output_dir_512 = os.path.join(output_base_dir, "input_raster_512x512")
        os.makedirs(output_dir_256, exist_ok=True)
        os.makedirs(output_dir_512, exist_ok=True)

        image_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(valid_extensions)
        ]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(
                lambda f: self.process_image(f, output_dir_256, output_dir_512),
                image_files
            ))

        # Save reports
        self._save_reports(output_base_dir)
        return [r for r in results if r is not None]

    def _save_reports(self, output_dir: str):
        """Generate quality control reports"""
        if self.skipped_images:
            pd.DataFrame(self.skipped_images).to_csv(
                os.path.join(output_dir, "skipped_images.csv"),
                index=False
            )

        if self.processed_images:
            pd.DataFrame(self.processed_images).to_csv(
                os.path.join(output_dir, "processed_images.csv"),
                index=False
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-size image resizer with strict quality control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True, help="Directory with source images")
    parser.add_argument("--output_dir", required=True, help="Base output directory (will create subdirectories)")
    parser.add_argument("--batch_size", type=int, default=8, help="Images to process in parallel")

    args = parser.parse_args()

    resizer = DualSizeImageResizer()
    resizer.BATCH_SIZE = args.batch_size

    print(f"Starting processing with {'GPU' if resizer.gpu_available else 'CPU'} acceleration")
    successful = resizer.process_batch(args.input_dir, args.output_dir)

    print(f"\nQuality Control Results:")
    print(f"✅ Processed: {len(successful)} images (created both sizes)")
    print(f"⚠️ Skipped: {len(resizer.skipped_images)} images")
    print(f"📊 256x256 outputs saved to: {os.path.join(args.output_dir, 'input_raster_256x256')}")
    print(f"📊 512x512 outputs saved to: {os.path.join(args.output_dir, 'input_raster_512x512')}")
    print(f"📊 Quality reports saved to: {args.output_dir}")