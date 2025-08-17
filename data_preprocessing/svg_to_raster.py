"""
WARNING: The module assumes you've already run `svg_smart_resizer.py` and have properly sized 512x512 SVGs as input. The output PNGs will match the SVG dimensions exactly.
"""
"""
SVG to Raster Converter
Converts SVG files to high-quality PNG raster images with transparency support.

Notice: this process can be slow and take up to few hours depending on your hardware

Why It's It's relatively slow? it's Not Using GPU:
CairoSVG is CPU-only: The cairosvg library you're using doesn't have GPU acceleration capabilities
ThreadPoolExecutor uses CPU: Your parallel processing uses CPU threads, not GPU cores

"""

import os
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import cairosvg
import pandas as pd
from typing import List, Dict
from PIL import Image


class SvgToRasterConverter:
    def __init__(self, workers: int = 16):
        """
        Initialize the converter with specified number of workers.

        Args:
            workers: Number of parallel conversion processes (default: 16)
        """
        self.successful_conversions = []
        self.failed_conversions = []
        self.workers = workers

    def convert_svg_to_png(self, svg_path: str, output_dir: str) -> Dict:
        """
        Convert a single SVG file to PNG.

        Args:
            svg_path: Path to input SVG file
            output_dir: Directory to save output PNG

        Returns:
            Dictionary containing conversion results
        """
        try:
            filename = os.path.basename(svg_path)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)

            # Convert with CairoSVG (maintains transparency)
            cairosvg.svg2png(
                url=svg_path,
                write_to=output_path
            )
            # print image dimensions
            img = Image.open(output_path)
            print(f"Rendered size: {img.size}")

            # Verify output was created
            if not os.path.exists(output_path):
                raise ValueError("Output file not created")

            return {
                "input_file": filename,
                "output_file": output_filename,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "resolution": "512x512"
            }

        except Exception as e:
            return {
                "input_file": os.path.basename(svg_path),
                "output_file": "",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "resolution": ""
            }

    def process_batch(self, svg_dir: str, output_dir: str) -> int:
        """
        Process all SVG files in a directory.

        Args:
            svg_dir: Directory containing SVG files
            output_dir: Output directory for PNG files

        Returns:
            Number of successful conversions
        """
        os.makedirs(output_dir, exist_ok=True)

        svg_files = [
            os.path.join(svg_dir, f)
            for f in os.listdir(svg_dir)
            if f.lower().endswith('.svg')
        ]

        print(f"Found {len(svg_files)} SVG files to process...")

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.convert_svg_to_png, svg, output_dir): svg
                for svg in svg_files
            }

            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "success":
                    self.successful_conversions.append(result)
                else:
                    self.failed_conversions.append(result)

        self._generate_reports(os.path.dirname(output_dir))
        return len(self.successful_conversions)

    def _generate_reports(self, report_dir: str):
        """Generate CSV reports for conversions"""
        os.makedirs(report_dir, exist_ok=True)

        # Successful conversions report
        if self.successful_conversions:
            success_df = pd.DataFrame(self.successful_conversions)
            success_path = os.path.join(report_dir, "rasterization_report.csv")
            success_df.to_csv(success_path, index=False)
            print(f"✓ Success report saved to {success_path}")

        # Error report
        if self.failed_conversions:
            error_df = pd.DataFrame(self.failed_conversions)
            error_path = os.path.join(report_dir, "rasterization_report_errors.csv")
            error_df.to_csv(error_path, index=False)
            print(f"✗ Error report saved to {error_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SVG files to high-quality PNG raster images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--svg_input",
        required=True,
        help="Directory containing SVG files"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for PNG files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers to use"
    )

    args = parser.parse_args()

    converter = SvgToRasterConverter(workers=args.workers)
    print(f"Starting SVG to PNG conversion with {converter.workers} workers...")

    successful = converter.process_batch(args.svg_input, args.output_dir)

    print(f"\nConversion Complete:")
    print(f"✅ Successfully converted: {successful} files")
    print(f"⚠️ Failed conversions: {len(converter.failed_conversions)} files")
    print(f"📊 Reports saved to: {os.path.dirname(args.output_dir)}")


if __name__ == "__main__":
    main()
