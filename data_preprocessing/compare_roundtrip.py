"""
SVG Roundtrip Image Quality Evaluation Tool
This script compares original raster images with their SVG-roundtrip rasterized versions to evaluate how much quality is retained during the conversion → vectorization → reconversion process.

🎯 Purpose
The goal is to verify the visual fidelity of your vectorization pipeline by computing:

🔬 PSNR – Peak Signal-to-Noise Ratio

🧠 SSIM – Structural Similarity Index

🧮 MSE – Mean Squared Error
Along with visual difference maps that highlight pixel-level changes.

🖥️ How to Use
Run from the terminal:

▶️ 256×256 Image Comparison
bash
Copy
Edit
python compare_roundtrip.py \
  --originals output/input_raster_256x256 \
  --svg_roundtrip output/svg_rasterized_256X256 \
  --output output/raster_comparison_results_256x256
▶️ 512×512 Image Comparison
bash
Copy
Edit
python compare_roundtrip.py \
  --originals output/input_raster_512x512 \
  --svg_roundtrip output/svg_rasterized_512x512 \
  --output output/raster_comparison_results_512x512
🛠 Features
✅ Robust Alignment

Auto-resizes mismatched images

Normalizes color channels (grayscale/RGB/BGRA support)

📏 Accurate Metrics

PSNR – Measures pixel fidelity

SSIM – Captures structural similarity

MSE – Measures absolute error

🖼 Visual Inspection

Saves heatmap difference images

Shows side-by-side comparisons

📊 Reports

CSV with metrics for each image

summary.txt with min/max/avg per metric

Logs any failed comparisons with reasons

📖 Interpretation Guide
PSNR
Range	Meaning
≥30 dB	Excellent quality
25–30 dB	Good quality
20–25 dB	Fair quality
<20 dB	Poor quality

SSIM
Range	Meaning
0.95–1.00	Virtually identical
0.90–0.95	Very similar
<0.90	Noticeable change

MSE
Lower is better

0 means identical

📂 Output Files
After running, the output folder will contain:

roundtrip_comparison.csv → Per-image PSNR, SSIM, MSE

summary.txt → Aggregated quality statistics

diffs/*.jpg → Visual side-by-side comparisons

"""

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
import traceback

class RoundtripComparator:
    def __init__(self):
        self.results = []
        self.failures = []
        self.metric_stats = {
            'psnr': {'values': [], 'min': float('inf'), 'max': -float('inf'), 'avg': 0},
            'ssim': {'values': [], 'min': float('inf'), 'max': -float('inf'), 'avg': 0},
            'mse': {'values': [], 'min': float('inf'), 'max': -float('inf'), 'avg': 0}
        }

    def _mse(self, img1, img2):
        """Calculate Mean Squared Error"""
        return np.mean((img1 - img2) ** 2)

    def _align_images(self, orig, processed):
        """Ensure images have same dimensions and number of channels"""
        try:

            # Resize to match spatial dimensions
            if orig.shape[:2] != processed.shape[:2]:
                processed = cv2.resize(processed, (orig.shape[1], orig.shape[0]))

            # Match channel count
            if len(orig.shape) == 2:
                orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
            elif orig.shape[2] == 4:
                orig = cv2.cvtColor(orig, cv2.COLOR_BGRA2BGR)

            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            elif processed.shape[2] == 4:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)

            return orig, processed
        except Exception as e:
            print(f"Alignment error: {str(e)}")
            raise

    def compare_pair(self, orig_path, svgrt_path):
        """Compare original image with SVG-roundtrip version"""
        try:
            orig = cv2.imread(orig_path, cv2.IMREAD_UNCHANGED)
            svgrt = cv2.imread(svgrt_path, cv2.IMREAD_UNCHANGED)

            if orig is None:
                raise ValueError(f"Failed to load original image: {orig_path}")
            if svgrt is None:
                raise ValueError(f"Failed to load SVG-roundtrip image: {svgrt_path}")

            orig, svgrt = self._align_images(orig, svgrt)

            if len(orig.shape) == 2:
                ssim_val = ssim(orig, svgrt, data_range=255)
            else:
                ssim_val = ssim(orig, svgrt, data_range=255, multichannel=True, channel_axis=2)

            psnr_val = psnr(orig, svgrt, data_range=255)
            mse_val = self._mse(orig, svgrt)

            self._update_stats('psnr', psnr_val)
            self._update_stats('ssim', ssim_val)
            self._update_stats('mse', mse_val)

            diff_vis = self._visual_diff(orig, svgrt)

            return {
                'original': os.path.basename(orig_path),
                'processed': os.path.basename(svgrt_path),
                'psnr': psnr_val,
                'ssim': ssim_val,
                'mse': mse_val,
                'diff_vis': diff_vis,
                'status': 'success'
            }

        except Exception as e:
            error = {
                'original': os.path.basename(orig_path),
                'processed': os.path.basename(svgrt_path),
                'error': str(e),
                'status': 'failed'
            }
            self.failures.append(error)
            return error

    def _update_stats(self, metric, value):
        """Update statistics for a given metric"""
        if value is not None:
            self.metric_stats[metric]['values'].append(value)
            self.metric_stats[metric]['min'] = min(self.metric_stats[metric]['min'], value)
            self.metric_stats[metric]['max'] = max(self.metric_stats[metric]['max'], value)
            self.metric_stats[metric]['avg'] = sum(self.metric_stats[metric]['values']) / len(self.metric_stats[metric]['values'])

    def _visual_diff(self, orig, processed):
        """Generate visual difference map"""
        if len(orig.shape) == 3:
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = orig
            proc_gray = processed

        diff = cv2.absdiff(orig_gray, proc_gray)
        diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)

        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        comparison = np.hstack([orig, processed, diff_vis])
        return comparison

    def process_batch(self, orig_dir, svgrt_dir, output_dir):
        """Process all matching images in directories"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'diffs'), exist_ok=True)

        orig_files = {os.path.splitext(f)[0]: f for f in os.listdir(orig_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        svgrt_files = {os.path.splitext(f)[0]: f for f in os.listdir(svgrt_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

        common = set(orig_files.keys()) & set(svgrt_files.keys())
        if not common:
            raise ValueError("No matching files found between directories")

        print(f"Found {len(common)} image pairs to compare")

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for base in common:
                orig_path = os.path.join(orig_dir, orig_files[base])
                svgrt_path = os.path.join(svgrt_dir, svgrt_files[base])
                futures.append(executor.submit(self.compare_pair, orig_path, svgrt_path))

            for future in futures:
                result = future.result()
                if result['status'] == 'success':
                    diff_path = os.path.join(output_dir, 'diffs',
                                           f"{os.path.splitext(result['original'])[0]}_diff.jpg")
                    cv2.imwrite(diff_path, result['diff_vis'])
                    result['diff_path'] = diff_path
                    self.results.append(result)

    def generate_report(self, output_dir):
        """Generate comprehensive comparison report"""
        df = pd.DataFrame(self.results)
        report_path = os.path.join(output_dir, 'roundtrip_comparison.csv')
        df.to_csv(report_path, index=False)

        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("SVG Roundtrip Quality Report\n")
            f.write("===========================\n\n")
            f.write(f"Total comparisons: {len(self.results)}\n")
            f.write(f"Failed comparisons: {len(self.failures)}\n\n")

            f.write("Quality Metrics Summary:\n")
            for metric, stats in self.metric_stats.items():
                if stats['values']:
                    f.write(f"{metric.upper()}:\n")
                    f.write(f"  Min: {stats['min']:.2f}\n")
                    f.write(f"  Max: {stats['max']:.2f}\n")
                    f.write(f"  Avg: {stats['avg']:.2f}\n\n")

            if self.failures:
                f.write("\nFirst 5 Failure Cases:\n")
                for fail in self.failures[:5]:
                    f.write(f"- {fail['original']}: {fail['error']}\n")

        print(f"Reports saved to {output_dir}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare original raster images with SVG-roundtrip versions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--originals", required=True, help="Directory containing original raster images")
    parser.add_argument("--svg_roundtrip", required=True, help="Directory containing SVG-roundtrip raster images")
    parser.add_argument("--output", required=True, help="Output directory for reports")

    args = parser.parse_args()

    if not os.path.exists(args.originals):
        print(f"Error: Original images directory does not exist: {args.originals}")
        return

    if not os.path.exists(args.svg_roundtrip):
        print(f"Error: SVG-roundtrip images directory does not exist: {args.svg_roundtrip}")
        return

    print("\nDirectory verification:")
    print(f"Originals: {args.originals} - {len(os.listdir(args.originals))} files")
    print(f"SVG Roundtrip: {args.svg_roundtrip} - {len(os.listdir(args.svg_roundtrip))} files")

    comparator = RoundtripComparator()

    print("\nStarting SVG roundtrip comparison...")
    try:
        comparator.process_batch(args.originals, args.svg_roundtrip, args.output)
        comparator.generate_report(args.output)

        print("\nComparison Complete:")
        print(f"✅ Successful comparisons: {len(comparator.results)}")
        print(f"⚠️ Failed comparisons: {len(comparator.failures)}")
        print(f"📊 Reports saved to: {args.output}")

        if comparator.failures:
            print("\nFirst 5 failure reasons:")
            for i, fail in enumerate(comparator.failures[:5]):
                print(f"{i+1}. {fail['original']}: {fail['error']}")
    except Exception as e:
        print(f"\nFatal error during comparison: {str(e)}")

if __name__ == "__main__":
    main()