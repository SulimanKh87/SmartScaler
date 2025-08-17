import os
import csv
import sys
import argparse
import subprocess
from pathlib import Path
from compare_images import load_image, calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output" / "dataset" / "256" / "test" / "original"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "output" / "dataset" / "512" / "test" / "original"
OUTPUT_DIR = PROJECT_ROOT / "output" / "test_output"
MODEL_PATH = PROJECT_ROOT / "saved_models" / "upscaler.h5"
CSV_PATH = OUTPUT_DIR / "batch_compare_metrics.csv"

def infer_one(input_path: Path, out_path: Path, model_path: Path):
    # Always invoke the same interpreter that launched THIS script (tf_venv if UI chose it)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data_processing" / "inference.py"),
        "--input", str(input_path),
        "--output", str(out_path),
        "--model", str(model_path)
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Batch upscale + compare")
    parser.add_argument("--mode", choices=["dataset", "uploads"], default="dataset")
    parser.add_argument("--inputs256", type=str, help="Folder with *_256.png (uploads mode)")
    parser.add_argument("--targets512", type=str, help="Folder with *_512.png (optional)")
    args = parser.parse_args()

    if args.mode == "dataset":
        input_dir = DEFAULT_INPUT_DIR
        target_dir = DEFAULT_TARGET_DIR
    else:
        if not args.inputs256:
            print("❌ uploads mode requires --inputs256")
            sys.exit(2)
        input_dir = Path(args.inputs256)
        target_dir = Path(args.targets512) if args.targets512 else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare CSV
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "PSNR", "SSIM", "MSE"])

    # List inputs
    if not input_dir.exists():
        print(f"❌ Input dir not found: {input_dir}")
        sys.exit(2)

    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    total = len(files)
    print(f"Found {total} input images in {input_dir}")

    for idx, p in enumerate(files, 1):
        print(f"\n[{idx}/{total}] 🖼️ Processing: {p.name}")
        out_name = p.name.replace("_256", "_upscaled")
        out_path = OUTPUT_DIR / out_name

        try:
            infer_one(p, out_path, MODEL_PATH)
            print(f"✅ Inference complete: {p.name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference failed for {p.name}: {e}")
            continue

        # Compare if we have a matching ground-truth
        psnr = ssim = mse = None
        if args.mode == "dataset":
            tgt_name = p.name.replace("_256", "_512")
            target_path = DEFAULT_TARGET_DIR / tgt_name
        else:
            target_path = None
            if target_dir:
                candidate = target_dir / p.name.replace("_256", "_512")
                if candidate.exists():
                    target_path = candidate

        try:
            if target_path and target_path.exists():
                generated = load_image(str(out_path))
                ground_truth = load_image(str(target_path), target_size=generated.shape[:2])
                psnr, ssim, mse = calculate_metrics(generated, ground_truth)
                print("🔍 Comparison Results:")
                print(f"  PSNR : {psnr:.2f} dB")
                print(f"  SSIM : {ssim:.4f}")
                print(f"  MSE  : {mse:.6f}")
            else:
                print("ℹ️ No matching 512 target found — writing N/A for metrics.")
        except Exception as e:
            print(f"❌ Comparison failed for {p.name}: {e}")

        with CSV_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                p.name,
                f"{psnr:.2f}" if psnr is not None else "N/A",
                f"{ssim:.4f}" if ssim is not None else "N/A",
                f"{mse:.6f}" if mse is not None else "N/A",
            ])

    print(f"\n📊 Metrics saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
