"""
bash
python data_processing/compare_images.py --output output/test_output.png --target output/dataset/512/test/original/149_bird-9950_1280_512.png

"""

import tensorflow as tf
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import argparse
from skimage.metrics import structural_similarity, mean_squared_error



# This function loads an image from disk and optionally resizes it
def load_image(path, target_size=None):
    # Read image file
    image = tf.io.read_file(path)
    # Decode PNG/JPG and ensure 3 RGB channels
    image = tf.image.decode_image(image, channels=3)
    # Normalize image to [0, 1] float32
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize if target size is given
    if target_size:
        image = tf.image.resize(image, target_size)
    return image.numpy()


# Compute PSNR, SSIM, and MSE between two images
def calculate_metrics(img1, img2):
    # Convert float32 [0,1] images to uint8 [0,255] for SSIM
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)

    psnr = peak_signal_noise_ratio(img2, img1, data_range=1.0)
    ssim = structural_similarity(img1_uint8, img2_uint8, win_size=7, channel_axis=-1, data_range=255)
    mse = mean_squared_error(img2, img1)
    return psnr, ssim, mse


if __name__ == "__main__":
    # Accept input/output paths from the command line
    parser = argparse.ArgumentParser(description="Compare super-resolved image to ground truth")
    parser.add_argument("--output", required=True,
                        help="Path to model-generated output image (e.g., output/test_output.png)")
    parser.add_argument("--target", required=True, help="Path to original high-res ground truth image")
    args = parser.parse_args()

    # Load the two images
    generated = load_image(args.output)
    ground_truth = load_image(args.target, target_size=generated.shape[:2])  # match the resolution

    # Calculate similarity metrics
    psnr, ssim, mse = calculate_metrics(generated, ground_truth)

    # Print results
    print("🔍 Comparison Results:")
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM : {ssim:.4f}")
    print(f"  MSE  : {mse:.6f}")