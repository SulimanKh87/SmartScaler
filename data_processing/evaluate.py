"""
run this script using bash

 python data_processing/evaluate.py --data_dir output/dataset --model_path saved_models/upscaler.h5

"""

import tensorflow as tf
import numpy as np
import data_loader
import argparse
import os
import pandas as pd

def calculate_psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def calculate_ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def evaluate_model(model, dataset):
    psnr_values = []
    ssim_values = []
    filenames = []

    for (low_res, high_res), path_batch in dataset:
        pred = model.predict(low_res)

        for i in range(pred.shape[0]):
            psnr = calculate_psnr(high_res[i], pred[i]).numpy()
            ssim = calculate_ssim(high_res[i], pred[i]).numpy()

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            filename = path_batch[i]
            filenames.append(str(filename))
    return filenames, psnr_values, ssim_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="saved_models/upscaler.h5")
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # Load validation dataset
    val_ds = data_loader.get_data_pipeline(args.data_dir, batch_size=1, split="val")

    filenames, psnr_scores, ssim_scores = evaluate_model(model, val_ds)

    # Print average metrics
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    print(f"\nEvaluation Metrics (Validation Set):")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Save detailed results to CSV
    results_df = pd.DataFrame({
        "Filename": [os.path.basename(f) for f in filenames],
        "PSNR": psnr_scores,
        "SSIM": ssim_scores
    })
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Saved evaluation_results.csv ✅")
