"""
bash

python data_processing/inference.py --input output/dataset/256/test/original/149_bird-9950_1280_256.png --output output/test_output.png


"""
import tensorflow as tf
import numpy as np
import argparse
import os
from PIL import Image

def load_image(path, size=(256, 256)):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def save_image(tensor, path):
    array = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)

def run_inference(image_path, model_path, output_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    image = load_image(image_path)
    image = tf.expand_dims(image, 0)  # Add batch dim

    output = model(image, training=False)[0].numpy()  # Remove batch dim
    save_image(output, output_path)
    print(f"Upscaled image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to 256x256 input image")
    parser.add_argument("--model", default="saved_models/upscaler.h5", help="Path to model")
    parser.add_argument("--output", default="upscaled_output.png", help="Path to save upscaled result")
    args = parser.parse_args()

    run_inference(args.input, args.model, args.output)
