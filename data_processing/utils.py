import tensorflow as tf
import numpy as np
import cv2


def preprocess_image(image_path, target_size=(256, 256)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def postprocess_output(image_tensor):
    image = image_tensor.numpy() * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def save_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def visualize_comparison(low_res, high_res, pred, output_path=None):
    low_res = postprocess_output(low_res)
    high_res = postprocess_output(high_res)
    pred = postprocess_output(pred)
    low_res_big = cv2.resize(low_res, (high_res.shape[1], high_res.shape[0]))
    comparison = np.hstack([low_res_big, high_res, pred])

    if output_path:
        save_image(comparison, output_path)
    return comparison