import tensorflow as tf
from pathlib import Path
from config import config


def load_image(image_path, target_size=(256, 256)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])  # Ensure shape is known for resizing
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def create_dataset(base_dir, split="val"):
    base_path = Path(base_dir)

    low_res_paths = sorted((base_path / "256" / split / "original").glob("*.png"))
    svg_low_paths = sorted((base_path / "256" / split / "svg_rasterized").glob("*.png"))
    high_res_paths = sorted((base_path / "512" / split / "original").glob("*.png"))
    svg_high_paths = sorted((base_path / "512" / split / "svg_rasterized").glob("*.png"))

    print(f"Low-res: {len(low_res_paths)}")
    print(f"SVG low-res: {len(svg_low_paths)}")
    print(f"High-res: {len(high_res_paths)}")
    print(f"SVG high-res: {len(svg_high_paths)}")

    if not low_res_paths or not high_res_paths or not svg_high_paths:
        print("🛑 One or more input folders are empty!")
        return tf.data.Dataset.from_tensor_slices([])

    def get_prefix(path):
        return path.stem.split("_")[0]  # "123" from "123_image.png"

    # Check matching prefixes
    for a, b in zip(low_res_paths, svg_low_paths):
        if get_prefix(a) != get_prefix(b):
            print("❌ Mismatch:", a.name, b.name)

    for a, b in zip(high_res_paths, svg_high_paths):
        if get_prefix(a) != get_prefix(b):
            print("❌ Mismatch:", a.name, b.name)

    # Convert to strings
    low_res_paths = [str(p) for p in low_res_paths]
    high_res_paths = [str(p) for p in high_res_paths]
    svg_high_paths = [str(p) for p in svg_high_paths]

    # Map
    low_res_ds = tf.data.Dataset.from_tensor_slices(low_res_paths).map(
        lambda x: load_image(x, target_size=config.INPUT_SHAPE[:2]),
        num_parallel_calls=tf.data.AUTOTUNE)

    high_res_ds = tf.data.Dataset.from_tensor_slices(high_res_paths).map(
        lambda x: load_image(x, target_size=config.OUTPUT_SHAPE[:2]),
        num_parallel_calls=tf.data.AUTOTUNE)

    svg_high_ds = tf.data.Dataset.from_tensor_slices(svg_high_paths).map(
        lambda x: load_image(x, target_size=config.OUTPUT_SHAPE[:2]),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Final zip
    return tf.data.Dataset.zip(
        (tf.data.Dataset.zip((low_res_ds, high_res_ds)),
         tf.data.Dataset.zip((high_res_ds, svg_high_ds)))
    )



def get_data_pipeline(data_dir, batch_size=2, split="train"):
    dataset = create_dataset(data_dir, split)
    dataset = dataset.cache()
    dataset = dataset.shuffle(512)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
