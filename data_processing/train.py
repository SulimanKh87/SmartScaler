import os
# Reduce TensorFlow logging verbosity before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from models import build_generator, build_discriminator, build_vgg_loss_model
from data_loader import get_data_pipeline  # <- this function returns the dataset
from config import config
import datetime
import argparse

# Log device placement
tf.debugging.set_log_device_placement(True)

# Force TensorFlow to use only the first visible GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ Using GPU:", gpus[0])
    except RuntimeError as e:
        print("❌ RuntimeError while setting GPU:", e)
else:
    print("❌ No GPU detected. Training will run on CPU.")


class UpscalerTrainer:
    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        self.vgg_loss_model = build_vgg_loss_model()

        self.g_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.d_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(config.LOG_DIR, current_time)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    def perceptual_loss(self, y_true, y_pred):
        true_features = self.vgg_loss_model(y_true)
        pred_features = self.vgg_loss_model(y_pred)
        return tf.reduce_mean(tf.square(true_features - pred_features))

    @tf.function
    def train_step(self, inputs):
        (low_res, _), (high_res, svg_high) = inputs

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated = self.generator(low_res)

            real_output = self.discriminator(high_res)
            fake_output = self.discriminator(generated)

            pixel_loss = tf.reduce_mean(tf.square(high_res - generated))
            perc_loss = self.perceptual_loss(high_res, generated)
            svg_loss = tf.reduce_mean(tf.square(svg_high - generated))
            adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))

            total_gen_loss = (
                config.PIXEL_WEIGHT * pixel_loss +
                config.PERC_WEIGHT * perc_loss +
                config.SVG_WEIGHT * svg_loss +
                config.ADV_WEIGHT * adv_loss
            )

            disc_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output))
            disc_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output))
            total_disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)

        gen_gradients = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(total_disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return total_gen_loss, total_disc_loss, pixel_loss, perc_loss

    def train(self, train_ds, epochs):
        for epoch in range(epochs):
            for step, inputs in enumerate(train_ds):
                gen_loss, disc_loss, pixel_loss, perc_loss = self.train_step(inputs)

                if step % 10 == 0:
                    with self.summary_writer.as_default():
                        tf.summary.scalar('gen_loss', gen_loss, step=epoch * len(train_ds) + step)
                        tf.summary.scalar('disc_loss', disc_loss, step=epoch * len(train_ds) + step)
                        tf.summary.scalar('pixel_loss', pixel_loss, step=epoch * len(train_ds) + step)
                        tf.summary.scalar('perc_loss', perc_loss, step=epoch * len(train_ds) + step)

            print(f"Epoch {epoch + 1}/{epochs}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")

            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                self.generator.save(config.MODEL_SAVE_PATH)
                print(f"Model saved at epoch {epoch + 1}")

        self.generator.save(config.MODEL_SAVE_PATH)
        print("Training completed and model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size for training")
    args = parser.parse_args()

    # Load and optimize dataset pipeline
    train_ds = get_data_pipeline(args.data_dir, batch_size=args.batch_size)

    trainer = UpscalerTrainer()
    trainer.train(train_ds, args.epochs)
