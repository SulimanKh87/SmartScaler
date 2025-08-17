class Config:
    # Data
    INPUT_SHAPE = (256, 256, 3)  # Low-res input
    OUTPUT_SHAPE = (512, 512, 3)  # High-res target
    HR_SHAPE = (512, 512, 3)

    # Model
    GEN_FILTERS = 64
    GEN_RES_BLOCKS = 5
    SCALE_FACTOR = 2  # 256→512 upscale

    # Training parameters
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 1
    SAVE_INTERVAL = 10
    MODEL_SAVE_PATH = "saved_models/upscaler.h5"
    LOG_DIR = "logs/"

    # Loss weights
    PIXEL_WEIGHT = 1.0
    PERC_WEIGHT = 1.0
    SVG_WEIGHT = 1.0
    ADV_WEIGHT = 1e-3

config = Config()