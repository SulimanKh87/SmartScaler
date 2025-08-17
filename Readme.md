# 🧠 AI Super-Resolution Pipeline (TensorFlow + CUDA)

### UI: Users instructions to use the app (supports Windows x64):
# step 1:
install requirements: 
double click on install_requirements.bat
# step 2: 
use the UI (app):
double click on run_smartscaler.bat



### Readme.md Code review below:
This project performs image super-resolution and SVG conversion using TensorFlow with GPU acceleration (CUDA 11.8 + cuDNN 8.6).
---

Project directory:
project_root/
│
├── data_preprocessing/
│   ├── 
├── data_processing/
│   ├── data_loader.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── train.py
│   ├── models.py
│   ├── evaluate.py
│   ├── inference.py
│   
│
├── smartscaler_ui.py
├── saved_models/
│   └── upscaler.h5
├──streamlit_app/
│   ├── smartscaler_ui.py     # UI entry point
│   ├── ui_utils.py           # Helper functions for UI
│   └── ui_inference.py       # Inference logic (no training code)
├── venv_tf/ # Virtual environment for TensorFlow-related packages
├── venv_310/ # Virtual environment for other packages
├── requirementsTF_VENV.txt # Requirements for the venv_tf
├── requirementsVENV310.txt # Requirements for venv_310
├── run_smartscaler.bat              # For Windows
├── run_smartscaler.command          # For macOS
└── README.md # You're here!


------------------------------------------------------------------------------------------
## ⚙️ Setup Instructions

🧪 Virtual Environments:
# VENV Name	Use Case	                                                                                requirements file
 TF_VENV	    Used for Step 2: for modules: training, inference, evaluation, and model comparison (TensorFlow)	                requirementsTF_VENV.txt
 VENV310	    Used for Step 1: for modules: Alternate Python 3.10 environment, lighter version, fewer tools (no scikit-image/imageio)	requirementsVENV310.txt

### *** read below: PyCharm Setup – Interpreter Configuration ***
| #  | Module Name                | Purpose                                                         | Recommended VENV | Key Dependencies                                        |
| -- | -------------------------- | --------------------------------------------------------------- | ---------------- | ------------------------------------------------------- |
| 1  | `smart_resizer.py`         | Resize input images to 256x256 and 512x512 with quality control | `TF_VENV`        | `opencv-python`, `numpy`, `cupy`, `pandas`              |
| 2  | `raster_to_svg_256X256.py` | Convert 256x256 raster images to pixel-accurate SVGs            | `TF_VENV`        | `opencv-python`, `svgwrite`, `cupy`, `pandas`           |
| 3  | `raster_to_svg_512x512.py` | Convert 512x512 raster images to pixel-accurate SVGs            | `TF_VENV`        | `opencv-python`, `svgwrite`, `cupy`, `pandas`           |
| 4  | `svg_to_raster.py`         | Convert SVGs to transparent PNGs using CairoSVG                 | `VENV310`        | `cairosvg`, `pandas`, `Pillow`                          |
| 5  | `compare_roundtrip.py`     | Compare raster vs. SVG-roundtrip images using SSIM/PSNR/MSE     | `TF_VENV`        | `opencv-python`, `scikit-image`, `pandas`, `matplotlib` |
| 6  | `train.py`                 | Train upscaling model with perceptual and SVG loss              | `TF_VENV`        | `tensorflow`, `keras`, `cupy`, `pandas`                 |
| 7  | `evaluate.py`              | Evaluate model on val set using PSNR/SSIM                       | `TF_VENV`        | `tensorflow`, `pandas`, `numpy`                         |
| 8  | `inference.py`             | Use trained model to upscale a single low-res image             | `TF_VENV`        | `tensorflow`, `Pillow`, `numpy`                         |
| 9  | `compare_images.py`        | Compare predicted image to ground truth (SSIM, PSNR, MSE)       | `TF_VENV`        | `tensorflow`, `scikit-image`, `numpy`                   |
| 10 | `data_loader.py`           | Prepares data pipeline for train/eval from folder structure     | `TF_VENV`        | `tensorflow`, `pandas`, `numpy`, `pathlib`              |
| 11 | `models.py`                | Defines generator, discriminator, VGG loss model                | `TF_VENV`        | `tensorflow.keras`                                      |
| 12 | `config.py`                | Stores config constants: shapes, hyperparameters, loss weights  | `TF_VENV`        | Python built-in only                                    |
| 13 | `utils.py`                 | Preprocess, postprocess, visualize output images                | `TF_VENV`        | `opencv-python`, `tensorflow`, `numpy`                  |

Step A:
### 1. 🔧 Create and activate virtual environment (this project required TWO virtual environments!!!)
### 2. and install requirements for each virtual environment (follow the instructions below)
```powershell 
### recommended to use Pycharm Terminal

```
✅ Step A1: Set Python 3.9 for Your Project
After adding Python 3.9, make sure it's selected as the project interpreter in:
File → Settings → Project: <your_project> → Python Interpreter
✅ Step A2 : Create a New Virtual Environment Using Python 3.9
File → Settings → Python Interpreter
⚙️ → Add → Choose "New Virtual Environment"
Choose base interpreter: Python 3.9
Click OK — it will create a venv using Python 3.9

# hint: do the same for python 3.10
```

Step 1:
#first download python 3.10 from the internet (google it)
# If python points to another version, ensure you use Python 3.10 explicitly:
# is required for train.py

# use the commands: 
###  Virtual Environments installation (part 1):
python -m venv venv310 # only once for new project setup
.\venv310\Scripts\activate # use this bash command to active (use) venv310
# now install requirementsVENV310.txt 
# use the commands:
.\venv310\Scripts\activate
pip install -r requirementsVENV310.txt   #  Install dependencies

step 2: 
# now, download python 3.9
# this is required to run evaluate.py and interfernce.py
###  Virtual Environments installation (part 2):
py -3.9 -m venv tf_venv # only once for new project setup
.\tf_venv\Scripts\activate # TO ACTIVATE tf_venv
pip install -r  requirementsTF_VENV.txt #  Install dependencies

step 3: google  CUDA 11.8 and cuDNN 8.6 and CUDA toolkit 11.8
from the internet download then Install CUDA 11.8 and cuDNN 8.6
now download and Install CUDA Toolkit 11.8

for windows users:
Choose:
OS: Windows 64
Architecture: x86_64
choose Version between: 11 > 8

Step 4: google and Install cuDNN 8.6 for CUDA 11.x
google and find cuDNN
search and Download cuDNN 8.6 for CUDA 11.x
Unzip and copy contents to:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
Copy files into:
bin\ → cudnn64_8.dll
include\ → *.h
lib\x64\ → *.lib
```
--------------------------------------------------------------
```
# to use GPU (you can skip this part then you use CPU instead)
🛠️ Set Environment Variables (TEMPORARY) 
bash
# PowerShell session only
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;" + $env:PATH
python train.py --data_dir path/to/dataset 
🖼️ Run Inference
💡 Common Issues
Issue	Fix
GPU not detected	Check CUDA/cuDNN install, check environment variables
TensorFlow CUDA: False	Use TF 2.10.0, Python 3.10, and match Keras version
Import errors with Keras	Uninstall keras-nightly, use keras==2.10.0
Built with GPU: False	Reinstall TensorFlow with GPU support (2.10.0 only)
Final Note
Your GPU must support CUDA 11.8 and cuDNN 8.6 — if you see Built with CUDA: False, your TensorFlow version is likely incorrect, or environment variables are misconfigured.

END OF GPU usage
```
--------------------------------------------------------------------------------------------------------
# Step B:
### PyCharm Setup – Interpreter Configuration
To avoid errors, configure two interpreters:

# STEP-BY-STEP INSTRUCTIONS (Requires "Step A" completion):
Step 1: Verify Python 3.9 is Installed
Run in PowerShell or CMD:
bash
py -3.9 --version
Expected output:
Python 3.9.x

Step 2: Verify Python 3.10 Is Installed
py -3.10 --version
Expected:
Python 3.10.x

➤ Add venv310:
File → Settings → Project → Python Interpreter → Add
Point to venv310\\Scripts\\python.exe
Use for: train.py, svg_to_raster.py, general scripts

➤ Add tf_venv:
Add another interpreter
Point to tf_venv\\Scripts\\python.exe
Use for: inference.py, evaluate.py, compare_images.py
✅ You can switch per script in PyCharm via:

Right-click script → Run → Edit Configuration → Choose Interpreter

⚠️ Common Mistake: Wrong Python Version
TensorFlow 2.10 is only compatible with Python 3.9 or 3.10

If you're using Python 3.11+, you’ll get GPU or import errors
--------------------------------------------------------------
### 1 - data_preprocessing:

#  preparing data set
# step 1 image resizing
 python data_preprocessing/smart_resizer.py
```
Command Syntax
Usage:
bash
python data_preprocessing/smart_resizer.py --input_dir ./input --output_dir ./output --batch_size 8
This will create:

output/
├── input_raster_256x256/
│   ├── image1_256.png
│   ├── image2_256.png
│   └── ...
├── input_raster_512x512/
│   ├── image1_512.png
│   ├── image2_512.png
│   └── ...
├── processed_images.csv
└── skipped_images.csv

requirement for GPU (can skip to use CPU):
pip install cupy-cuda11x  # Replace with your CUDA version
```

""" WHY STEP1 image_resize was used? : """
INPUT KAGGLE BIRD IMAGES 187 RASTER IMAGES WITH VARRYING ASPECT RATIO:

1. Aspect Ratio Handling
Your SVGs have varying aspect ratios (e.g., 640×426, 1280×961, etc.). Scaling them to a fixed 512×512 will:

Non-Square SVGs (e.g., 640×427)

Stretch/Squish: The image will distort to fit 512×512, altering proportions.
Example: A tall hummingbird (640×459) becomes shorter/fatter.

Square SVGs (e.g., 640×640 or 1280×1280)

Perfect Fit: No distortion (scale factor = 0.8x for 640→512).

2. Visual Impact
Original Dimensions	Scaling Effect	Result
Wide (e.g., 640×360)	Horizontal compression	Birds look "squished" vertically
Tall (e.g., 441×640)	Vertical compression	Birds look "squished" horizontally
Square (e.g., 640×640)	Uniform scaling	No distortion
3. Technical Process
The script:

Calculates scale factors separately for width/height:

width_scale = 512 / original_width

height_scale = 512 / original_height

Applies scaling to all SVG elements:

Positions (x, y)

Sizes (width, height)

Strokes (averaged scale)

4. Example Transformations
Filename	Original Size	Scaled Size	Effect
005_bird-1045954_640.svg	640×426	512×512	19.6% wider
022_hummingbird-1823829_640.svg	441×640	512×512	16% taller
023_kingfisher-1068684_640.svg	640×640	512×512	Perfect (0.8x)
025_gull-1090835_640.svg	640×201	512×512	2.5× taller (extreme stretch)
5. Recommendations
To preserve aspect ratio, modify the script to:

Add Padding (letterboxing):

python
# Calculate scale to fit within 512×512 without distortion
scale = min(512/original_width, 512/original_height)
new_width = original_width * scale
new_height = original_height * scale
Center the Image with transparent/colored borders.

Or use cropping to fill 512×512 while maintaining proportions.

6. Expected Output Structure
output/scaled_colored_svgs/
├── scaled_005_bird-1045954_640.svg  # Now 512×512 (distorted)
├── scaled_022_hummingbird-1823829_640.svg  # 512×512 (distorted)
└── ...
Critical Notes
Quality Loss: Extreme scaling (e.g., 640×201 → 512×512) degrades details.

Stroke Widths: May appear too thick/thin if not scaled proportionally.

Metadata: The CSV will log original/scaled dimensions for audit purposes.

HOW TO SOLVE THIS??? 
USING smart_resizer.py # handles input. to allow only good images to fit 512x512 and 256x256 downscaling images only
-------------------------------------------------------------------

# step 2 - create svg images from raster images
 python data_preprocessing/raster_to_svg_256X256.py 
```
"""
This module converts raster images (e.g., PNG, JPG) into per-pixel SVG vector graphics,
while preserving the original aspect ratio with transparent padding.

Key Features:
-------------
- Maintains original aspect ratio with transparent padding
- Outputs consistent 256X256 SVG files with transparency
- Supports both GPU (CuPy) and CPU processing
- Preserves transparency from source images
- Generates detailed metadata for each conversion
- Processes images in parallel for better performance

Input:
-------
- A directory of raster images (PNG/JPG/JPEG) stored in: `output/input_raster_256X256`

Output:
--------
- 256X256 SVG files with transparent padding saved to: `output/svgs_256X256`
- Metadata CSV summarizing each conversion
"""
```
# step 3
 python data_preprocessing/raster_to_svg_512X512.py
```
"""
This module converts raster images (e.g., PNG, JPG) into per-pixel SVG vector graphics,
while preserving the original aspect ratio with transparent padding.

Key Features:
-------------
- Maintains original aspect ratio with transparent padding
- Outputs consistent 512x512 SVG files with transparency
- Supports both GPU (CuPy) and CPU processing
- Preserves transparency from source images
- Generates detailed metadata for each conversion
- Processes images in parallel for better performance

Input:
-------
- A directory of raster images (PNG/JPG/JPEG) stored in: `output/input_raster_512x512`

Output:
--------
- 512x512 SVG files with transparent padding saved to: `output/svgs_512x512`
- Metadata CSV summarizing each conversion
"""
```
 
# step 4 - SVG to Raster Converter

**Note**: This tool assumes input SVGs are already properly sized. For resizing SVGs, see our [SVG Smart Resizer](https://github.com/example/svg_resizer).


## Usage
### Basic Command
```bash
python data_preprocessing/svg_to_raster.py --svg_input output/svgs_512x512 --output_dir output/svg_rasterized_512x512
```

### Advanced Options
# 512X512 RESOLUTION
```bash
python data_preprocessing/svg_to_raster.py --svg_input output/svgs_512x512 --output_dir output/svg_rasterized_512x512 --workers 8  # Set number of parallel processes
```
# 256X256 RESOLUTION
```bash
python data_preprocessing/svg_to_raster.py --svg_input output/svgs_256X256 --output_dir output/svg_rasterized_256X256 --workers 8  # Set number of parallel processes

## Output Structure

```
output_directory/
├── converted_pngs/          # All PNG outputs
│   ├── image1.png
│   ├── image2.png
├── rasterization_report.csv         # Successful conversions
└── rasterization_report_errors.csv  # Failed conversions
```
```
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A high-performance tool for converting SVG vector graphics to PNG raster images while preserving quality and transparency.

## Features

- 🚀 **High-Speed Conversion**: Processes ~500 SVGs/minute
- 🖼️ **Quality Preservation**: Maintains vector sharpness in raster output
- 🔍 **Precision Rendering**: Exact 512x512 pixel output
- 📊 **Detailed Reporting**: CSV logs of all conversions
- ⚙️ **Parallel Processing**: Configurable worker threads
- 💧 **Transparency Support**: Maintains alpha channels

## Installation

1. **Prerequisites**:
   - Python 3.7+
   - Cairo graphics library (`libcairo2` on Linux)

2. **Install dependencies**:
```bash
pip install cairosvg pandas
```


## Report Formats

### Successful Conversions (`rasterization_report.csv`)
```csv
input_file,output_file,status,timestamp,resolution
design1.svg,design1.png,success,2023-08-20T14:30:00,512x512
```

### Failed Conversions (`rasterization_report_errors.csv`)
```csv
input_file,output_file,status,error,timestamp,resolution
broken.svg,,failed,"File format error",2023-08-20T14:31:00,
```

## Performance Tips

1. For best performance:
   - Use SSD storage
   - Set workers to 1.5× your CPU cores
   - Process SVGs in batches of <10,000 files

2. Expected performance:
   - 4-core CPU: ~300 files/minute
   - 8-core CPU: ~500 files/minute
   - 16-core CPU: ~900 files/minute

## Troubleshooting

**Common Issues:**
1. **Missing Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libcairo2-dev
   
   # MacOS
   brew install cairo
   ```

2. **Transparency Issues**:
   - Ensure SVGs use proper opacity attributes
   - Check for invalid clipping paths

3. **Performance Problems**:
   - Reduce number of workers
   - Process smaller batches
---------------------------------------------------------------------

# step 5 - SVG Roundtrip Image Quality Evaluation Tool
This script compares original raster images with their SVG-roundtrip 
rasterized versions to evaluate how much quality is retained during the 
conversion → vectorization → reconversion process.

🎯 Purpose
The goal is to verify the visual fidelity of your vectorization pipeline by computing:
🔬 PSNR – Peak Signal-to-Noise Ratio
🧠 SSIM – Structural Similarity Index
🧮 MSE – Mean Squared Error
Along with visual difference maps that highlight pixel-level changes.

How to Use
Run from the terminal:
# 256×256 Image Comparison  
# bash

python compare_roundtrip.py --originals output/input_raster_256x256 --svg_roundtrip output/svg_rasterized_256X256 --output output/raster_comparison_results_256x256
# 512×512 Image Comparison 
# bash
python compare_roundtrip.py --originals output/input_raster_512x512 --svg_roundtrip output/svg_rasterized_512x512 --output output/raster_comparison_results_512x512

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
≥30 dB	    Excellent quality
25–30 dB	Good quality
20–25 dB	Fair quality
<20 dB	    Poor quality

SSIM
Range	    Meaning
0.95–1.00	Virtually identical
0.90–0.95	Very similar
<0.90	    Noticeable change

MSE
Lower is better
0 means identical

📂 Output Files
After running, the output folder will contain:
roundtrip_comparison.csv → Per-image PSNR, SSIM, MSE
summary.txt → Aggregated quality statistics
diffs/*.jpg → Visual side-by-side comparisons
------------------------------------------------------------


# step 6 - compare the original 512×512 raster images with their SVG-rasterized versions  
# Is the SVG roundtrip conversion visually accurate and safe for training or deployment?
# How to Use (Bash)
python compare_svg_to_raster_results.py

# Output
After running, you'll get:
📊 output/comparison_report.csv: SSIM, MSE, match status
❌ output/comparison_report_errors.csv: if any files failed
🖼 output/comparison_results/: visual diff images (e.g. diff_owl.png)

# Why This Is Useful
Ensures SVG-based preprocessing doesn't degrade image quality
Helps you automatically detect bad conversions
Gives both numerical metrics (SSIM, MSE) and visual feedback
Essential for validating SVG use in ML datasets

-------------------------------------------
-------------------------------------------
-------------------------------------------

### 2 - data_processing:
# This folder contains the full TensorFlow pipeline for training, evaluating, and comparing a deep learning model that
# upscales images from 256×256 to 512×512, using optional SVG supervision.

Modules Overview & How to Use
🔧 train.py
Train the super-resolution model using raster + optional SVG data.
# bash
python data_processing/train.py --data_dir output/dataset --epochs 100
# example: If you have 100 training images and set epochs=50, 
# the model will go through all 100 images 50 times during training.
## this will output saved_models/upscaler.h5 

📊 evaluate.py
Evaluate the model using PSNR/SSIM on the validation set.
# bash
python data_processing/evaluate.py --data_dir output/dataset --model_path saved_models/upscaler.h5

📤 inference.py
Upscale a single image using a trained model.
# bash
python data_processing/inference.py --input path/to/256_image.png --output result.png
example:
python data_processing/inference.py --input output/dataset/256/test/original/152_kingfisher-3159334_1280_256.png --output result.png

🆚 compare_images.py
Compare an upscaled output image to its high-res ground truth using SSIM, PSNR, MSE.
# bash
python data_processing/compare_images.py --output output/test_output.png --target dataset/512/test/original/image.png


⚙️ data_loader.py
Used internally to:
Load 256×256 low-res
Load 512×512 high-res and SVG targets
Prepare TensorFlow datasets with prefetch, shuffle, batch
You don’t run this directly.

🧠 models.py
Defines the Generator, Discriminator, and VGG perceptual loss model. Used internally by train.py.

⚙️ config.py
Holds all configuration constants like image shape, learning rate, batch size, and loss weights.

🛠 utils.py
Utility functions for:
Image preprocessing/postprocessing
Visualization and saving images
Used internally by training and inference

------------------------------------------------

### Super-Resolution Pipeline – How to Use (Step 7–10)


## Step 7: Train the Model
Module: train.py
Env: TF_VENV
Why: Trains a deep learning model that learns to upscale 256×256 images into 512×512 high-res targets.

# bash
python data_processing/train.py --data_dir output/dataset --epochs 100 --batch_size 1

Depends On:
tensorflow, keras, vgg19 (→ TF_VENV)
Modules: models.py, data_loader.py, config.py

## Step 8: Evaluate the Model
Module: evaluate.py
Env: TF_VENV
Why: Runs the trained model on validation images and computes PSNR and SSIM.

# This script evaluates your trained super-resolution model on a validation set using 
# two important image quality metrics:
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)

# bash
python data_processing/evaluate.py --data_dir output/dataset --model_path saved_models/upscaler.h5

| Argument       | Description                                       | Default                    |
| -------------- | ------------------------------------------------- | -------------------------- |
| `--data_dir`   | Root dataset folder with 256x256 & 512x512 splits | **Required**               |
| `--model_path` | Path to your trained `.h5` model file             | `saved_models/upscaler.h5` |

Output: evaluation_results.csv
Output
Printed to Console:
Average PSNR (in dB)
Average SSIM (0.0 to 1.0)
Saved to Disk:
evaluation_results.csv — Per-image PSNR and SSIM values

Depends On:
tensorflow, pandas, numpy (→ TF_VENV)
Modules: data_loader.py, config.py



##  Step 9: Inference – Upscale a Single Image
# create upscaled raster image using the deep learning model we built
Module: inference.py
Env: TF_VENV

This script uses a trained super-resolution model to upscale a single 256×256 
input image into a 512×512 high-resolution output. Ideal for testing model output
visually or generating predictions for demo purposes.

# bash
python data_processing/inference.py --input output/dataset/256/test/original/sample.png --output output/test_output.png --model saved_models/upscaler.h5

| Argument   | Description                       | Default                    |
| ---------- | --------------------------------- | -------------------------- |
| `--input`  | Path to 256×256 input image       | **Required**               |
| `--output` | Path to save 512×512 output image | `upscaled_output.png`      |
| `--model`  | Path to trained model `.h5` file  | `saved_models/upscaler.h5` |


Depends On:
tensorflow, Pillow, numpy (→ TF_VENV)
Can be used with VENV310 if no visualizations needed

## Step 10: Compare Output to Ground Truth
Module: compare_images.py
Env: TF_VENV

This script compares an upscaled image generated by your model against its corresponding 
ground truth 512×512 target using:
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)
MSE (Mean Squared Error)

This helps you verify the pixel-level accuracy of a single prediction.
# bash
python data_processing/compare_images.py --output output/test_output.png --target output/dataset/512/test/original/sample.png

| Argument   | Description                                  | Required |
| ---------- | -------------------------------------------- | -------- |
| `--output` | Path to model-generated 512×512 image        | ✅        |
| `--target` | Path to ground truth 512×512 reference image | ✅        |

📥 Input Format
Both images must be:
In PNG or JPG format
RGB (3 channels)
--output and --target images must correspond to the same content

📊 Output
After execution, you'll find:
⬆️ Upscaled images: output_dir/*.png
📄 Evaluation report:
output_dir/batch_inference_evaluation.csv
CSV format:
Filename	        PSNR	SSIM	MSE
149_bird_256.png	31.87	0.9214	0.00073
150_crane_256.png	30.12	0.8947	0.00112

This script calls other scripts internally:
inference.py for upscaling
compare_images.py for metric evaluation
All subprocesses must print metrics in the expected format
Errors are caught and written as None in the CSV

Depends On:
scikit-image, tensorflow, numpy (→ TF_VENV)
Not available in VENV310

| Module              | TF | scikit-image | PIL | pandas | matplotlib |
| ------------------- | -- | ------------ | --- | ------ | ---------- |
| `train.py`          | ✅  | ❌            | ❌   | ❌      | ❌          |
| `evaluate.py`       | ✅  | ❌            | ❌   | ✅      | ❌          |
| `inference.py`      | ✅  | ❌            | ✅   | ❌      | ❌          |
| `compare_images.py` | ✅  | ✅            | ❌   | ❌      | ❌          |

The script auto-resizes the target image to match the dimensions of the output image if needed.
Internally converts both images to float32 [0, 1] and then back to uint8 [0, 255] for SSIM.
(The image is converted to uint8 [0, 255] before SSIM because
skimage's structural_similarity() is optimized for standard 8-bit images
and gives more accurate, consistent results in that format.)

Uses:
peak_signal_noise_ratio from skimage.metrics
structural_similarity (SSIM)
mean_squared_error from skimage.metrics

## Step 11:
Module: compare_images.py
Env: TF_VENV
batch_inference_with_compare_metrics.py
This script performs batch image upscaling using a trained
super-resolution model and automatically evaluates each result using:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
It also handles fallback resizing and logs both successful and failed predictions to CSV files.


What It Does:
This script performs batch super-resolution and evaluation:
Upscales all 256×256 test images using your trained model (inference.py)
Compares each result with its 512×512 ground truth using compare_images.py
Computes image quality metrics:
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)
MSE (Mean Squared Error)

The script will:
Find all *_256.png images
Run inference with your model
Run comparison using compare_images.py
Save results to:
output/test_output/batch_compare_metrics.csv

If any image fails to process, the error is logged in the CSV.
You can safely rerun the script — it overwrites the output folder.
GPU is automatically used if available and TensorFlow is properly installed.

batch_inference_evaluation.csv example:
| Filename       | PSNR  | SSIM   | MSE     |
| -------------- | ----- | ------ | ------- |
| image\_256.png | 32.12 | 0.9458 | 0.00087 |


Step 11: automated pipeline
How to Run:
# bash
python data_processing/batch_inference_with_compare_metrics.py

Folder Structure
The script assumes the following directory layout:
output/
├── dataset/
│   ├── 256/test/original/         # contains *_256.png images (input)
│   └── 512/test/original/         # contains *_512.png images (ground truth)
├── test_output/                   # will be created if not exists
│   ├── *_upscaled.png             # inference output
│   └── batch_compare_metrics.csv  # metrics saved here
saved_models/
└── upscaler.h5                    # your trained model

Filenames must follow this convention:
149_bird-9950_1280_256.png → upscales to → 149_bird-9950_1280_upscaled.png
Compares against → 149_bird-9950_1280_512.png

 Output CSV Example
Filename	                    PSNR	SSIM	MSE
149_bird-9950_1280_256.png	    32.14	0.9274	0.00071
150_crane-540657_1280_256.png	30.48	0.8892	0.00113

The CSV file will be overwritten each time you run this script.
Missing ground truth images (*_512.png) will be skipped with a warning.
All subprocesses use the same Python environment via sys.executable, so TensorFlow works correctly.
