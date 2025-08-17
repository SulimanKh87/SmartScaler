Dataset Structure Explanation
For your 130 image pairs (original + SVG-rasterized at both 256x256 and 512x512), here's exactly how to split them:

# bash
mkdir -p dataset/{256,512}/{train,val,test}/{original,svg_rasterized}

Directory Breakdown
dataset/
├── 256/                    # 256x256 version
│   ├── train/              # Training set (100 pairs)
│   │   ├── original/       # 100 original 256x256 images
│   │   └── svg_rasterized/ # 100 matching SVG→raster conversions
│   ├── val/                # Validation set (20 pairs)
│   │   ├── original/       # 20 original
│   │   └── svg_rasterized/ # 20 matching SVG→raster
│   └── test/               # Test set (10 pairs)
│       ├── original/       # 10 original
│       └── svg_rasterized/ # 10 matching SVG→raster
└── 512/                    # Same structure for 512x512
    ├── train/
    │   ├── original/       # Same 100 originals at 512x512
    │   └── svg_rasterized/ # Their SVG→raster 512x512 versions
    ├── val/                # Same 20 pairs at 512x512
    └── test/               # Same 10 pairs at 512x512
File Count Verification
Resolution	Split	Original Images	SVG-Rasterized	Total Pairs
256x256	Train	100	100	100
256x256	Val	20	20	20
256x256	Test	10	10	10
512x512	Train	100	100	100
512x512	Val	20	20	20
512x512	Test	10	10	10
Key Notes
Same Images, Different Resolutions:

The 100 training images in 256/train/original/ and 512/train/original/ are the same images, just at different resolutions

Their corresponding SVG-rasterized versions maintain 1:1 pairing

Validation/Test Sets:

The 20 validation and 10 test images are separate from training data

These should be your most representative/cleanest images

File Naming Convention:
Ensure matching filenames between original and SVG-rasterized:

original/
├── image_001.png
├── image_002.png
└── ...

svg_rasterized/
├── image_001.png  # Matches original/image_001.png
├── image_002.png  # Matches original/image_002.png
└── ...
Python Code to Verify Pairing
python
from pathlib import Path

def verify_pairs(base_dir):
    for res in ['256', '512']:
        for split in ['train', 'val', 'test']:
            orig_path = Path(f"{base_dir}/{res}/{split}/original")
            svg_path = Path(f"{base_dir}/{res}/{split}/svg_rasterized")
            
            orig_files = sorted([f.name for f in orig_path.glob("*.png")])
            svg_files = sorted([f.name for f in svg_path.glob("*.png")])
            
            assert orig_files == svg_files, \
                f"Mismatch in {res}/{split}:\nOriginal: {orig_files}\nSVG: {svg_files}"
            
            print(f"✓ {res}/{split}: {len(orig_files)} pairs verified")

verify_pairs("dataset")
Why This Split?
Training (100 pairs): Maximum data for learning

Validation (20 pairs): Tune hyperparameters without overfitting

Test (10 pairs): Final unbiased evaluation

For your 130 total images, this gives you:

Training: 100 images (~77%)

Validation: 20 images (~15%)

Test: 10 images (~8%)

This follows machine learning best practices for small datasets while ensuring you have enough validation/test samples for reliable evaluation.