# EECS 545 FINAL PROJECT

This is the final project**:  A Portable Correctly-Worn Face Mask Detection In Real-Time.**

This  project consist of two parts: Training, and implementation on Raspberry Pi

## Requirements:

- For training, we use python 3.8.10 with following libraries:
    - numpy
    - pandas
    - TensorFlow
    - MobileNetV2
    - OpenCV
    - scikit-learn
    - matplotlib
    - Seaborn
- For the implementation on Raspberry Pi:
    - Hardware:
        - Raspberry Pi 4B with 4GB RAM
        - Raspberry Pi camera module V2
    - Software:
        - Raspberry Pi OS
        - Python 3.9
        - MediaPipe
        - numpy
        - pandas
        - TensorFlow
        - MobileNetV2
        - OpenCV
        - Seaborn

## Usage:

### Training:

```bash
python train_mask_correctly_worn.py --dataset ../Dataset
```

### In Raspberry Pi:

```bash
python RPI_implementation.py
```