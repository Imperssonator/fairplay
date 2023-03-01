# Fairplay
 Fairplay is a library for the procedural generation of random scientific charts and corresponding labeled images for the training of image segmentation models.

## Installation
Clone the repository. From within `fairplay/`:
```
pip install -e ./
```
Dependencies will be installed automatically.
Python >= 3.9 recommended.

## Usage
```
python ./src/fairplay/gen/generate_random_scatter.py ./data/demo -n 20 -t 10
```
**Arguments**
- `./data/demo`: directory to build image dataset and corresponding labeled images
- `-n 20`: 20 training images
- `-t 10`: 10 test images

## Example Output

RGB values for class labels (e.g. x ticks, markers, background) are defined as `label_colors` in `generate_random_scatter.py`

| Simulated             |  Labeled |
:-------------------------:|:-------------------------:
![simulated](data/demo/train/000014.png) | ![labeled](data/demo/train_labels/000014.png)
![simulated2](data/demo/train/000004.png) |  ![labeled2](data/demo/train_labels/000004.png)
![simulated3](data/demo/train/000013.png) |  ![labeled3](data/demo/train_labels/000013.png)



