# Fairplay
 Fairplay is a library for the procedural generation of random scientific charts.

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

### Simulated Plot
![simulated](data/demo/train/000014.png)

### With Class Labels
![labeled](data/demo/train_labels/000014.png)

RGB values for class labels (e.g. x ticks, markers, background) are defined as `label_colors` in `generate_random_scatter.py`
