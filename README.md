# Fairplay
 Fairplay is a library built with the goal of enabling fully automated extraction of data from plots. One of the primary components is a plot simulator which can be used to generate training images for various tasks.

## Installation
Clone the repository. Then, from within `fairplay/`:
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

### Simulation Config

Simulation parameters are configured in `data/plot_params`. There are two key files:

* `continuous.csv`: parameters to define a truncated normal distribution from which to sample continuously-defined plotting arguments, like `marker_alpha`. Attributes are:
    * `min`: lowest allowable value
    * `max`: highest allowable value
    * `mean`: if specified, can place the mean of the truncated normal somewhere other than the midpoint of min and max, which is the default
    * `n_stds`: number of SD's of the normal distribution to "fit" between min and max. Default is 1, and if `mean` is also unspecified, this means the amplitude of the truncated normal PDF at `min` and `max` will be that of -1 and +1 SD. Higher values will result in more concentrated sampling near the mean and less at the edges.
* `discrete.csv`: literally-defined Lists on which to perform uniform sampling of discretely-defined plotting arguments, such as `marker_style`. To weight a member more heavily, simply add more copies of that member to the list. Very rudimentary.


## Example Output

RGB values for class labels (e.g. x ticks, markers, background) are defined as `label_colors` in `generate_random_scatter.py`

| Simulated             |  Labeled |
:-------------------------:|:-------------------------:
![simulated](data/demo/train/000014.png) | ![labeled](data/demo/train_labels/000014.png)
![simulated2](data/demo/train/000004.png) |  ![labeled2](data/demo/train_labels/000004.png)
![simulated3](data/demo/train/000013.png) |  ![labeled3](data/demo/train_labels/000013.png)



