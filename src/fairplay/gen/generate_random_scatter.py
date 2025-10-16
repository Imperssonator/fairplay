import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb

import os
import click
from glob import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml

import numpy as np
import scipy.stats as stats

from PIL import Image
import cv2
import copy

matplotlib.use('Agg')  #To make sure plots are not being displayed during the generation.
np.random.seed(0)  #For consistent data generation.


# SET UP PARAMETER SAMPLING ###
def discrete_sample(df, param):
    """
    Given a dataframe with column names corresponding
    to parameters and rows with discrete parameter options,
    return a uniformly random sampled parameter choice -
    probably a string.
    """
    
    return df[param].dropna().sample(1).iloc[0]


def continuous_sample(df, param):
    """
    Given a dataframe with index corresponding
    to parameter names and a column called "sampler"
    containing truncated normal sampler objects,
    return a sample from that parameter's distribution
    between its lower and upper bounds
    """
    
    return df.loc[param, 'sampler'].rvs(1)[0]


def trunc_norm_sampler(lower, upper, mu, n_stds):
    """
    Return a truncated normal distribution sampler object
    that will only return values between lower and upper
    with a normal pdf with a number of standard deviations
    between the midpoint and each edge equal to n_stds
    """
    
    if pd.isnull(mu):
        mu = np.mean([lower, upper])
    else:
        mu = float(mu)

    if pd.isnull(n_stds):
        n_stds = 1
    else:
        n_stds = float(n_stds)
    
    sigma = (upper - lower) / 2 / n_stds
    
    X = stats.truncnorm(
        (lower - mu) / sigma,
        (upper - mu) / sigma,
        loc=mu,
        scale=sigma
        )
    return X


def dist_sample(name, dfd, dfc, dfu):
    """
    'name' will either be a binary probability between 0 and 1
    or the name of a distribution in either dfc or dfd
    
    If it's a number, return True or False with p(True) = name
    If it's a string, find the row of dfc or column of dfd and sample it
    """
    
    try:
        thresh = float(name)
        return np.random.rand()<thresh

    except:
        if name in dfc.index.values:
            return continuous_sample(dfc, name)
        elif name in dfd.columns:
            return discrete_sample(dfd, name)
        elif name=="axis_label":
            return generate_random_axis_label(dfu)
        else:
            print('No distribution named {}'.format(name))
            return None


def build_kw_dict(kwcsv, dfd, dfc, dfu):
    """
    A kwcsv file has two columns: param and dist
    param refers to a field of kwargs for a matplotlib function
    dist refers to the name of a distribution
    distributions are either rows of dfd or columns of dfc
    """
    
    df = pd.read_csv(kwcsv)
    kw_dict = {
        p: dist_sample(d, dfd, dfc, dfu)
        for p, d
        in zip( df['param'], df['dist'] )
        }
    
    return kw_dict


# FIGURE GENERATION #

def generate_figure(figwidth=5, figaspect=1.25, dpi=150, facecolor='w'):
    """
    Generate a basic Figure object given width, aspect ratio, dpi
    and facecolor
    """
    
    figsize = (figwidth, figwidth / figaspect)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    plt.minorticks_on()
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    
    return fig, ax


# DATA GENERATION #

def power_data_gen(
    x_min=0,
    x_range=3,
    n_points=20,
    poly_scale=1,
    poly_deg=1,
    noise_std_prct=10
    ):
    """Generate a power-law data series

    Y = poly_scale * X ** (poly_deg) + Y_err

    where X is a vector of linear- or log-spaced points
    between 10^x_min and 10^x_min + 10^x_range.
    Linear spacing if x_range < 3, else log.

    Noise (Y_err) is normally distributed
    with sigma = Y * noise_std_prct / 100
    and is not applied to Y, but returned separately.

    Parameters
    ----------
    x_min: float
        log minimum of x range (actual min will be 10^x_min)
    x_range: float
        log range of x values (max will be 10^x_min + 10^x_range)
    n_points: int
        number of points in simulated data series
    
    Returns
    -------
    X: np.array(float)
    Y: np.array(float)
    Y_err: np.array(float)
    x_spacing: str
        'linear' or 'log'
    y_spacing: str
        'linear' or 'log'
    """

    log_cutoff = 3
    
    if x_range > log_cutoff:
        x_spacing = 'log'
        X = np.logspace(
            x_min,
            x_min + x_range,
            int(n_points)
            )
    else:
        X = np.linspace(
            10 ** x_min,
            10 ** x_min + 10 ** x_range,
            int(n_points)
            )
        x_spacing = 'linear'
    
    Y = poly_scale * X ** poly_deg
    
    if (
        (max(Y) * min(Y) < 0)
        or (np.abs(np.log10(max(Y) / min(Y))) < log_cutoff)
    ):
        y_spacing = 'linear'
    else:
        y_spacing = 'log'

    if y_spacing == 'log' and np.any(Y < 0):
        Y = np.abs(Y)
    
    Y_err = np.random.normal(
        loc=np.zeros(Y.shape),
        scale=np.abs(Y * noise_std_prct / 100)
        )
    
    return X, Y, Y_err, x_spacing, y_spacing


def generate_random_axis_label(dfu):
    """Using units of measure dataset, generate a random axis label"""
    if np.random.rand() < 0.5:
        lb = "("
        rb = ")"
    else:
        lb = "["
        rb = "]"
    
    dfu_row = dfu.sample(1).iloc[0]
    title = dfu_row.Name
    units = dfu_row.Symbol
    axis_label = f"{title} {lb}{units}{rb}"

    return axis_label


# FULL PLOT GENERATION #

def generate_training_plot(
    data_folder,
    id_str,
    label_colors,
    dfd,
    dfc,
    dfu
):
    """
    Given a folder and the ID# for a new random plot, generate it and stick
    it in the folder
    """

    # GENERATE FIGURE #
    fig_kwargs = build_kw_dict('data/plot_params/fig_properties.csv', dfd, dfc, dfu)
    fig, ax = generate_figure(**fig_kwargs)

    # PLOT DATA #
    data_kwargs = build_kw_dict('data/plot_params/data_gen.csv', dfd, dfc, dfu)
    marker_kwargs = build_kw_dict('data/plot_params/marker_styles.csv', dfd, dfc, dfu)
    X, Y, Ye, x_spacing, y_spacing = power_data_gen(**data_kwargs)
    ax.plot(X, Y + Ye, **marker_kwargs)
    ax.set_xscale(x_spacing)
    ax.set_yscale(y_spacing)

    # ERROR BARS #
    if np.random.rand() > 0.5:
        error_kwargs = build_kw_dict('data/plot_params/errorbar_styles.csv', dfd, dfc, dfu)
        error_kwargs['linestyle'] = 'None'
        ax.errorbar(
            X,
            Y + Ye,
            yerr=np.abs(Y * data_kwargs['noise_std_prct'] / 100),
            **error_kwargs
        )

    # BOX AND GRID #
    plt_bools = build_kw_dict('data/plot_params/plt_boolean.csv', dfd, dfc, dfu)
    for k, v in plt_bools.items():
        eval('plt.{}({})'.format(k, v))
    plt.grid(True)  # Whether or not grid shows up gets varied in tick_params

    # TICKS #
    tick_param_kwargs = build_kw_dict('data/plot_params/tick_params_major.csv', dfd, dfc, dfu)
    ax.tick_params(which='major', **tick_param_kwargs)
    tick_param_minor_kwargs = build_kw_dict('data/plot_params/tick_params_minor.csv', dfd, dfc, dfu)
    ax.tick_params(which='minor', **tick_param_minor_kwargs)

    # TICK LABELS #
    tick_font_kwargs = build_kw_dict('data/plot_params/font_properties.csv', dfd, dfc, dfu)
    tick_font = font_manager.FontProperties(**tick_font_kwargs)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(tick_font)

    # AXIS LABELS #
    plt.xlabel(generate_random_axis_label(dfu), color=tick_param_kwargs["labelcolor"])
    plt.ylabel(generate_random_axis_label(dfu), color=tick_param_kwargs["labelcolor"])
    ax.xaxis.get_label().set_font_properties(tick_font)
    ax.yaxis.get_label().set_font_properties(tick_font)

    # LEGEND #
    if np.random.rand() < 0.35:
        # legend_kwargs = build_kw_dict('data/plot_params/legend.csv')
        plt.legend(
            [dfu.sample(1).iloc[0].Name[:15]],
            prop=tick_font,
            labelcolor=tick_param_kwargs["labelcolor"]
        )

    plt.tight_layout()
    
    # SAVE RAW AND LABELED IMAGES #
    fig.savefig(
        '{}/{}.png'.format(data_folder, id_str),
        facecolor=fig.get_facecolor(),
        edgecolor='none'
    )
    label_img_array = generate_label_image(fig, ax, label_colors)
    label_img = Image.fromarray(label_img_array)
    label_img.save('{}/{}.png'.format(data_folder + '_labels', id_str))

    # SAVE BOUNDING BOXES
    fig.canvas.draw() # Ensure the figure is rendered
    bboxes = generate_bounding_boxes(fig, ax)
    df_bboxes = pd.DataFrame([(k, *v) for k, val in bboxes.items() for v in val], columns=['class', 'x_min', 'y_min', 'x_max', 'y_max'])
    df_bboxes.to_csv('{}/{}.csv'.format(data_folder + '_bboxes', id_str), index=False)

    
    return fig, ax

def generate_bounding_boxes(fig, ax):
    """
    Given the Figure and Axes objects of a random plot,
    return a dictionary of bounding boxes for each plot element.
    The bounding boxes are in the format [x_min, y_min, x_max, y_max].
    """
    bboxes = {}
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)

    # Markers
    marker_boxes = []
    for line in ax.lines:
        if line.get_marker() and line.get_marker() != 'None':
            # For markers, get_window_extent() is not reliable.
            # We transform data coordinates to display coordinates.
            xy_data = line.get_xydata()
            xy_pixels = ax.transData.transform(xy_data)
            markersize = line.get_markersize()
            # markersize is in points (1/72 inch). Convert to pixels.
            box_size_px = markersize * fig.dpi / 72.0
            for x_pix, y_pix in xy_pixels:
                x0 = x_pix - box_size_px / 2.0
                y0 = y_pix - box_size_px / 2.0
                x1 = x_pix + box_size_px / 2.0
                y1 = y_pix + box_size_px / 2.0
                # Invert y-axis to match image coordinates (origin top-left)
                h = fig.get_figheight() * fig.dpi
                marker_boxes.append([x0, h - y1, x1, h - y0])
    bboxes['markers'] = marker_boxes

    # Error Bars
    error_boxes = []
    for container in ax.containers:
        if isinstance(container, matplotlib.container.ErrorbarContainer):
            for line_collection in container.lines:
                # line_collection can be a LineCollection or a tuple of Line2D
                if isinstance(line_collection, matplotlib.collections.LineCollection):
                    for path in line_collection.get_paths():
                        # path.get_extents() gives bbox in data coords
                        bbox_data = path.get_extents()
                        # transform to display coords
                        bbox_disp = bbox_data.transformed(ax.transData)
                        h = fig.get_figheight() * fig.dpi
                        error_boxes.append([bbox_disp.x0, h - bbox_disp.y1, bbox_disp.x1, h - bbox_disp.y0])
    bboxes['error_bars'] = error_boxes

    # Ticks, Tick Labels, and Axis Labels
    for aa in ['x', 'y']:
        axis = getattr(ax, f'{aa}axis')
        
        # Tick marks
        tick_boxes = []
        for tick in axis.get_major_ticks():
            # A tick is only visible if its location is within the axis limits
            loc = tick.get_loc()
            vmin, vmax = axis.get_view_interval()
            if tick.get_visible() and vmin <= loc <= vmax:
                tick_bbox = tick.tick1line.get_window_extent(renderer)
                # Ensure the tick is actually within the axes bounds
                if tick_bbox.width > 0 and tick_bbox.height > 0 and ax_bbox.overlaps(tick_bbox):
                     h = fig.get_figheight() * fig.dpi
                     tick_boxes.append([tick_bbox.x0, h - tick_bbox.y1, tick_bbox.x1, h - tick_bbox.y0])
        bboxes[f'{aa}_ticks'] = tick_boxes
        
        # Tick labels
        tick_label_boxes = []
        # We iterate through the ticks, not the labels, to check their location.
        for tick in axis.get_major_ticks():
            loc = tick.get_loc()
            vmin, vmax = axis.get_view_interval()
            if tick.label1.get_visible() and tick.label1.get_text() and vmin <= loc <= vmax:
                bbox = tick.label1.get_window_extent(renderer)
                h = fig.get_figheight() * fig.dpi
                tick_label_boxes.append([bbox.x0, h - bbox.y1, bbox.x1, h - bbox.y0])
        bboxes[f'{aa}_tick_labels'] = tick_label_boxes

        # Axis label
        axis_label = axis.get_label()
        bbox = axis_label.get_window_extent(renderer)
        h = fig.get_figheight() * fig.dpi
        bboxes[f'{aa}_axis_label'] = [[bbox.x0, h - bbox.y1, bbox.x1, h - bbox.y0]]

    return bboxes

def generate_label_image(fig, ax, label_colors):
    """
    This somehow turned out more complicated than plot generation...
    Given the Figure and Axes objects of a random plot,
    and label_colors {'plot element': [r, g, b] as uint8}
    Return label_image, an image (numpy array) where the pixels representing
    each plot component have been labeled according to the provided colors (label_colors)
    so it can be used as input to Semantic Segmentation Suite... which I am not using anymore
    Also df_lc: dataframe of label colors that can be dumped to csv for the dataset
    """
    mask_dict = {}
    # probably need some defensive code to check the label_colors dict
    bg_color = np.array([int(c * 255) for c in fig.get_facecolor()])[:3].astype(np.uint8)
    kids = ax.get_children()

    # MARKERS #
    visible = [0, 1]
    for i in range(len(kids)):
        if i not in visible:
            kids[i].set_visible(False)
        else:
            kids[i].set_visible(True)
            kids[i].set_linestyle('None')

    fig.canvas.draw()
    class_img = np.array(fig.canvas.renderer._renderer)[:, :, :3]
    mask_dict['markers'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)

#     # ERROR BARS #
#     visible = [0,3,4]
#     for i in range(len(kids)):
#         if i not in visible:
#             kids[i].set_visible(False)
#         else:
#             kids[i].set_visible(True)

#     fig.canvas.draw()
#     class_img = np.array(fig.canvas.renderer._renderer)[:,:,:3]
#     mask_dict['error_bars'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)

    # TICKS, TICK LABELS AND AXIS LABELS #
    
    for aa in ['x', 'y']:
        axis = eval('ax.{}axis'.format(aa))
        mlf = copy.copy(axis.get_major_formatter())

        # Make only the _axis visible
        [k.set_visible(False) for k in kids]
        axis.set_visible(True)

        # Make only the major ticks visible
        [t.set_visible(False) for t in axis.get_minor_ticks()]
        [g.set_visible(False) for g in axis.get_gridlines()]  # Make gridlines invisible
        axis.get_label().set_visible(False)
        axis.set_major_formatter(plt.NullFormatter())  # This makes the tick labels invisible

        # Generate tick mask
        fig.canvas.draw_idle()
        class_img = np.array(fig.canvas.renderer._renderer)[:, :, :3]
        mask_dict[aa + '_ticks'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)

        # Make only the tick labels visible
        axis.set_major_formatter(mlf)  # This brings back the tick labels
        [
            [ch.set_visible(False) for ch in tick.get_children() if not hasattr(ch, '_text')]
            for tick in axis.get_major_ticks()
            ]

        # Generate label mask
        fig.canvas.draw_idle()
        class_img = np.array(fig.canvas.renderer._renderer)[:, :, :3]
        mask_dict[aa + '_tick_labels'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)

        # Make only the axis labels visible
        # axis.set_major_formatter(plt.NullFormatter())  # This makes the tick labels invisible
        axis.set_tick_params(which="major", labelcolor=tuple(bg_color / 255))  # No, THIS makes the tick labels "invisible".. but it is irreversible
        axis.get_label().set_visible(True)

        # Generate axis label mask
        fig.canvas.draw_idle()
        class_img = np.array(fig.canvas.renderer._renderer)[:, :, :3]
        cv2.imwrite('temp/label_test.png', class_img)
        mask_dict[aa + '_axis_label'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)
        
        # Reset visibilities
        [k.set_visible(True) for k in kids]
        [t.set_visible(True) for t in axis.get_major_ticks()]
        [t.set_visible(True) for t in axis.get_minor_ticks()]
        [g.set_visible(True) for g in axis.get_gridlines()]
        axis.set_major_formatter(mlf)  # This brings back the tick labels
        axis.get_label().set_visible(True)


    # FINAL LABEL IMAGE #
    label_image = np.zeros(class_img.shape).astype(np.uint8)
    for kk, mm in mask_dict.items():
        label_image = set_color_mask(label_image, mm, label_colors[kk])
    bg_mask = np.all(label_image == np.zeros(3).astype(np.uint8), axis=-1)
    label_image = set_color_mask(label_image, bg_mask, label_colors['background'])

    return label_image


def str2color(color_string):
    """ Convert color string to uint8 array representation """
    return (np.array(to_rgb(color_string)) * 255).astype(np.uint8)


def set_color_mask(A, M, c):
    """ Given image array A (h, w, 3) and mask M (h, w, 1),
    Apply color c (3,) to pixels in array A at locations M
    """
    for i in range(3):
        A_i = A[:, :, i]
        A_i[M] = c[i]
        A[:, :, i] = A_i
    return A


def get_distribution_configs(
    discrete_path="data/plot_params/discrete.csv",
    continuous_path="data/plot_params/continuous.csv",
    units_path="data/plot_params/units-of-measure.csv"
):
    """Retrieve the distribution config dataframes"""

    # DISCRETE PARAMETERS
    dfd = pd.read_csv(discrete_path)
    dfu = pd.read_csv(units_path)
    dfu = dfu.dropna(subset="Symbol")
    dfu = dfu[dfu["Name"].str.len() < 30]  # Don't want super long axis labels

    # CONTINUOUS PARAMETERS
    dfc = pd.read_csv(continuous_path, index_col='param')
    dfc['sampler'] = \
        dfc.apply(
            lambda row:
                trunc_norm_sampler(
                    row['min'],
                    row['max'],
                    row['mean'],
                    row['n_stds']
                    ),
            axis=1)
    return dfd, dfc, dfu


def display_bounding_boxes(base_folder, dataset, plot_id):
    """
    Displays a plot with its ground truth bounding boxes overlaid.

    Args:
        base_folder (str): The root folder of the dataset.
        dataset (str): The dataset split, e.g., 'train' or 'test'.
        plot_id (str): The identifier for the plot (e.g., '000000').
    """

    img_path = os.path.join(base_folder, dataset, f"{plot_id}.png")
    bbox_path = os.path.join(base_folder, f"{dataset}_bboxes", f"{plot_id}.csv")

    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
    if not os.path.exists(bbox_path):
        print(f"Error: Bounding box data not found at {bbox_path}")
        return

    # Load the image and bounding box data
    img = Image.open(img_path)
    df_bboxes = pd.read_csv(bbox_path)

    # Create a figure to display the image
    fig, ax = plt.subplots(1, figsize=(12, 12 * (img.height / img.width)))
    ax.imshow(img)

    # Create a color map for the different classes
    unique_classes = sorted(df_bboxes['class'].unique())
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(unique_classes)))
    class_to_color = {cls: color for cls, color in zip(unique_classes, colors)}

    # Add each bounding box to the plot
    for _, row in df_bboxes.iterrows():
        x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
        width = x_max - x_min
        height = y_max - y_min
        rect = matplotlib.patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=1.5, edgecolor=class_to_color[row['class']], facecolor='none', label=row['class']
        )
        ax.add_patch(rect)

    ax.axis('off')
    plt.tight_layout()
    plt.show()

def convert_to_yolo_format(base_folder: str):
    """
    Converts the generated CSV bounding box data to YOLO's .txt format
    and creates the dataset.yaml file. This is run after the main
    dataset generation is complete.
    """
    print("\nConverting dataset to YOLO format...")
    data_dir = Path(base_folder)

    # --- 1. Find all unique classes and create a mapping ---
    all_classes = set()
    found_splits = []
    for split_name in ['train', 'val', 'test']:
        image_dir = data_dir / split_name
        if not image_dir.is_dir():
            continue
        
        found_splits.append(split_name)
        bbox_dir = data_dir / f"{split_name}_bboxes"
        if not bbox_dir.is_dir():
            continue
        for csv_file in bbox_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                if 'class' in df.columns:
                    all_classes.update(df['class'].unique())
            except pd.errors.EmptyDataError:
                # It's possible to have an image with no bounding boxes, so the csv is empty.
                pass

    if not all_classes:
        print("Warning: No classes found. Skipping YOLO format conversion.")
        return

    class_to_id = {name: i for i, name in enumerate(sorted(list(all_classes)))}
    print(f"Found {len(class_to_id)} classes for YOLO conversion: {list(class_to_id.keys())}")

    # --- 2. Convert CSVs to YOLO .txt format for existing splits ---
    for split in found_splits:
        image_dir = data_dir / split
        bbox_dir = data_dir / f"{split}_bboxes"
        label_dir = data_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting '{split}' split to YOLO format...")
        for img_file in tqdm(list(image_dir.glob('*.png'))):
            csv_file = bbox_dir / f"{img_file.stem}.csv"
            txt_file = label_dir / f"{img_file.stem}.txt"

            if not csv_file.exists(): continue

            with Image.open(img_file) as img:
                img_w, img_h = img.size

            df = pd.read_csv(csv_file)
            with open(txt_file, 'w') as f:
                for _, row in df.iterrows():
                    class_id = class_to_id[row['class']]
                    box_w = row['x_max'] - row['x_min']
                    box_h = row['y_max'] - row['y_min']
                    x_center = (row['x_min'] + box_w / 2) / img_w
                    y_center = (row['y_min'] + box_h / 2) / img_h
                    width_norm = box_w / img_w
                    height_norm = box_h / img_h
                    f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")

    # --- 3. Create dataset.yaml ---
    dataset_yaml_path = data_dir / "dataset.yaml"
    yaml_content = {'path': str(data_dir.resolve()), 'names': {v: k for k, v in class_to_id.items()}}
    for split in found_splits:
        yaml_content[split] = split
    
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"✅ YOLO dataset preparation complete. Config file at: {dataset_yaml_path}")

@click.command()
@click.argument('base_folder', type=click.Path())
@click.option('--num-train', '-n', type=int, default=1000)
@click.option('--num-val', '-v', type=int, default=400) # Default to 400 to ensure validation set by default
@click.option('--num-test', '-t', type=int, default=0)  # Default to 0 to make test set optional
def generate_dataset(
    base_folder,
    num_train=1000,
    num_val=400,
    num_test=0
):
    """Main function for plot dataset generation"""
    os.makedirs(base_folder, exist_ok=True)

    # SET LABEL PIXEL COLORS FOR GROUND TRUTH MASKS
    label_colors = {
        'markers': str2color('xkcd:blue'),
        'x_ticks': str2color('xkcd:dark red'),
        'x_tick_labels': str2color('xkcd:red'),
        'x_axis_label': str2color('xkcd:light red'),
        'y_ticks': str2color('xkcd:forest green'),
        'y_tick_labels': str2color('xkcd:green'),
        'y_axis_label': str2color('xkcd:light green'),
        'error_bars': str2color('xkcd:dark grey'),
        'background': str2color('xkcd:eggshell')
        }
    # label_ints = {
    #     'markers': 0,
    #     'x_ticks': 1,
    #     'x_tick_labels': 2,
    #     'x_axis_label': 3,
    #     'y_ticks': 4,
    #     'y_tick_labels': 5,
    #     'y_axis_label': 6,
    #     'error_bars': 7,
    #     'background': 8
    #     }

    df_lc = pd.DataFrame.from_dict(label_colors).transpose().reset_index()
    df_lc.columns = ['name', 'r', 'g', 'b']
    df_lc.to_csv(os.path.join(base_folder, 'class_dict.csv'), index=False)

    # GET DISTRIBUTION CONFIGS
    dfd, dfc, dfu = get_distribution_configs()

    # GENERATE PLOT IMAGES AND CLASS LABEL IMAGES
    splits_to_generate = [s for s in ['train', 'val', 'test'] if eval(f'num_{s}') > 0]
    for dataset in splits_to_generate:
        os.makedirs(os.path.join(base_folder, dataset), exist_ok=True)
        os.makedirs(os.path.join(base_folder, dataset + '_labels'), exist_ok=True)
        os.makedirs(os.path.join(base_folder, dataset + '_bboxes'), exist_ok=True)

    for dataset in splits_to_generate:
        print('Generating ', dataset)
        num_to_gen = eval('num_' + dataset)
        for i in tqdm(range(eval('num_' + dataset))):
            data_folder = os.path.join(base_folder, dataset)
            fig, ax = generate_training_plot(
                data_folder,
                str(i).zfill(6),
                label_colors,
                dfd,
                dfc,
                dfu
            )
            plt.close(fig)

    # Convert the generated dataset to YOLO format
    convert_to_yolo_format(base_folder)

    return


if __name__ == '__main__':
    generate_dataset()
