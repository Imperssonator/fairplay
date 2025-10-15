import os
import random
import click
import matplotlib

from fairplay.gen.generate_random_scatter import display_bounding_boxes


@click.command()
@click.argument('base_folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--dataset', '-d', type=click.Choice(['train', 'test', 'val']), default='train', help="Dataset split to visualize.")
@click.option('--num-images', '-n', type=int, default=3, help="Number of random images to display.")
@click.option('--backend', '-b', type=str, default=None, help="Matplotlib backend to use (e.g., 'TkAgg', 'Qt5Agg').")
def verify(base_folder, dataset, num_images, backend):
    """
    Pulls random images from a dataset and displays them with their
    bounding boxes overlaid for verification.

    BASE_FOLDER: The root directory of the generated dataset.
    """
    if backend:
        matplotlib.use(backend)
    else:
        # Try to find a suitable interactive backend
        try:
            matplotlib.use('TkAgg')
        except ImportError:
            try:
                matplotlib.use('Qt5Agg')
            except ImportError:
                print("Warning: Could not find 'TkAgg' or 'Qt5Agg' backend.")
                print("You may need to install a GUI toolkit for matplotlib, e.g., `pip install pyqt5`.")

    image_folder = os.path.join(base_folder, dataset)
    if not os.path.isdir(image_folder):
        print(f"Error: Image directory not found at '{image_folder}'")
        return

    all_images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    if len(all_images) == 0:
        print(f"Error: No images found in '{image_folder}'")
        return

    num_to_show = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_show)

    for image_file in selected_images:
        plot_id = os.path.splitext(image_file)[0]
        print(f"Displaying plot: {plot_id}")
        display_bounding_boxes(base_folder, dataset, plot_id)

if __name__ == '__main__':
    verify()