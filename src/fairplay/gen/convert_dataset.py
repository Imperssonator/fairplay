import click
from fairplay.gen.generate_random_scatter import convert_to_yolo_format

@click.command()
@click.argument('base_folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(base_folder):
    """
    Converts an existing dataset with CSV bounding boxes to the YOLO .txt format.

    This script will:
    1. Find all 'train', 'val', and 'test' splits.
    2. Read the .csv bounding box files.
    3. Create corresponding .txt files in the YOLO format inside a 'labels/' subdirectory.
    4. Generate a 'dataset.yaml' file in the root of the dataset folder.

    BASE_FOLDER: The root directory of the generated dataset to convert.
    """
    convert_to_yolo_format(base_folder)

if __name__ == '__main__':
    main()