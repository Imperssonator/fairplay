import os
import argparse
from pathlib import Path
from datetime import datetime
import yaml

import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Configuration ---
def get_args():
    """Parses command-line arguments for YOLOv8 training."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 for plot object detection.")
    
    default_data_path = os.path.expanduser('~/data/mpl_scatter_1k')
    default_runs_path = os.path.expanduser('~/runs/fairplay_yolo')
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=default_data_path,
        help=f'Path to the dataset directory. Defaults to {default_data_path}'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=default_runs_path, 
        help='Parent directory to save YOLO runs.'
    )
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model to use (e.g., yolov8n.pt, yolov8s.pt).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for data loading.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest run in the output directory.')
    
    return parser.parse_args()


def prepare_yolo_dataset(data_dir: Path):
    """
    Converts the CSV bounding box data to YOLO's .txt format and creates the dataset.yaml file.
    
    Returns:
        Path to the generated dataset.yaml file.
    """
    print("Preparing dataset for YOLO training...")
    
    # --- 1. Find all unique classes and create a mapping ---
    # Also, determine which splits actually exist.
    all_classes = set()
    found_splits = []
    for split_name in ['train', 'val', 'test']:
        image_dir = data_dir / split_name
        if not image_dir.is_dir():
            continue
        
        found_splits.append(split_name)
        bbox_dir = data_dir / f"{split_name}_bboxes"
        for csv_file in bbox_dir.glob('*.csv'):
            df = pd.read_csv(csv_file)
            all_classes.update(df['class'].unique())
    
    class_to_id = {name: i for i, name in enumerate(sorted(list(all_classes)))}
    print(f"Found splits: {found_splits}")
    print(f"Found {len(class_to_id)} classes: {list(class_to_id.keys())}")
    
    # --- 2. Convert CSVs to YOLO .txt format for existing splits ---
    for split in found_splits:
        image_dir = data_dir / split
        bbox_dir = data_dir / f"{split}_bboxes"
        label_dir = data_dir / "labels" / split
        
        if not image_dir.is_dir():
            print(f"Skipping '{split}' split, directory not found.")
            continue
            
        label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting '{split}' split...")
        for img_file in tqdm(list(image_dir.glob('*.png'))):
            csv_file = bbox_dir / f"{img_file.stem}.csv"
            txt_file = label_dir / f"{img_file.stem}.txt"

            if not csv_file.exists():
                continue

            # Get image dimensions for normalization
            with Image.open(img_file) as img:
                img_w, img_h = img.size

            df = pd.read_csv(csv_file)
            
            with open(txt_file, 'w') as f:
                for _, row in df.iterrows():
                    class_id = class_to_id[row['class']]
                    
                    # Convert (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height)
                    box_w = row['x_max'] - row['x_min']
                    box_h = row['y_max'] - row['y_min']
                    x_center = row['x_min'] + box_w / 2
                    y_center = row['y_min'] + box_h / 2
                    
                    # Normalize
                    x_center_norm = x_center / img_w
                    y_center_norm = y_center / img_h
                    width_norm = box_w / img_w
                    height_norm = box_h / img_h
                    
                    f.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

    # --- 3. Create dataset.yaml ---
    dataset_yaml_path = data_dir / "dataset.yaml"
    yaml_content = {
        'path': str(data_dir.resolve()), # Ultralytics needs the absolute path
        'names': {v: k for k, v in class_to_id.items()}
    }
    # Dynamically add the splits that were found
    for split in found_splits:
        yaml_content[split] = split
    
    print(f"Generated YAML content: {yaml_content}")
    
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"✅ YOLO dataset preparation complete. Config file at: {dataset_yaml_path}")
    return dataset_yaml_path


def on_new_best_model(trainer):
    """
    Callback function to save prediction visualizations when a new best model is found.
    This is triggered at the end of each validation epoch.
    """
    # The 'trainer' object is an instance of ultralytics.engine.trainer.BaseTrainer
    # trainer.best_fitness is the best metric so far (lower is better for loss)
    # trainer.fitness is the metric for the current epoch
    if trainer.fitness == trainer.best_fitness:
        print("📸 New best model found! Generating prediction visualizations...")
        
        # Create a directory for visualizations if it doesn't exist
        visuals_dir = Path(trainer.save_dir) / "visuals"
        visuals_dir.mkdir(exist_ok=True)

        # Get a batch from the validation dataloader
        val_loader = trainer.validator.dataloader
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            print("Could not generate visualization: validation loader is empty.")
            return

        # Run prediction on the batch
        # The model is already on the correct device
        preds = trainer.model(batch['img'].to(trainer.device), verbose=False)

        # Plot and save up to 4 images from the batch
        num_to_visualize = min(len(preds), 4)
        for i in range(num_to_visualize):
            # The plot() method from ultralytics conveniently returns the image with boxes drawn
            img_with_boxes = preds[i].plot()
            
            # Convert from BGR (OpenCV) to RGB (matplotlib) for correct color display
            img_with_boxes_rgb = img_with_boxes[..., ::-1]

            # Save the image
            output_path = visuals_dir / f"epoch_{trainer.epoch+1}_best_pred_{i}.png"
            plt.imsave(output_path, img_with_boxes_rgb)


def main():
    args = get_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    # Prepare the dataset (conversion and YAML creation)
    dataset_yaml = prepare_yolo_dataset(data_dir)

    # Load the YOLO model
    model = YOLO(args.model)

    # Add the callback for visualizing predictions
    # 'on_fit_epoch_end' is triggered after each epoch's validation phase
    model.add_callback("on_fit_epoch_end", on_new_best_model)

    # Define run name
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Train the model
    print(f"\nStarting YOLOv8 training...")
    print(f"  - Project: {args.output_dir}")
    print(f"  - Run Name: {run_name}")
    print(f"  - Epochs: {args.epochs}, Batch Size: {args.batch_size}, Image Size: {args.img_size}")
    
    # Determine if validation should be run
    run_validation = 'val' in dataset_yaml.get('val', '') or 'test' in dataset_yaml.get('test', '')

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        val=run_validation, # Explicitly control validation
        imgsz=args.img_size,
        workers=args.num_workers,
        project=args.output_dir,
        name=run_name,
        resume=args.resume,
        exist_ok=False # Don't overwrite existing runs
    )

    print(f"\nTraining complete. Results saved in: {Path(args.output_dir) / run_name}")

if __name__ == "__main__":
    main()