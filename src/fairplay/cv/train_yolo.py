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
    Checks for the existence of the dataset.yaml file.
    
    Returns:
        Path to the generated dataset.yaml file.
    """
    dataset_yaml_path = data_dir / "dataset.yaml"
    if not dataset_yaml_path.exists():
        print(f"Error: 'dataset.yaml' not found at {dataset_yaml_path}.")
        print("Please run the `convert-dataset` script on this directory first.")
        return None

    with open(dataset_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # --- Update the 'path' in dataset.yaml to the current data_dir ---
    current_data_path = str(data_dir.resolve())
    if data_config.get('path') != current_data_path:
        print(f"Updating 'path' in dataset.yaml from '{data_config.get('path')}' to '{current_data_path}'")
        data_config['path'] = current_data_path
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)
        print(f"✅ Updated dataset.yaml with new base path.")
    else:
        print(f"✅ 'path' in dataset.yaml is already correct: '{current_data_path}'")

    # Validate paths listed in dataset.yaml
    for split_key in ['train', 'val', 'test']:
        if split_key in data_config:
            split_path = data_dir / data_config[split_key]
            if not split_path.is_dir():
                print(f"Error: Dataset '{split_key}' path '{split_path}' specified in 'dataset.yaml' does not exist.")
                print("Please ensure the directory exists or re-run `convert-dataset` to update `dataset.yaml`.")
                return None
    
    print(f"✅ Found and validated YOLO dataset configuration at: {dataset_yaml_path}")
    return dataset_yaml_path


def on_new_best_model(trainer): # This callback is only triggered if validation is run
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

    # Check for the pre-converted dataset
    dataset_yaml = prepare_yolo_dataset(data_dir)
    if not dataset_yaml:
        return

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
    
    # Determine if validation should be run by checking the yaml file
    with open(dataset_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    run_validation = 'val' in data_config or 'test' in data_config

    run_validation = 'val' in data_config # Only validate if a 'val' split is explicitly defined

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