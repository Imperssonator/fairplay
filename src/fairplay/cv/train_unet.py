import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp


# --- Configuration ---
def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train UNet for plot segmentation.")
    
    # Use os.path.expanduser to handle '~'
    default_data_path = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-imperssonator@gmail.com/My Drive/data/231230-mpl5k')
    default_runs_path = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-imperssonator@gmail.com/My Drive/runs/fairplay')
    
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
        help='Directory to save outputs (checkpoints, logs, visualizations).'
    )
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--img_size', type=int, default=512, help='Image size for resizing.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes for data loading.')
    
    return parser.parse_args()


# --- Custom Dataset ---
class PlotSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for loading plot images and their segmentation masks.
    """
    def __init__(self, base_dir, split='train', transforms=None, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.transforms = transforms
        
        self.image_dir = base_dir / split
        self.mask_dir = base_dir / f"{split}_labels"
        
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        # Load class dictionary and create RGB to integer mapping
        class_df = pd.read_csv(base_dir / 'class_dict.csv')
        self.class_rgb_values = [tuple(row[['r', 'g', 'b']]) for _, row in class_df.iterrows()]
        
        print(f"Found {len(self.image_files)} images in '{split}' set.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # Resize using NEAREST for mask to avoid introducing new colors
        image = F.resize(image, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.img_size, self.img_size], interpolation=T.InterpolationMode.NEAREST)

        # Convert RGB mask to integer class mask
        mask_np = np.array(mask)
        int_mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        for i, rgb in enumerate(self.class_rgb_values):
            matches = np.all(mask_np == rgb, axis=-1)
            int_mask[matches] = i
        
        mask = Image.fromarray(int_mask.astype(np.uint8))

        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        return image, mask


# --- Data Augmentation ---
class SegmentationAugmentations:
    """
    Applies the same set of random augmentations to both image and mask.
    """
    def __init__(self, is_train=True, img_size=512):
        self.is_train = is_train
        self.img_size = img_size

    def __call__(self, image, mask):
        # Apply shared transformations
        if self.is_train:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            
            # Random rotation
            angle = T.RandomRotation.get_params([-30, 30])
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

            # Random affine (includes scaling/zoom)
            scale_factor = torch.rand(1).item() * 0.4 + 0.8 # Scale between 0.8x and 1.2x
            affine_params = T.RandomAffine.get_params(
                degrees=(0, 0), translate=None, scale_ranges=(scale_factor, scale_factor), 
                shears=None, img_size=image.size
            )
            image = F.affine(image, *affine_params)
            mask = F.affine(mask, *affine_params, interpolation=T.InterpolationMode.NEAREST)

        # Convert to tensor (scales to [0, 1]) and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Convert mask to tensor without scaling
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return image, mask


# --- Visualization ---
def visualize(output_dir, image, true_mask, pred_mask, class_rgb_values):
    """
    Saves a visualization of input, true mask, and predicted mask.
    """
    def to_rgb(mask_tensor):
        """Converts an integer mask tensor to an RGB image for visualization."""
        output = np.zeros((mask_tensor.shape[0], mask_tensor.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(class_rgb_values):
            output[mask_tensor == i] = color
        return Image.fromarray(output)

    # Denormalize image for display
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image = inv_normalize(image.cpu()).permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    
    true_mask_rgb = to_rgb(true_mask.cpu().numpy())
    pred_mask_rgb = to_rgb(pred_mask.cpu().numpy())
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis('off')

    ax2.imshow(true_mask_rgb)
    ax2.set_title("True Mask")
    ax2.axis('off')
    
    ax3.imshow(pred_mask_rgb)
    ax3.set_title("Predicted Mask")
    ax3.axis('off')

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(output_dir / f"prediction_{timestamp}.png", dpi=300)
    plt.close(fig)


# --- Training Loop ---
def main():
    args = get_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return
        
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoints_dir = output_dir / "checkpoints"
    visuals_dir = output_dir / "visuals"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Datasets and DataLoaders
    train_transforms = SegmentationAugmentations(is_train=True, img_size=args.img_size)
    val_transforms = SegmentationAugmentations(is_train=False, img_size=args.img_size)

    train_dataset = PlotSegmentationDataset(data_dir, split='train', transforms=train_transforms, img_size=args.img_size)
    val_dataset = PlotSegmentationDataset(data_dir, split='val', transforms=val_transforms, img_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(train_dataset.class_rgb_values)
    print(f"Number of classes: {num_classes}")

    # Create Model (UNet with a pre-trained ResNet34 encoder)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(device)

    # --- Transfer Learning: Freeze the encoder weights ---
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Frozen encoder parameters for transfer learning.")

    # Loss function and optimizer
    # Find the background class index to ignore it during loss calculation.
    # This is crucial for handling class imbalance.
    class_df = pd.read_csv(data_dir / 'class_dict.csv')
    background_class_name = 'background' 
    background_row = class_df[class_df['name'].str.contains(background_class_name, case=False, na=False)]
    
    background_index = -1 # Default to an invalid index
    if not background_row.empty:
        background_index = background_row.index[0]
        print(f"✅ Found '{background_row.iloc[0]['name']}' at index {background_index}. This class will be ignored during training.")
        criterion = nn.CrossEntropyLoss(ignore_index=background_index)
    else:
        print(f"⚠️ Warning: '{background_class_name}' class not found in class_dict.csv. Training without ignoring any class.")
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training and validation loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for images, masks in progress_bar_val:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                progress_bar_val.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model and visualize its predictions
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoints_dir / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best model saved to {checkpoint_path}")

            # Visualize predictions for the new best model from the first validation batch
            print("📸 Generating visualization for new best model...")
            model.eval() # Ensure model is in evaluation mode
            with torch.no_grad():
                # Get the first batch from the validation loader
                try:
                    images, masks = next(iter(val_loader))
                    images, masks = images.to(device), masks.to(device)
                    
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    
                    # Visualize a few images from the batch to avoid clutter
                    for j in range(min(4, images.size(0))): # Visualize up to 4 images
                        visualize(
                            visuals_dir,
                            images[j],
                            masks[j],
                            preds[j],
                            val_dataset.class_rgb_values
                        )
                except StopIteration:
                    print("Could not generate visualization: validation loader is empty.")

    print(f"Training complete. All outputs saved in {output_dir}")

if __name__ == "__main__":
    main()
