"""
SCRD4AD Inference on BraTS using H5 files.
- FSL FAST segmented images from: brats_t1_fast.h5 or brats_t2_fast.h5
- Tumor ground truth from: brats_tumor_gt.h5
"""

import os
import argparse
import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import model components
from models.rd4ad_mlp import RdadAtten
from torchvision import transforms
from utils import cal_anomaly_map_param


def dice_score(pred, gt, threshold=0.5):
    """Calculate Dice score."""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0
    
    return 2.0 * intersection / union


def iou_score(pred, gt, threshold=0.5):
    """Calculate IoU score."""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0
    
    return intersection / union


def save_visualization(img, anomaly_map, gt_mask, save_path, subject_name, slice_idx):
    """Save visualization of image, anomaly map, and ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'{subject_name} - Slice {slice_idx}')
    axes[0].axis('off')
    
    # Ground truth mask overlay
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(gt_mask, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth Tumor')
    axes[1].axis('off')
    
    # Anomaly map overlay
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(anomaly_map, cmap='hot', alpha=0.5)
    axes[2].set_title('Anomaly Map')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def normalize_to_uint8(slice_2d):
    """Normalize slice to 0-255 uint8."""
    slice_min = slice_2d.min()
    slice_max = slice_2d.max()
    if slice_max > slice_min:
        normalized = (slice_2d - slice_min) / (slice_max - slice_min) * 255
    else:
        normalized = np.zeros_like(slice_2d)
    return normalized.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--h5_dir', type=str, required=True,
                        help='Directory containing H5 files')
    parser.add_argument('--modality', type=str, choices=['t1', 't2'], required=True,
                        help='MRI modality to evaluate')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='brats_inference',
                        help='Experiment name for output subdirectory')
    parser.add_argument('--slice_range', type=float, nargs=2, default=[0.2, 0.8],
                        help='Slice range to use (fraction of total slices)')
    parser.add_argument('--max_vis', type=int, default=50, help='Max visualizations to save')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Model: {args.checkpoint}")
    print(f"Modality: {args.modality}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_name, f'inference_{args.modality}')
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict):
        model = RdadAtten()
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Transform (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # H5 file paths
    fast_h5_path = os.path.join(args.h5_dir, f'brats_{args.modality}_fast.h5')
    gt_h5_path = os.path.join(args.h5_dir, 'brats_tumor_gt.h5')
    
    print(f"Loading FSL FAST images from: {fast_h5_path}")
    print(f"Loading tumor ground truth from: {gt_h5_path}")
    
    # Open H5 files
    fast_h5 = h5py.File(fast_h5_path, 'r')
    gt_h5 = h5py.File(gt_h5_path, 'r')
    
    subject_keys = sorted(list(fast_h5.keys()))
    print(f"Found {len(subject_keys)} subjects")
    
    # Metrics storage
    all_preds = []
    all_gts = []
    all_image_scores = []
    all_image_labels = []
    dice_scores_list = []
    iou_scores_list = []
    
    # Per-subject metrics
    subject_preds = defaultdict(list)
    subject_gts = defaultdict(list)
    subject_dice_scores = defaultdict(list)
    
    slice_range = args.slice_range
    vis_count = 0
    
    for subj_idx, subject_key in enumerate(subject_keys):
        # Load volumes
        fast_volume = fast_h5[subject_key][:]
        gt_volume = gt_h5[subject_key][:]
        
        # Handle 4D volumes
        if fast_volume.ndim == 4:
            fast_volume = fast_volume[:, :, :, 0]
        if gt_volume.ndim == 4:
            gt_volume = gt_volume[:, :, :, 0]
        
        num_slices = fast_volume.shape[2]
        start_idx = int(num_slices * slice_range[0])
        end_idx = int(num_slices * slice_range[1])
        
        for slice_idx in range(start_idx, end_idx):
            img_slice = fast_volume[:, :, slice_idx]
            gt_slice = gt_volume[:, :, slice_idx]
            
            # Skip empty slices
            if np.count_nonzero(img_slice) < 500:
                continue
            
            # Preprocess image (FSL FAST has values 0,1,2,3)
            # Normalize to 0-255 range
            slice_uint8 = normalize_to_uint8(img_slice)
            slice_rgb = np.stack([slice_uint8, slice_uint8, slice_uint8], axis=-1)
            img_for_model = cv2.resize(slice_rgb / 255., (256, 256))
            
            # Transform and run inference
            img_tensor = transform(img_for_model).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                inputs, outputs, param = model(img_tensor)
                anomaly_map, _ = cal_anomaly_map_param(inputs, outputs, 256, amap_mode='a', param=param)
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            # Resize ground truth to match anomaly map
            gt_resized = cv2.resize(gt_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # Normalize anomaly map to [0, 1]
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            else:
                anomaly_map_norm = anomaly_map
            
            # Store for pixel-level metrics (downsample to reduce memory)
            pred_downsampled = anomaly_map_norm[::4, ::4].flatten()
            gt_downsampled = (gt_resized[::4, ::4] > 0).astype(np.float32).flatten()
            all_preds.append(pred_downsampled)
            all_gts.append(gt_downsampled)
            
            # Store for per-subject metrics
            subject_preds[subject_key].append(anomaly_map_norm.flatten())
            subject_gts[subject_key].append((gt_resized > 0).astype(np.float32).flatten())
            
            # Image-level score
            img_score = np.max(anomaly_map)
            has_tumor = np.any(gt_resized > 0)
            all_image_scores.append(img_score)
            all_image_labels.append(1 if has_tumor else 0)
            
            # Compute per-slice metrics
            slice_dice = dice_score(anomaly_map_norm, gt_resized, threshold=0.5)
            slice_iou = iou_score(anomaly_map_norm, gt_resized, threshold=0.5)
            
            if has_tumor:  # Only count slices with actual tumor
                dice_scores_list.append(slice_dice)
                iou_scores_list.append(slice_iou)
                subject_dice_scores[subject_key].append(slice_dice)
            
            # Save visualization
            if vis_count < args.max_vis and has_tumor:
                vis_path = os.path.join(vis_dir, f'{subject_key}_slice{slice_idx:03d}.png')
                img_for_vis = cv2.resize(slice_uint8, (256, 256))
                save_visualization(img_for_vis, anomaly_map_norm, gt_resized, vis_path, subject_key, slice_idx)
                vis_count += 1
        
        if (subj_idx + 1) % 10 == 0:
            print(f"  Processed {subj_idx + 1}/{len(subject_keys)} subjects")
    
    # Close H5 files
    fast_h5.close()
    gt_h5.close()
    
    # Compute per-subject Dice (volume-level)
    per_subject_dice = {}
    for subject_key in subject_preds.keys():
        if len(subject_preds[subject_key]) > 0:
            # Concatenate all slices for this subject
            subj_pred = np.concatenate(subject_preds[subject_key])
            subj_gt = np.concatenate(subject_gts[subject_key])
            # Compute volume-level Dice
            volume_dice = dice_score(subj_pred, subj_gt, threshold=0.5)
            per_subject_dice[subject_key] = volume_dice
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("PER-SUBJECT METRICS")
    print("=" * 60)
    dice_values = list(per_subject_dice.values())
    print(f"Average Dice (per-subject/volume): {np.mean(dice_values):.4f} ± {np.std(dice_values):.4f}")
    print(f"Number of subjects: {len(dice_values)}")
    
    # Save per-subject Dice to CSV
    csv_path = os.path.join(output_dir, 'per_subject_dice.csv')
    with open(csv_path, 'w') as f:
        f.write("subject,dice\n")
        for subj, dice_val in sorted(per_subject_dice.items()):
            f.write(f"{subj},{dice_val:.6f}\n")
    
    print("\n" + "=" * 60)
    print("PIXEL-LEVEL METRICS")
    print("=" * 60)
    
    # Pixel-level AUROC
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    
    if len(np.unique(all_gts)) > 1:
        pixel_auroc = roc_auc_score(all_gts, all_preds)
        print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
    else:
        pixel_auroc = float('nan')
        print("Pixel-level AUROC: N/A (single class)")
    
    # Best Dice at optimal threshold
    best_dice = 0
    best_threshold = 0
    for thresh in np.arange(0.1, 0.95, 0.05):
        d = dice_score(all_preds, all_gts, threshold=thresh)
        if d > best_dice:
            best_dice = d
            best_threshold = thresh
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best Dice (at optimal threshold): {best_dice:.4f}")
    
    # Average metrics
    if dice_scores_list:
        print(f"Average Dice (per-slice, tumor only): {np.mean(dice_scores_list):.4f} ± {np.std(dice_scores_list):.4f}")
    if iou_scores_list:
        print(f"Average IoU (per-slice, tumor only): {np.mean(iou_scores_list):.4f} ± {np.std(iou_scores_list):.4f}")
    
    # Image-level AUROC
    print("\n" + "=" * 60)
    print("IMAGE-LEVEL METRICS")
    print("=" * 60)
    
    if len(np.unique(all_image_labels)) > 1:
        image_auroc = roc_auc_score(all_image_labels, all_image_scores)
        print(f"Image-level AUROC: {image_auroc:.4f}")
    else:
        image_auroc = float('nan')
        print("Image-level AUROC: N/A (single class)")
    
    # Save results
    results_path = os.path.join(output_dir, 'metrics.txt')
    with open(results_path, 'w') as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Modality: {args.modality}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"H5 Directory: {args.h5_dir}\n\n")
        f.write("=" * 60 + "\n")
        f.write("PER-SUBJECT METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Average Dice (per-subject/volume): {np.mean(dice_values):.4f} ± {np.std(dice_values):.4f}\n")
        f.write(f"Number of subjects: {len(dice_values)}\n\n")
        f.write("=" * 60 + "\n")
        f.write("PIXEL-LEVEL METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Pixel-level AUROC: {pixel_auroc:.4f}\n")
        f.write(f"Best threshold: {best_threshold:.2f}\n")
        f.write(f"Best Dice: {best_dice:.4f}\n")
        if dice_scores_list:
            f.write(f"Average Dice (per-slice, tumor only): {np.mean(dice_scores_list):.4f} ± {np.std(dice_scores_list):.4f}\n")
        if iou_scores_list:
            f.write(f"Average IoU (per-slice, tumor only): {np.mean(iou_scores_list):.4f} ± {np.std(iou_scores_list):.4f}\n")
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("IMAGE-LEVEL METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Image-level AUROC: {image_auroc:.4f}\n")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Per-subject Dice saved to: {csv_path}")
    print(f"Visualizations saved to: {vis_dir}")


if __name__ == '__main__':
    main()
