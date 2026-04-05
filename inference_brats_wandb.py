#!/usr/bin/env python
"""
Pixel-level evaluation of SCRD4AD on BraTS dataset with W&B logging.
Adapted from inference_brats.py for multiple experiments.
"""

import os
import gc
import glob
import argparse
import numpy as np
import nibabel as nib
import torch
import cv2
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from models.rd4ad_mlp import RdadAtten
from utils import cal_anomaly_map_param

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def normalize_to_uint8(img_slice):
    """Normalize slice to 0-255 range."""
    img_slice = img_slice.astype(np.float32)
    if img_slice.max() == img_slice.min():
        return np.zeros_like(img_slice, dtype=np.uint8)
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
    return (img_slice * 255).astype(np.uint8)


def dice_score(pred, gt, threshold=0.5):
    """Compute Dice score between prediction and ground truth."""
    pred_binary = (pred >= threshold).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0
    
    return (2.0 * intersection) / union


def iou_score(pred, gt, threshold=0.5):
    """Compute IoU score between prediction and ground truth."""
    pred_binary = (pred >= threshold).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0
    
    return intersection / union


def find_best_threshold(preds_flat, gts_flat):
    """Find threshold that maximizes Dice score."""
    best_thresh = 0.5
    best_dice = 0
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred_binary = (preds_flat >= thresh).astype(np.float32)
        gt_binary = (gts_flat > 0).astype(np.float32)
        
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary)
        
        if union > 0:
            dice = (2.0 * intersection) / union
            if dice > best_dice:
                best_dice = dice
                best_thresh = thresh
    
    return best_thresh, best_dice


def save_visualization(img, anomaly_map, gt_mask, save_path, subject_name, slice_idx):
    """Save visualization with original image, anomaly map, and ground truth."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(gt_mask, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Anomaly map
    axes[2].imshow(anomaly_map, cmap='hot')
    axes[2].set_title('Anomaly Map')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(img, cmap='gray')
    axes[3].imshow(anomaly_map, cmap='hot', alpha=0.5)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.suptitle(f'{subject_name} - Slice {slice_idx}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def run_inference(args):
    """Run inference on BraTS dataset."""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Experiment: {args.experiment_name}", flush=True)
    print(f"Model: {args.model_path}", flush=True)
    print(f"BraTS modality: {args.modality}", flush=True)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict):
        # New format: state_dict saved
        model = RdadAtten()
        model.load_state_dict(checkpoint)
    else:
        # Old format: complete model saved
        model = checkpoint
    model = model.to(device)
    model.eval()
    print("Model loaded successfully", flush=True)
    
    # Initialize W&B
    if WANDB_AVAILABLE and args.wandb_mode != 'disabled':
        wandb.init(
            project='thesis-uad',
            entity=args.wandb_entity,
            name=f'{args.experiment_name}_infer_{args.modality}',
            group='SCRD4AD_inference',
            tags=['SCRD4AD', 'inference', 'BraTS', args.modality, args.experiment_name],
            mode=args.wandb_mode,
            config={
                'experiment_name': args.experiment_name,
                'model_path': args.model_path,
                'modality': args.modality,
                'brats_path': args.brats_path,
            }
        )
    
    # Create output directories
    output_dir = os.path.join(args.output_dir, args.experiment_name, f'inference_{args.modality}')
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Transform (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize((256, 256))
    ])
    
    # Find all BraTS subjects
    # The BraTS subjects are inside BraTS_raw folder with names like "BraTS20_Training_001"
    brats_raw_path = os.path.join(args.brats_path, 'BraTS_raw')
    if os.path.isdir(brats_raw_path):
        brats_subjects = sorted(glob.glob(os.path.join(brats_raw_path, "BraTS*")))
    else:
        # Fallback: try direct path for backward compatibility
        brats_subjects = sorted([d for d in glob.glob(os.path.join(args.brats_path, "BraTS*")) 
                                  if os.path.isdir(d) and "BraTS20" in os.path.basename(d)])
    print(f"Found {len(brats_subjects)} BraTS subjects", flush=True)
    
    # Storage for metrics
    all_preds = []
    all_gts = []
    all_image_scores = []
    all_image_labels = []
    dice_scores = []
    iou_scores_list = []
    vis_count = 0
    
    # Per-subject storage
    subject_dice_scores = {}  # subject_name -> list of slice dice scores
    subject_preds = {}  # subject_name -> list of predictions
    subject_gts = {}  # subject_name -> list of ground truths
    
    slice_range = (0.2, 0.8)  # Middle 60% of slices
    
    for subj_idx, subject_dir in enumerate(brats_subjects):
        subject_name = os.path.basename(subject_dir)
        
        # Initialize per-subject storage
        subject_dice_scores[subject_name] = []
        subject_preds[subject_name] = []
        subject_gts[subject_name] = []
        
        # Find modality and segmentation files
        modality_files = glob.glob(os.path.join(subject_dir, f"*_{args.modality}.nii.gz"))
        seg_files = glob.glob(os.path.join(subject_dir, "*_seg.nii.gz"))
        
        if not modality_files or not seg_files:
            print(f"  [WARN] Missing {args.modality} or seg file in {subject_dir}")
            continue
        
        # Load volumes
        img_nii = nib.load(modality_files[0])
        seg_nii = nib.load(seg_files[0])
        img_volume = img_nii.get_fdata()
        seg_volume = seg_nii.get_fdata()
        
        # Handle 4D volumes
        if img_volume.ndim == 4:
            img_volume = img_volume[:, :, :, 0]
        if seg_volume.ndim == 4:
            seg_volume = seg_volume[:, :, :, 0]
        
        num_slices = img_volume.shape[2]
        start_idx = int(num_slices * slice_range[0])
        end_idx = int(num_slices * slice_range[1])
        
        for slice_idx in range(start_idx, end_idx):
            img_slice = img_volume[:, :, slice_idx]
            seg_slice = seg_volume[:, :, slice_idx]
            
            # Skip empty slices
            if np.count_nonzero(img_slice) < 500:
                continue
            
            # Preprocess image (same as training)
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
            seg_resized = cv2.resize(seg_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # Normalize anomaly map to [0, 1]
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            else:
                anomaly_map_norm = anomaly_map
            
            # Store for pixel-level metrics (downsample more aggressively to reduce memory)
            pred_downsampled = anomaly_map_norm[::4, ::4].flatten()  # 4x downsample instead of 2x
            gt_downsampled = (seg_resized[::4, ::4] > 0).astype(np.float32).flatten()
            all_preds.append(pred_downsampled)
            all_gts.append(gt_downsampled)
            
            # Store for per-subject metrics
            subject_preds[subject_name].append(anomaly_map_norm.flatten())
            subject_gts[subject_name].append((seg_resized > 0).astype(np.float32).flatten())
            
            # Image-level score
            img_score = np.max(anomaly_map)
            has_tumor = np.any(seg_resized > 0)
            all_image_scores.append(img_score)
            all_image_labels.append(1 if has_tumor else 0)
            
            # Compute per-slice metrics
            slice_dice = dice_score(anomaly_map_norm, seg_resized, threshold=0.5)
            slice_iou = iou_score(anomaly_map_norm, seg_resized, threshold=0.5)
            
            if has_tumor:  # Only count slices with actual tumor
                dice_scores.append(slice_dice)
                iou_scores_list.append(slice_iou)
                subject_dice_scores[subject_name].append(slice_dice)
            
            # Save visualization
            if vis_count < args.max_vis and has_tumor:
                vis_path = os.path.join(vis_dir, f'{subject_name}_slice{slice_idx:03d}.png')
                img_resized = cv2.resize(slice_uint8, (256, 256))
                save_visualization(img_resized, anomaly_map_norm, seg_resized, vis_path, subject_name, slice_idx)
                vis_count += 1
        
        if (subj_idx + 1) % 10 == 0:
            print(f"  Processed {subj_idx + 1}/{len(brats_subjects)} subjects", flush=True)
            gc.collect()  # Free memory
    
    # ============== Compute Per-Subject Dice ==============
    print("\n" + "=" * 60)
    print("PER-SUBJECT METRICS")
    print("=" * 60)
    
    subject_dice_results = {}
    for subject_name in subject_preds.keys():
        if subject_preds[subject_name] and subject_gts[subject_name]:
            # Concatenate all slices for this subject
            subj_pred = np.concatenate(subject_preds[subject_name])
            subj_gt = np.concatenate(subject_gts[subject_name])
            
            # Compute Dice at optimal threshold (0.5 or use best_thresh later)
            subj_dice = dice_score(subj_pred, subj_gt, threshold=0.5)
            subj_iou = iou_score(subj_pred, subj_gt, threshold=0.5)
            
            # Also compute mean of per-slice dice for this subject
            if subject_dice_scores[subject_name]:
                mean_slice_dice = np.mean(subject_dice_scores[subject_name])
            else:
                mean_slice_dice = 0.0
            
            subject_dice_results[subject_name] = {
                'volume_dice': subj_dice,
                'volume_iou': subj_iou,
                'mean_slice_dice': mean_slice_dice,
                'num_slices': len(subject_dice_scores[subject_name])
            }
    
    # Compute average per-subject Dice
    volume_dice_list = [v['volume_dice'] for v in subject_dice_results.values()]
    avg_subject_dice = np.mean(volume_dice_list) if volume_dice_list else 0
    std_subject_dice = np.std(volume_dice_list) if volume_dice_list else 0
    
    print(f"Average Dice (per-subject/volume): {avg_subject_dice:.4f} ± {std_subject_dice:.4f}")
    print(f"Number of subjects: {len(subject_dice_results)}")
    
    # ============== Compute Final Metrics ==============
    print("\n" + "=" * 60)
    print("PIXEL-LEVEL METRICS")
    print("=" * 60)
    
    # Concatenate all predictions and ground truths
    all_preds_flat = np.concatenate(all_preds)
    all_gts_flat = np.concatenate(all_gts)
    
    # Pixel-level AUROC
    pixel_auroc = roc_auc_score(all_gts_flat, all_preds_flat)
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
    
    # Find best threshold
    best_thresh, best_dice = find_best_threshold(all_preds_flat, all_gts_flat)
    print(f"Best threshold: {best_thresh:.2f}")
    print(f"Best Dice (at optimal threshold): {best_dice:.4f}")
    
    # Average Dice and IoU on tumor slices
    avg_dice = np.mean(dice_scores) if dice_scores else 0
    std_dice = np.std(dice_scores) if dice_scores else 0
    avg_iou = np.mean(iou_scores_list) if iou_scores_list else 0
    std_iou = np.std(iou_scores_list) if iou_scores_list else 0
    
    if dice_scores:
        print(f"Average Dice (per-slice, tumor only): {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"Average IoU (per-slice, tumor only): {avg_iou:.4f} ± {std_iou:.4f}")
    
    print("\n" + "=" * 60)
    print("IMAGE-LEVEL METRICS")
    print("=" * 60)
    
    # Image-level AUROC
    image_auroc = roc_auc_score(all_image_labels, all_image_scores)
    print(f"Image-level AUROC: {image_auroc:.4f}")
    
    # Log to W&B
    if WANDB_AVAILABLE and args.wandb_mode != 'disabled':
        wandb.log({
            'pixel_auroc': pixel_auroc,
            'image_auroc': image_auroc,
            'best_threshold': best_thresh,
            'best_dice': best_dice,
            'avg_dice_slice': avg_dice,
            'std_dice_slice': std_dice,
            'avg_dice_subject': avg_subject_dice,
            'std_dice_subject': std_subject_dice,
            'avg_iou': avg_iou,
            'std_iou': std_iou,
            'total_slices': len(all_preds),
            'total_subjects': len(brats_subjects),
        })
        
        # Log some visualizations
        vis_files = sorted(glob.glob(os.path.join(vis_dir, '*.png')))[:10]
        if vis_files:
            wandb.log({
                'visualizations': [wandb.Image(f) for f in vis_files]
            })
        
        wandb.finish()
    
    # Save results to file
    results_file = os.path.join(output_dir, 'metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f"SCRD4AD Inference Results on BraTS ({args.modality.upper()})\n")
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PER-SUBJECT METRICS\n")
        f.write(f"  Average Dice (per-subject): {avg_subject_dice:.4f} ± {std_subject_dice:.4f}\n")
        f.write(f"  Number of subjects: {len(subject_dice_results)}\n\n")
        
        f.write("PIXEL-LEVEL METRICS\n")
        f.write(f"  Pixel-level AUROC: {pixel_auroc:.4f}\n")
        f.write(f"  Best threshold: {best_thresh:.2f}\n")
        f.write(f"  Best Dice (optimal threshold): {best_dice:.4f}\n")
        if dice_scores:
            f.write(f"  Average Dice (per-slice): {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"  Average IoU (per-slice): {avg_iou:.4f} ± {std_iou:.4f}\n")
        f.write("\nIMAGE-LEVEL METRICS\n")
        f.write(f"  Image-level AUROC: {image_auroc:.4f}\n")
        f.write(f"\nTotal slices evaluated: {len(all_preds)}\n")
        f.write(f"Total subjects: {len(brats_subjects)}\n")
    
    # Save per-subject Dice to CSV
    subject_csv_file = os.path.join(output_dir, 'per_subject_dice.csv')
    with open(subject_csv_file, 'w') as f:
        f.write("subject,volume_dice,volume_iou,mean_slice_dice,num_slices\n")
        for subj_name, metrics in sorted(subject_dice_results.items()):
            f.write(f"{subj_name},{metrics['volume_dice']:.4f},{metrics['volume_iou']:.4f},{metrics['mean_slice_dice']:.4f},{metrics['num_slices']}\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Per-subject Dice saved to: {subject_csv_file}")
    print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCRD4AD BraTS Inference with W&B')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--modality', type=str, required=True, choices=['t1', 't2'], help='BraTS modality to test')
    parser.add_argument('--brats_path', type=str, required=True, help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--max_vis', type=int, default=50, help='Max visualizations to save')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'], help='W&B mode')
    
    args = parser.parse_args()
    run_inference(args)
