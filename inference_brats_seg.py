#!/usr/bin/env python
"""
Inference on FSL FAST segmented BraTS images.
Uses pre-segmented BraTS images from BraTS_T1_seg and BraTS_T2_seg directories.
"""

import os
import gc
import glob
import argparse
import numpy as np
import nibabel as nib
import torch
import cv2
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from models.rd4ad_mlp import RdadAtten


def cal_anomaly_map_param(fs_list, ft_list, out_size=224, amap_mode='mul', param=None):
    """Calculate anomaly map with parameters."""
    weight = param[0]
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i] * weight[i]
        ft = ft_list[i] * weight[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
            
    return anomaly_map, a_map_list


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


def run_inference(args):
    """Run inference on FSL FAST segmented BraTS images."""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Experiment: {args.experiment_name}", flush=True)
    print(f"Model: {args.model_path}", flush=True)
    print(f"BraTS modality: {args.modality}", flush=True)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict):
        model = RdadAtten()
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    model = model.to(device)
    model.eval()
    print("Model loaded successfully", flush=True)
    
    # Paths for FSL FAST segmented images and raw GT
    if args.modality == 't1':
        seg_brats_path = os.path.join(args.brats_path, 'BraTS_T1_seg')
    else:
        seg_brats_path = os.path.join(args.brats_path, 'BraTS_T2_seg')
    
    raw_brats_path = os.path.join(args.brats_path, 'BraTS_raw')
    
    print(f"Using FSL FAST segmented images from: {seg_brats_path}", flush=True)
    print(f"Using GT masks from: {raw_brats_path}", flush=True)
    
    # Create output directories
    output_dir = os.path.join(args.output_dir, f'inference_{args.modality}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Transform (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize((256, 256))
    ])
    
    # Find all segmented BraTS files
    seg_files = sorted(glob.glob(os.path.join(seg_brats_path, "*.nii.gz")))
    print(f"Found {len(seg_files)} FSL FAST segmented files", flush=True)
    
    # Storage for metrics
    all_preds = []
    all_gts = []
    all_image_scores = []
    all_image_labels = []
    dice_scores_list = []
    iou_scores_list = []
    
    # Per-subject storage
    subject_dice_scores = {}
    subject_preds = {}
    subject_gts = {}
    
    slice_range = (0.2, 0.8)  # Middle 60% of slices
    
    for file_idx, seg_file in enumerate(seg_files):
        # Extract subject name from filename (e.g., BraTS20_Training_001_t1_seg.nii.gz)
        filename = os.path.basename(seg_file)
        # Parse subject name - handle various naming conventions
        parts = filename.replace('.nii.gz', '').split('_')
        if len(parts) >= 3:
            subject_name = '_'.join(parts[:3])  # BraTS20_Training_001
        else:
            subject_name = filename.replace('.nii.gz', '')
        
        # Find corresponding GT mask in raw BraTS
        subject_dir = os.path.join(raw_brats_path, subject_name)
        if not os.path.isdir(subject_dir):
            # Try without the modality suffix
            potential_dirs = glob.glob(os.path.join(raw_brats_path, f"{subject_name}*"))
            if potential_dirs:
                subject_dir = potential_dirs[0]
                subject_name = os.path.basename(subject_dir)
            else:
                continue
        
        seg_mask_files = glob.glob(os.path.join(subject_dir, "*_seg.nii.gz"))
        if not seg_mask_files:
            continue
        
        # Initialize per-subject storage
        subject_dice_scores[subject_name] = []
        subject_preds[subject_name] = []
        subject_gts[subject_name] = []
        
        # Load FSL FAST segmented image
        seg_img_nii = nib.load(seg_file)
        seg_img_volume = seg_img_nii.get_fdata()
        
        # Load GT tumor mask
        gt_nii = nib.load(seg_mask_files[0])
        gt_volume = gt_nii.get_fdata()
        
        # Handle 4D volumes
        if seg_img_volume.ndim == 4:
            seg_img_volume = seg_img_volume[:, :, :, 0]
        if gt_volume.ndim == 4:
            gt_volume = gt_volume[:, :, :, 0]
        
        num_slices = seg_img_volume.shape[2]
        start_idx = int(num_slices * slice_range[0])
        end_idx = int(num_slices * slice_range[1])
        
        for slice_idx in range(start_idx, end_idx):
            img_slice = seg_img_volume[:, :, slice_idx]
            gt_slice = gt_volume[:, :, slice_idx]
            
            # Skip empty slices
            if np.count_nonzero(img_slice) < 500:
                continue
            
            # Preprocess image
            slice_uint8 = normalize_to_uint8(img_slice)
            slice_rgb = np.stack([slice_uint8, slice_uint8, slice_uint8], axis=-1)
            img_for_model = cv2.resize(slice_rgb / 255., (256, 256))
            
            # Transform and run inference
            img_tensor = transform(img_for_model).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                inputs, outputs, param = model(img_tensor)
                anomaly_map, _ = cal_anomaly_map_param(inputs, outputs, 256, amap_mode='a', param=param)
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            # Resize GT to match anomaly map
            gt_resized = cv2.resize(gt_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # Normalize anomaly map to [0, 1]
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            else:
                anomaly_map_norm = anomaly_map
            
            # Store for pixel-level metrics
            pred_downsampled = anomaly_map_norm[::4, ::4].flatten()
            gt_downsampled = (gt_resized[::4, ::4] > 0).astype(np.float32).flatten()
            all_preds.append(pred_downsampled)
            all_gts.append(gt_downsampled)
            
            # Store for per-subject metrics
            subject_preds[subject_name].append(anomaly_map_norm.flatten())
            subject_gts[subject_name].append((gt_resized > 0).astype(np.float32).flatten())
            
            # Image-level score
            img_score = np.max(anomaly_map)
            has_tumor = np.any(gt_resized > 0)
            all_image_scores.append(img_score)
            all_image_labels.append(1 if has_tumor else 0)
            
            # Compute per-slice metrics
            slice_dice = dice_score(anomaly_map_norm, gt_resized, threshold=0.5)
            slice_iou = iou_score(anomaly_map_norm, gt_resized, threshold=0.5)
            
            if has_tumor:
                dice_scores_list.append(slice_dice)
                iou_scores_list.append(slice_iou)
                subject_dice_scores[subject_name].append(slice_dice)
        
        if (file_idx + 1) % 50 == 0:
            print(f"  Processed {file_idx + 1}/{len(seg_files)} files", flush=True)
            gc.collect()
    
    # Compute Per-Subject Dice
    print("\n" + "=" * 60)
    print("PER-SUBJECT METRICS")
    print("=" * 60)
    
    subject_dice_results = {}
    for subject_name in subject_preds.keys():
        if subject_preds[subject_name] and subject_gts[subject_name]:
            subj_pred = np.concatenate(subject_preds[subject_name])
            subj_gt = np.concatenate(subject_gts[subject_name])
            
            subj_dice = dice_score(subj_pred, subj_gt, threshold=0.5)
            subj_iou = iou_score(subj_pred, subj_gt, threshold=0.5)
            
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
    
    volume_dice_list = [v['volume_dice'] for v in subject_dice_results.values()]
    avg_subject_dice = np.mean(volume_dice_list) if volume_dice_list else 0
    std_subject_dice = np.std(volume_dice_list) if volume_dice_list else 0
    
    print(f"Average Dice (per-subject/volume): {avg_subject_dice:.4f} ± {std_subject_dice:.4f}")
    print(f"Number of subjects: {len(subject_dice_results)}")
    
    # Compute Pixel-Level Metrics
    print("\n" + "=" * 60)
    print("PIXEL-LEVEL METRICS")
    print("=" * 60)
    
    all_preds_flat = np.concatenate(all_preds)
    all_gts_flat = np.concatenate(all_gts)
    
    pixel_auroc = roc_auc_score(all_gts_flat, all_preds_flat)
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
    
    best_thresh, best_dice = find_best_threshold(all_preds_flat, all_gts_flat)
    print(f"Best threshold: {best_thresh:.2f}")
    print(f"Best Dice (at optimal threshold): {best_dice:.4f}")
    
    avg_dice = np.mean(dice_scores_list) if dice_scores_list else 0
    std_dice = np.std(dice_scores_list) if dice_scores_list else 0
    avg_iou = np.mean(iou_scores_list) if iou_scores_list else 0
    std_iou = np.std(iou_scores_list) if iou_scores_list else 0
    
    if dice_scores_list:
        print(f"Average Dice (per-slice, tumor only): {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"Average IoU (per-slice, tumor only): {avg_iou:.4f} ± {std_iou:.4f}")
    
    print("\n" + "=" * 60)
    print("IMAGE-LEVEL METRICS")
    print("=" * 60)
    
    try:
        image_auroc = roc_auc_score(all_image_labels, all_image_scores)
        print(f"Image-level AUROC: {image_auroc:.4f}")
    except:
        print("Image-level AUROC: nan (all same class)")
    
    # Save results
    results_file = os.path.join(output_dir, 'metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f"SCRD4AD Inference on FSL FAST Segmented BraTS ({args.modality.upper()})\n")
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
        if dice_scores_list:
            f.write(f"  Average Dice (per-slice): {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"  Average IoU (per-slice): {avg_iou:.4f} ± {std_iou:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCRD4AD Inference on FSL FAST Segmented BraTS')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--modality', type=str, required=True, choices=['t1', 't2'], help='BraTS modality')
    parser.add_argument('--brats_path', type=str, required=True, help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    run_inference(args)
