#!/usr/bin/env python3
"""
Visualize training results (GT vs predicted images) from render_test and reconstruction directories.
Provides interactive slider to browse through images.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2
import glob
import sys

def load_image_paths(directory):
    """Load all GT and predicted PNG images from directory."""
    gt_files = sorted(glob.glob(os.path.join(directory, "*_gt.png")))
    pred_files = sorted(glob.glob(os.path.join(directory, "*_pred.png")))
    # Ensure matching indices
    gt_indices = {os.path.basename(f).split('_')[0] for f in gt_files}
    pred_indices = {os.path.basename(f).split('_')[0] for f in pred_files}
    common_indices = sorted(gt_indices.intersection(pred_indices))
    # Build paired list
    pairs = []
    for idx in common_indices:
        gt = os.path.join(directory, f"{idx}_gt.png")
        pred = os.path.join(directory, f"{idx}_pred.png")
        if os.path.exists(gt) and os.path.exists(pred):
            pairs.append((gt, pred))
    return pairs

def load_images(gt_path, pred_path):
    """Load GT and predicted images as numpy arrays."""
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    return gt_img, pred_img

def visualize_directory(directory, title_suffix="", show_diff=True):
    """Create interactive visualization for a single directory."""
    pairs = load_image_paths(directory)
    if not pairs:
        print(f"No paired GT/pred images found in {directory}")
        return

    print(f"Found {len(pairs)} image pairs.")
    # Load first pair
    gt_img, pred_img = load_images(pairs[0][0], pairs[0][1])
    diff_img = np.abs(gt_img.astype(float) - pred_img.astype(float)).astype(np.uint8)

    # Create figure
    fig, axes = plt.subplots(1, 3 if show_diff else 2, figsize=(12, 4))
    if show_diff:
        ax_gt, ax_pred, ax_diff = axes
    else:
        ax_gt, ax_pred = axes
        ax_diff = None
    fig.subplots_adjust(bottom=0.25)

    # Display initial images
    im_gt = ax_gt.imshow(gt_img, cmap='gray', vmin=0, vmax=255)
    ax_gt.set_title(f'GT {title_suffix}')
    ax_gt.axis('off')

    im_pred = ax_pred.imshow(pred_img, cmap='gray', vmin=0, vmax=255)
    ax_pred.set_title(f'Pred {title_suffix}')
    ax_pred.axis('off')

    if show_diff:
        im_diff = ax_diff.imshow(diff_img, cmap='hot', vmin=0, vmax=255)
        ax_diff.set_title('Difference')
        ax_diff.axis('off')

    # Add slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Index', 0, len(pairs)-1, valinit=0, valstep=1)

    # Update function
    def update(val):
        idx = int(slider.val)
        gt_path, pred_path = pairs[idx]
        gt_img, pred_img = load_images(gt_path, pred_path)
        im_gt.set_data(gt_img)
        im_pred.set_data(pred_img)
        if show_diff:
            diff_img = np.abs(gt_img.astype(float) - pred_img.astype(float)).astype(np.uint8)
            im_diff.set_data(diff_img)
        fig.suptitle(f'Pair {idx}: {os.path.basename(gt_path)}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Add buttons for previous/next
    ax_prev = plt.axes([0.2, 0.05, 0.1, 0.04])
    ax_next = plt.axes([0.7, 0.05, 0.1, 0.04])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')

    def prev_event(event):
        if slider.val > 0:
            slider.set_val(slider.val - 1)

    def next_event(event):
        if slider.val < len(pairs)-1:
            slider.set_val(slider.val + 1)

    btn_prev.on_clicked(prev_event)
    btn_next.on_clicked(next_event)

    fig.suptitle(f'Pair 0: {os.path.basename(pairs[0][0])}')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize training results.')
    parser.add_argument('--render_test', type=str,
                        default='output/synthetic_dataset/cone_ntrain_25_angle_360/0_foot_cone/test/iter_30000/render_test',
                        help='Path to render_test directory')
    parser.add_argument('--reconstruction', type=str,
                        default='output/synthetic_dataset/cone_ntrain_25_angle_360/0_foot_cone/test/iter_30000/reconstruction',
                        help='Path to reconstruction directory')
    parser.add_argument('--no_diff', action='store_true', help='Do not show difference image')
    parser.add_argument('--which', choices=['both', 'render', 'recon'], default='both',
                        help='Which directory to visualize')
    args = parser.parse_args()

    show_diff = not args.no_diff

    if args.which in ('both', 'render'):
        if os.path.isdir(args.render_test):
            visualize_directory(args.render_test, title_suffix='(Render Test)', show_diff=show_diff)
        else:
            print(f"Render test directory not found: {args.render_test}")

    if args.which in ('both', 'recon'):
        if os.path.isdir(args.reconstruction):
            visualize_directory(args.reconstruction, title_suffix='(Reconstruction)', show_diff=show_diff)
        else:
            print(f"Reconstruction directory not found: {args.reconstruction}")

if __name__ == '__main__':
    main()