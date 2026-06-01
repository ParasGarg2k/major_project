import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# === CONFIGURATION ===
GT_ROOT_DIR = r"D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo\data\new_4_tasks_dataset\ground_truth"
PRED1_ROOT_DIR = r"D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo\results\new_4_tasks_dataset\split_test"
PRED2_ROOT_DIR = r"D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo\results\other_model_predictions"
OUTPUT_DIR = r"D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo"

GT_FILE = "2076.txt"
PRED1_FILE = "2076"
PRED2_FILE = "2076"
OUTPUT_IMAGE = "comparison_two_models.png"

# === UTILITY ===
def load_gt_labels(path):
    with open(path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def load_pred_labels(path):
    with open(path, 'r') as f:
        lines = f.readlines()[1:]  # skip header line
        labels = [line.strip() for line in lines if line.strip()]
    return labels

def make_color_map(n):
    cmap = plt.get_cmap('tab20', n)
    return [cmap(i) for i in range(n)]

def plot_label_bars(gt_ids, pred1_ids, pred2_ids, label_to_id, output_file):
    unique_labels = len(label_to_id)
    colors = make_color_map(unique_labels)

    fig, ax = plt.subplots(3, 1, figsize=(12, 3), sharex=True)
    label_sets = [(gt_ids, 'GT'), (pred1_ids, 'Model 1'), (pred2_ids, 'Model 2')]

    for ax_idx, (data, title) in enumerate(label_sets):
        color_bar = [colors[i] for i in data]
        ax[ax_idx].imshow([color_bar], aspect='auto')
        ax[ax_idx].set_yticks([])
        ax[ax_idx].set_ylabel(title, rotation=0, labelpad=30, fontsize=12)
        ax[ax_idx].set_xticks([])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved visualization to {output_file}")

# === MAIN ===
if __name__ == "__main__":
    gt_labels = load_gt_labels(os.path.join(GT_ROOT_DIR, GT_FILE))
    pred1_labels = load_pred_labels(os.path.join(PRED1_ROOT_DIR, PRED1_FILE))
    pred2_labels = load_pred_labels(os.path.join(PRED2_ROOT_DIR, PRED2_FILE))

    assert len(gt_labels) == len(pred1_labels) == len(pred2_labels), "Mismatch in number of frames."

    combined_labels = list(set(gt_labels + pred1_labels + pred2_labels))
    label_to_id = {label: idx for idx, label in enumerate(sorted(combined_labels))}

    gt_ids = [label_to_id[lbl] for lbl in gt_labels]
    pred1_ids = [label_to_id[lbl] for lbl in pred1_labels]
    pred2_ids = [label_to_id[lbl] for lbl in pred2_labels]

    plot_label_bars(gt_ids, pred1_ids, pred2_ids, label_to_id, os.path.join(OUTPUT_DIR, OUTPUT_IMAGE))
