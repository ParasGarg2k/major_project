import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# === CONFIGURATION ===
GT_ROOT_DIR = r"D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo\data\new_4_tasks_dataset_our_approach\ground_truth"
PRED_ROOT_DIR = r"D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo\results\new_4_tasks_dataset_our_approach\split_test"
OUTPUT_DIR = r"D:\Artificial Intelligence\IIT Mandi Internship\Rijul's Internship Work\Identification of Sub-Tasks\MS-TCN Approach\testing results\Comparison Images for MS-TCN Paper"

GT_FILE = r"2076.txt"
PRED_FILE = r"2076"
OUTPUT_IMAGE = r"custom_2_our_approach.png"

MODEL_NAME = "Proposed"

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

def encode_labels(label_seq):
    label_to_id = {}
    id_seq = []
    label_idx = 0
    for label in label_seq:
        if label not in label_to_id:
            label_to_id[label] = label_idx
            label_idx += 1
        id_seq.append(label_to_id[label])
    return id_seq, label_to_id

def make_color_map(n):
    cmap = plt.get_cmap('gist_rainbow', n)  # or 'hsv' or 'nipy_spectral'
    return [cmap(i) for i in range(n)]

def plot_label_bars(gt_ids, pred_ids, label_to_id, output_file):
    n_frames = len(gt_ids)
    unique_labels = len(label_to_id)
    colors = make_color_map(unique_labels)

    fig, ax = plt.subplots(2, 1, figsize=(12, 2), sharex=True)

    for ax_idx, (data, title) in enumerate(zip([gt_ids, pred_ids], ['GT', MODEL_NAME])):
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
    pred_labels = load_pred_labels(os.path.join(PRED_ROOT_DIR, PRED_FILE))

    assert len(gt_labels) == len(pred_labels), "Mismatch in number of frames between GT and prediction."

    combined_labels = list(set(gt_labels + pred_labels))
    label_to_id = {label: idx for idx, label in enumerate(sorted(combined_labels))}

    gt_ids = [label_to_id[lbl] for lbl in gt_labels]
    pred_ids = [label_to_id[lbl] for lbl in pred_labels]

    plot_label_bars(gt_ids, pred_ids, label_to_id, os.path.join(OUTPUT_DIR, OUTPUT_IMAGE))