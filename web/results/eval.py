#!/usr/bin/env python3
"""
Evaluate predicted ground-truth files against reference files.

Computes:
- frame-wise (cumulative) accuracy
- edit score (based on Levenshtein on collapsed sequences)
- F1@{10,25,50} (segment-level IoU thresholds)

Usage: python eval.py --ref <ref_dir> --pred <pred_dir>

Writes `eval_summary.csv` into the prediction directory and prints per-file + overall metrics.
"""
import os
import argparse
import csv
from glob import glob
from collections import defaultdict
import numpy as np


def read_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f.readlines()]


def segments_from_labels(labels):
    """Return list of (label, start, end) with start/end inclusive (0-based)."""
    segs = []
    if not labels:
        return segs
    cur_label = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            segs.append((cur_label, start, i - 1))
            cur_label = labels[i]
            start = i
    segs.append((cur_label, start, len(labels) - 1))
    return segs


def collapse_sequence(labels):
    out = []
    prev = None
    for l in labels:
        if l != prev:
            out.append(l)
            prev = l
    return out


def levenshtein(a, b):
    # classic DP
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def get_labels_start_end_time(frame_wise_labels, bg_class=None):
    if bg_class is None:
        bg_class = ['background']
    labels = []
    starts = []
    ends = []
    if len(frame_wise_labels) == 0:
        return labels, starts, ends
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(len(frame_wise_labels) - 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(gt_labels, pred_labels, ignore_background=False, background_label='background'):
    bg = [background_label]
    P, _, _ = get_labels_start_end_time(pred_labels, bg)
    Y, _, _ = get_labels_start_end_time(gt_labels, bg)
    # normalized percent
    return levenstein(P, Y, norm=True)


def iou(seg1, seg2):
    # seg: (label, start, end)
    s1, e1 = seg1[1], seg1[2]
    s2, e2 = seg2[1], seg2[2]
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    if inter == 0:
        return 0.0
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union


def f1_at_threshold(gt_labels, pred_labels, thresh, ignore_background=False, background_label='background'):
    bg = [background_label]
    p_label, p_start, p_end = get_labels_start_end_time(pred_labels, bg)
    y_label, y_start, y_end = get_labels_start_end_time(gt_labels, bg)

    tp = 0.0
    fp = 0.0

    hits = np.zeros(len(y_label), dtype=np.int32)

    for j in range(len(p_label)):
        # compute IoU per ground-truth segment
        intersection = np.minimum(p_end[j], np.array(y_end)) - np.maximum(p_start[j], np.array(y_start))
        union = np.maximum(p_end[j], np.array(y_end)) - np.minimum(p_start[j], np.array(y_start))
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            IoU = (1.0 * intersection / union) * (np.array([p_label[j] == y for y in y_label]))
            IoU = np.nan_to_num(IoU)
        idx = int(np.array(IoU).argmax()) if len(IoU) > 0 else -1

        if idx >= 0 and IoU[idx] >= thresh and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = float(len(y_label) - np.sum(hits))

    TP = tp
    FP = fp
    FN = fn
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return TP, FP, FN, precision, recall, f1


def evaluate_directories(ref_dir, pred_dir, ignore_background=False):
    ref_files = sorted([os.path.basename(p) for p in glob(os.path.join(ref_dir, '*.txt'))])
    pred_files = sorted([os.path.basename(p) for p in glob(os.path.join(pred_dir, '*.txt'))])
    common = [f for f in ref_files if f in pred_files]
    if not common:
        raise RuntimeError('No common .txt files found between directories')

    overall = {
        'frames_total': 0,
        'matches': 0,
        'edit_scores': [],
        'f1': {0.1: {'TP':0,'FP':0,'FN':0}, 0.25: {'TP':0,'FP':0,'FN':0}, 0.5: {'TP':0,'FP':0,'FN':0}}
    }

    rows = []
    for name in common:
        gt = read_labels(os.path.join(ref_dir, name))
        pr = read_labels(os.path.join(pred_dir, name))
        n = min(len(gt), len(pr))
        matches = sum(1 for a, b in zip(gt[:n], pr[:n]) if a == b)
        overall['frames_total'] += n
        overall['matches'] += matches
        acc = matches / n if n > 0 else 0.0
        ed = edit_score(gt, pr, ignore_background=ignore_background)
        f1vals = {}
        for t in (0.1, 0.25, 0.5):
            TP, FP, FN, prec, rec, f1 = f1_at_threshold(gt, pr, t, ignore_background=ignore_background)
            overall['f1'][t]['TP'] += TP
            overall['f1'][t]['FP'] += FP
            overall['f1'][t]['FN'] += FN
            f1vals[t] = f1

        overall['edit_scores'].append(ed)
        rows.append({'file': name, 'frames': n, 'accuracy': acc, 'edit_score': ed,
                     'f1_10': f1vals[0.1], 'f1_25': f1vals[0.25], 'f1_50': f1vals[0.5]})

    # aggregate
    overall_acc = overall['matches'] / overall['frames_total'] if overall['frames_total'] > 0 else 0.0
    avg_edit = sum(overall['edit_scores']) / len(overall['edit_scores']) if overall['edit_scores'] else 0.0
    f1_agg = {}
    for t in (0.1, 0.25, 0.5):
        TP = overall['f1'][t]['TP']
        FP = overall['f1'][t]['FP']
        FN = overall['f1'][t]['FN']
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_agg[t] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return rows, overall_acc, avg_edit, f1_agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True, help='Reference directory with .txt files')
    p.add_argument('--pred', required=True, help='Prediction directory with .txt files')
    p.add_argument('--ignore-background', action='store_true', help='Ignore background segments for edit/F1')
    args = p.parse_args()

    rows, overall_acc, avg_edit, f1_agg = evaluate_directories(args.ref, args.pred, ignore_background=args.ignore_background)

    out_csv = os.path.join(args.pred, 'eval_summary.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['file','frames','accuracy','edit_score','f1@10','f1@25','f1@50'])
        for r in rows:
            writer.writerow([r['file'], r['frames'], f"{r['accuracy']:.4f}", f"{r['edit_score']:.2f}",
                             f"{r['f1_10']:.4f}", f"{r['f1_25']:.4f}", f"{r['f1_50']:.4f}"])
        writer.writerow([])
        writer.writerow(['overall', '', f"{overall_acc:.4f}", f"{avg_edit:.2f}", f"{f1_agg[0.1]:.4f}", f"{f1_agg[0.25]:.4f}", f"{f1_agg[0.5]:.4f}"])

    print('Per-file results:')
    for r in rows:
        print(f"{r['file']}: frames={r['frames']}, acc={r['accuracy']:.4f}, edit={r['edit_score']:.2f}, f1@10={r['f1_10']:.4f}, f1@25={r['f1_25']:.4f}, f1@50={r['f1_50']:.4f}")
    print('---')
    print(f'Overall frame-wise accuracy: {overall_acc:.4f}')
    print(f'Average edit score: {avg_edit:.2f}')
    print(f'F1@10: {f1_agg[0.1]:.4f}, F1@25: {f1_agg[0.25]:.4f}, F1@50: {f1_agg[0.5]:.4f}')
    print(f'Wrote summary CSV to: {out_csv}')


if __name__ == '__main__':
    main()
