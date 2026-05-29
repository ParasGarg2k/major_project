import os
import io
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import cv2
import tempfile

# Import eval utilities from this folder
import eval as ev


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GT_DIR = os.path.join(BASE, 'gtea', 'ground_truth')
PRED_ROOT = os.path.join(os.path.dirname(__file__))

ALLOWED = ['take','open','pour','close','shake','scoop','stir','put','fold','spread','background']


def read_labels_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f.readlines()]


def labels_to_ints(labels, mapping):
    return [mapping[l] if l in mapping else mapping.get('background', 0) for l in labels]


def plot_bars_and_match(gt_labels, pred_labels, save_bytes=False, current_frame=None):
    # Create label->int mapping for colors
    unique = ALLOWED[:]  # ensure ordering
    mapping = {l: i for i, l in enumerate(unique)}
    pred_int = labels_to_ints(pred_labels, mapping)
    gt_int = labels_to_ints(gt_labels, mapping)
    frames = min(len(gt_int), len(pred_int))
    pred_arr = np.array(pred_int[:frames])[None, :]
    gt_arr = np.array(gt_int[:frames])[None, :]

    match = np.array([1 if gt_labels[i]==pred_labels[i] else 0 for i in range(frames)])
    # smooth match for visualization
    win = max(1, frames // 50)
    match_smooth = np.convolve(match, np.ones(win)/win, mode='same')

    fig = plt.figure(figsize=(14, 3.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.6, 0.6, 1.0], hspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    # use a discrete colormap with one color per label to ensure consistent mapping
    discrete_cmap = plt.get_cmap('tab20', len(unique))
    ax0.imshow(pred_arr, aspect='auto', cmap=discrete_cmap)
    ax0.set_yticks([0])
    ax0.set_yticklabels(['Predicted'])
    ax0.set_xticks([])

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(gt_arr, aspect='auto', cmap=discrete_cmap)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['GroundTruth'])
    ax1.set_xticks([])

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(match_smooth, color='tab:blue')
    ax2.fill_between(range(frames), match_smooth, color='tab:blue', alpha=0.1)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Match (smoothed)')
    if current_frame is not None and 0 <= current_frame < frames:
        ax2.axvline(current_frame, color='red', linestyle='--', linewidth=1)
        ax0.axvline(current_frame, color='red', linestyle='--', linewidth=1)
        ax1.axvline(current_frame, color='red', linestyle='--', linewidth=1)

    if save_bytes:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    else:
        plt.close(fig)
        return fig


def main():
    st.title('Ground-truth vs Predictions Explorer')

    st.sidebar.header('Dataset')
    pred_dirs = [d for d in os.listdir(PRED_ROOT) if os.path.isdir(os.path.join(PRED_ROOT, d)) and d.startswith('gtea')]
    if not pred_dirs:
        st.sidebar.error('No prediction directories found under web/results')
        return
    pred_dirs = ['GTEA', 'Breakfast', '50Salads']
    pred_choice = st.sidebar.selectbox('Prediction folder', pred_dirs)
    pred_dir = os.path.join(PRED_ROOT, pred_choice)

    uploaded = st.file_uploader('Upload video file (filename used to find label files)', type=['mp4','avi','mov','mkv'])
    st.markdown('Or enter video file basename (without extension) to lookup labels:')
    manual_name = st.text_input('Video basename (e.g., S1_Cheese_C1)')

    video_frames = None
    video_fps = None
    if uploaded is not None:
        video_name = os.path.splitext(uploaded.name)[0]
        # save to temp file and extract frames
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded.name.split('.')[-1])
        tfile.write(uploaded.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = []
        ret = True
        max_frames = 10000
        while ret and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        video_frames = frames
    else:
        video_name = manual_name.strip()

    if not video_name:
        st.info('Provide a video file or a basename to continue')
        return

    gt_path = os.path.join(GT_DIR, video_name + '.txt')
    pred_path = os.path.join(pred_dir, video_name + '.txt')

    if not os.path.exists(gt_path):
        st.error(f'Reference ground-truth not found: {gt_path}')
        return
    if not os.path.exists(pred_path):
        st.error(f'Prediction file not found: {pred_path}')
        return

    gt_labels = read_labels_file(gt_path)
    pred_labels = read_labels_file(pred_path)

    frames = min(len(gt_labels), len(pred_labels))
    matches = sum(1 for i in range(frames) if gt_labels[i] == pred_labels[i])
    acc = matches / frames * 100.0
    edit = ev.edit_score(gt_labels, pred_labels)
    f1s = {}
    for t in (0.1, 0.25, 0.5):
        _,_,_,_,_,f1 = ev.f1_at_threshold(gt_labels, pred_labels, t)
        f1s[t] = f1 * 100.0

    st.subheader("Metrics")

    # First row: acc and edit
    col1, col2 = st.columns(2)
    col1.metric("Frame-wise accuracy", f"{acc:.2f}%")
    col2.metric("Edit score", f"{edit:.2f}%")

    # Second row: f1
    st.metric(
        "F1@10/25/50",
        f"{f1s[0.1]:.2f}% / {f1s[0.25]:.2f}% / {f1s[0.5]:.2f}%"
    )

    # Frame-by-frame table
    df = pd.DataFrame({
        'frame': list(range(frames)),
        'gt': gt_labels[:frames],
        'pred': pred_labels[:frames],
        'match': [1 if gt_labels[i]==pred_labels[i] else 0 for i in range(frames)]
    })

    st.subheader('Frame-by-frame predictions')
    st.dataframe(df.head(200))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download per-frame CSV', data=csv, file_name=f'{video_name}_framewise.csv', mime='text/csv')

    st.subheader('Visual comparison')
    # legend
    # st.markdown('**Per-class color legend**')
    # cols = st.columns(6)
    # # build a discrete colormap to match the plotting colormap
    # discrete_cmap = plt.get_cmap('tab20', len(ALLOWED))
    # for i, label in enumerate(ALLOWED):
    #     col = cols[i % len(cols)]
    #     rgba = discrete_cmap(i)
    #     hexc = mpl.colors.to_hex(rgba)
    #     col.markdown(f"<div style='display:flex;align-items:center'><div style='width:18px;height:12px;background:{hexc};margin-right:6px'></div>{label}</div>", unsafe_allow_html=True)

    current_frame = None
    if video_frames is not None and len(video_frames) > 0:
        st.video(uploaded)
        st.markdown('Use the slider to step through frames (synchronized with labels)')
        frame_idx = st.slider('Frame', 0, min(len(video_frames)-1, frames-1), 0)
        current_frame = frame_idx
        frame = video_frames[frame_idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels='RGB', caption=f'Frame {frame_idx}')
    else:
        # show only bars
        current_frame = st.slider('Frame (visual marker)', 0, max(0, frames-1), 0)

    # st.subheader('Visual comparison')
    # legend
    st.markdown('**Per-class color legend**')
    cols = st.columns(6)
    # build a discrete colormap to match the plotting colormap
    discrete_cmap = plt.get_cmap('tab20', len(ALLOWED))
    for i, label in enumerate(ALLOWED):
        col = cols[i % len(cols)]
        rgba = discrete_cmap(i)
        hexc = mpl.colors.to_hex(rgba)
        col.markdown(f"<div style='display:flex;align-items:center'><div style='width:18px;height:12px;background:{hexc};margin-right:6px'></div>{label}</div>", unsafe_allow_html=True)

    fig = plot_bars_and_match(gt_labels, pred_labels, current_frame=current_frame)
    st.pyplot(fig)

    # allow download of visualization
    buf = plot_bars_and_match(gt_labels, pred_labels, save_bytes=True, current_frame=current_frame)
    st.download_button('Download visualization PNG', data=buf, file_name=f'{video_name}_comparison.png', mime='image/png')


if __name__ == '__main__':
    main()
