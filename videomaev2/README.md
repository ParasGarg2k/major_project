# VidFeatX

**VidFeatX** is a modular and scalable video feature extractor built on top of **VideoMAEv2**. It supports frame-level feature extraction using multiple pretrained models including `VideoMAEv2-Base`, `VideoMAEv2-Large`, and `VideoMAEv2-Small`. Extracted features can be used for downstream video understanding tasks such as action segmentation, recognition, or localization.

---

## 🔧 Features

* ✅ Support for VideoMAEv2 base, large, and small variants
* ✅ Frame-wise feature extraction for any video dataset
* ✅ Easy integration with downstream pipelines (e.g., TAS models like MS-TCN++, ASFormer, etc.)
* ✅ Customizable video preprocessing
* ✅ Batch-wise GPU-accelerated inference

---

## 🏗️ Architecture

- Based on **VideoMAEv2**, a video Masked Autoencoder with Vision Transformer (ViT) backbone.
- Input videos are split into fixed-length clips (e.g., 16 frames), resized to 224×224.
- Clips are passed through the VideoMAEv2 **encoder** (without decoder) to extract spatio-temporal features.
- Frame-level features are obtained by pooling token embeddings per frame.
- Supports three pretrained variants: **Small (384-dim)**, **Base (768-dim)**, and **Large (1024-dim)**.
- Extracted features capture rich spatial and temporal video representations for downstream tasks.
- Modular and efficient: GPU-accelerated batch processing for fast feature extraction.

---

## 🧠 Models Supported

| Model Name       | Input Size | Feature Dim | Pretrained On |
| ---------------- | ---------- | ----------- | ------------- |
| VideoMAEv2-Small | 16×224×224 | 384         | Kinetics-400  |
| VideoMAEv2-Base  | 16×224×224 | 768         | Kinetics-400  |
| VideoMAEv2-Large | 16×224×224 | 1024        | Kinetics-400  |


---



## References

- I3D: Carreira, J. and Zisserman, A. "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset." CVPR 2017.

- VideoMAE v2: Wang, Y. et al. "VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking." CVPR 2023.
