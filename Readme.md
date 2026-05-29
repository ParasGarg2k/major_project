# Deep Temporal Networks Based Human Activity Recognition

Temporal action segmentation plays an important role in human activity recognition, especially when
dealing with long, untrimmed videos. It helps a system understand where one action starts and where
another ends, which is essential for making sense of human behavior over time. This has value in
many real-world areas, including healthcare, surveillance, sports analysis, assistive technologies, and
human-computer interaction.
In this project, we explore some of the most effective temporal modeling techniques for understanding
fine-grained actions in videos. To evaluate their performance, we use three challenging benchmark
datasets: GTEA, Breakfast & 50Salads. For feature extraction, we rely on four strong spatiotemporal
backbones. I3D is used for its ability to capture motion and activity patterns across time. Video
MAEv2 helps learn richer temporal relationships through its transformer-based approach. SlowFast
is designed to capture both slower semantic changes and faster motion cues through its two-pathway
structure. ResNet3D, extends traditional residual networks into the temporal domain to learn mean
ingful features from video sequences.
These features are then passed into the models developed taking state-of-the-art temporal segmenta
tion models as reference architectures such as MS-TCN, MS-TCN++, ASFormer & FACT. Each of
these models has its own strengths when it comes to recognizing short, quick actions as well as longer
and morecomplex activities. Together, they help improve the consistency and accuracy of frame-level
action prediction. The goal of this project is to improve how systems understand and segment human
actions in continuous video streams, making activity recognition more accurate, reliable, and useful
across a wide range of practical applications.