# SlowFast Network

## Model Architecture
- **Type**: Two-Stream Pathway (Slow + Fast)
- **Fast Pathway**: High temporal resolution (all 32 frames), low channel capacity. Captures motion.
- **Slow Pathway**: Low temporal resolution (4 frames, stride 8), high channel capacity. Captures spatial details.
- **Fusion**: Lateral connections fuse Fast features into Slow pathway at multiple stages to integrate motion information.
- **Input**: 32 Frames.

## Dataset Structure
Expects `Dataset` folder in parent directory.
```
Dataset/
├── violence/
└── no-violence/
```

## How to Run
1. Install dependencies: `torch`, `opencv-python`, `scikit-learn`, `numpy`.
2. Run `python train.py`.

## HuggingFace
Model Repository Link: https://huggingface.co/LEGIONM36/Video-Classification-SlowFast-Model
