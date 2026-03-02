# cryoEV

Instance segmentation of extracellular vesicles in cryo-EM micrographs.

Supports two model families:

- **UNet / FPN / DeepLabV3+** (via `segmentation_models_pytorch`) for pixel-level binary segmentation with connected-component instance extraction.
- **YOLOv8-seg** (via `ultralytics`) for direct instance segmentation with polygon masks.

Both pipelines share a **confidence-based interactive review** system that lets you inspect and accept/reject low-confidence detections before saving final results.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
cryoEV/
├── README.md
├── requirements.txt
├── data_utils/
│   ├── clean_filename.py       # Remove .rf.<hash> artifacts from filenames
│   ├── clean_label.py          # Remap / filter YOLO class IDs
│   └── split_dataset.py        # Random train/val/test split
├── datasets/
│   └── cryo_instance_dataset.py  # PyTorch Dataset for YOLO-polygon labels
├── transforms/
│   └── cryo_transforms.py      # Albumentations pipelines (light/standard/heavy)
├── training/
│   ├── train_unet.py           # UNet/FPN/DeepLabV3+ training loop
│   └── train_yolo.py           # YOLOv8-seg training + evaluation
├── inference/
│   └── inference.py            # Prediction, confidence scoring, interactive review
└── visualization/
    └── training_curves.py      # Loss/IoU/F1 curve plotting
```

## Data Preparation

### Expected folder layout

```
dataset/
├── images/
│   ├── sample_001.jpg
│   └── ...
└── labels/
    ├── sample_001.txt    # YOLO polygon format
    └── ...
```

Each label file uses **YOLO polygon format**: one line per object, `class_id x1 y1 x2 y2 ...` with normalised coordinates in [0, 1].

### Cleaning and splitting

```bash
# Remove .rf.<hash> from filenames exported by Roboflow
python -m data_utils.clean_filename

# Remap all classes to 0 and remove unwanted classes
python -m data_utils.clean_label

# Split into train / val / test
python -m data_utils.split_dataset
```

Edit the paths and config dicts inside each script before running.

## Training

### UNet / FPN / DeepLabV3+

```bash
python -m training.train_unet
```

Key parameters (edit `CONFIG` dict in `training/train_unet.py`):

| Parameter | Default | Description |
|---|---|---|
| `image_size` | 1024 | Input resolution |
| `batch_size` | 8 | Batch size |
| `epochs` | 300 | Max epochs |
| `patience` | 15 | Early-stopping patience |
| `base_lr` | 5e-4 | Initial learning rate |
| `augmentation` | `'light'` | `'light'`, `'standard'`, or `'heavy'` |
| `scheduler` | `'plateau'` | `'plateau'`, `'cosine'`, or `'onecycle'` |

### YOLOv8-seg

```bash
python -m training.train_yolo
```

See `training/train_yolo.py` for configuration options including YOLO model size, epochs, and evaluation thresholds.

### Visualise training curves

```bash
# Single experiment
python -m visualization.training_curves results/unet_resnet34_combined

# Compare multiple experiments
python -m visualization.training_curves results/unet_resnet34_combined results/fpn_resnet34_combined
```

## Inference

### Basic (no review)

```bash
python -m inference.inference
```

By default `accept_all=True`, so all detected objects are saved without interactive review.

### With interactive confidence review

Set `accept_all=False` and choose a `confidence_threshold` in the config:

```python
from inference.inference import predict_with_review

predict_with_review(
    model_type='unet',                    # or 'yolo'
    image_path='test_image.png',
    output_dir='review_output',
    checkpoint_path='best_model_iou.pth', # UNet checkpoint
    confidence_threshold=0.5,
    accept_all=False,                     # enables popup review
)
```

For each object with confidence below the threshold, a matplotlib popup will appear showing:

- **Left panel**: full image with all objects overlaid (green = accepted, yellow = under review).
- **Right panel**: zoomed crop of the current object.
- **Buttons**: Accept / Reject.

Decisions are saved to a CSV file (`<image_name>_decisions.csv`) with columns: `object_index`, `confidence`, `decision`, `bbox_x_min`, `bbox_y_min`, `bbox_x_max`, `bbox_y_max`.

### Key functions

| Function | Description |
|---|---|
| `predict_single_image()` | UNet inference on one image |
| `batch_inference()` | UNet inference on a DataLoader |
| `extract_instances_unet()` | Connected-component extraction with per-object confidence |
| `extract_instances_yolo()` | YOLO mask extraction (wraps `load_predictions_from_model`) |
| `interactive_review_objects()` | Popup-based accept/reject for low-confidence objects |
| `predict_with_review()` | End-to-end orchestrator: predict -> review -> save |
