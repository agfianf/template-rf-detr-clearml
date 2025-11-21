# RF-DETR Training Configuration Guide

This guide provides example configurations for training RF-DETR models with ClearML integration.

## Table of Contents

- [Image Segmentation with ONNX Export](#image-segmentation-with-onnx-export)
- [Image Object Detection](#image-object-detection)

---

## Image Segmentation with ONNX Export

### Model Choice: `seg_preview`

The `seg_preview` model variant is designed for **instance segmentation** tasks. Choose this model when you need:

- Pixel-level object boundaries (masks)
- Instance-aware segmentation output
- ONNX-compatible deployment

### Data Source: CVAT

CVAT (Computer Vision Annotation Tool) is recommended for segmentation because:

- Supports polygon and mask annotations
- Provides accurate boundary definitions
- Easy export to COCO format (required by RF-DETR)

### JSON Configuration Example

Create a file `config_segmentation.json`:

```json
{
  "model": {
    "model_type": "seg_preview",
    "pretrain_weights": null,
    "num_classes": null
  },
  "data": {
    "source_type": "cvat",
    "cvat": {
      "task_ids": [101, 102, 103],
      "annotations_only": false
    },
    "split": {
      "train_ratio": 0.8,
      "val_ratio": 0.1,
      "test_ratio": 0.1,
      "seed": 42
    },
    "output_dir": "./data/segmentation"
  },
  "training": {
    "epochs": 150,
    "batch_size": 2,
    "grad_accum_steps": 8,
    "lr": 1e-4,
    "resolution": 560,
    "device": "cuda",
    "num_workers": 4,
    "use_ema": true,
    "gradient_checkpointing": true,
    "checkpoint_interval": 10,
    "early_stopping": true,
    "early_stopping_patience": 15
  },
  "experiment": {
    "clearml": {
      "project_name": "Segmentation-Project",
      "task_name": "seg-training-v1",
      "tags": ["segmentation", "onnx", "production"],
      "output_uri": "s3://my-bucket/experiments",
      "auto_connect_frameworks": true
    },
    "logging": {
      "tensorboard": true,
      "log_interval": 10,
      "save_predictions": true,
      "max_prediction_images": 20
    },
    "output_dir": "./output/segmentation"
  },
  "validation": {
    "confidence_threshold": 0.25,
    "iou_threshold": 0.5
  },
  "export": {
    "format": "onnx",
    "half": false,
    "simplify": true
  },
  "use_clearml": true
}
```

### Running the Training

```bash
uv run src/run_training.py --config config_segmentation.json
```

### Key Configuration Notes

| Parameter | Value | Reason |
|-----------|-------|--------|
| `model_type` | `seg_preview` | Instance segmentation capability |
| `batch_size` | `2` | Segmentation requires more GPU memory |
| `grad_accum_steps` | `8` | Effective batch size = 16 |
| `gradient_checkpointing` | `true` | Reduces memory for mask prediction |
| `export.format` | `onnx` | ONNX for cross-platform deployment |
| `export.simplify` | `true` | Optimized ONNX graph |

---

## Image Object Detection

### Model Choices

| Model Type | Use Case | Speed | Accuracy |
|------------|----------|-------|----------|
| `nano` | Edge devices, real-time | Fastest | Good |
| `small` | Mobile/embedded | Fast | Better |
| `medium` | Balanced | Moderate | High |
| `base` | General purpose | Moderate | High |
| `large` | Maximum accuracy | Slow | Highest |

### CLI Example (Quick Start)

```bash
# Using local dataset
uv run src/run_training.py \
  --dataset-dir ./data/my-coco-dataset \
  --model-type base \
  --epochs 100 \
  --batch-size 4 \
  --project-name "Detection-Project" \
  --task-name "det-experiment-1" \
  --export-format onnx

# Using CVAT data source
uv run src/run_training.py \
  --cvat-task-ids 201 202 203 \
  --model-type large \
  --epochs 150 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --lr 5e-5 \
  --project-name "Detection-Project" \
  --tags detection production v2 \
  --tensorboard
```

### JSON Configuration Example

Create a file `config_detection.json`:

```json
{
  "model": {
    "model_type": "base",
    "pretrain_weights": null,
    "num_classes": null
  },
  "data": {
    "source_type": "local",
    "local": {
      "path": "./data/coco-detection"
    },
    "split": {
      "train_ratio": 0.85,
      "val_ratio": 0.1,
      "test_ratio": 0.05,
      "seed": 42
    },
    "output_dir": "./data/detection"
  },
  "training": {
    "epochs": 100,
    "batch_size": 4,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "lr_encoder": 1e-5,
    "resolution": 560,
    "weight_decay": 0.0001,
    "device": "cuda",
    "num_workers": 8,
    "use_ema": true,
    "gradient_checkpointing": false,
    "checkpoint_interval": 10,
    "early_stopping": false
  },
  "experiment": {
    "clearml": {
      "project_name": "Detection-Project",
      "task_name": "det-training-v1",
      "tags": ["detection", "baseline"],
      "output_uri": null,
      "auto_connect_frameworks": true
    },
    "logging": {
      "tensorboard": false,
      "log_interval": 10,
      "save_predictions": true,
      "max_prediction_images": 30
    },
    "output_dir": "./output/detection"
  },
  "validation": {
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5
  },
  "export": {
    "format": "onnx",
    "half": true,
    "simplify": true
  },
  "use_clearml": true
}
```

### Running Without ClearML

```bash
# Disable ClearML tracking for local experiments
uv run src/run_training.py \
  --dataset-dir ./data/my-dataset \
  --epochs 50 \
  --no-clearml
```

---

## Data Source Configuration

### CVAT

Required environment variables (see `.env.example`):

```env
CVAT_HOST=https://cvat.example.com
CVAT_USERNAME=your_username
CVAT_PASSWORD=your_password
```

### MinIO/S3

```json
{
  "data": {
    "source_type": "minio",
    "minio": {
      "bucket": "datasets",
      "prefix": "coco/my-project"
    }
  }
}
```

Required environment variables:

```env
MINIO_ENDPOINT=s3.example.com
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
```

---

## Export Formats

| Format | Use Case | Notes |
|--------|----------|-------|
| `onnx` | Production deployment | Cross-platform, TensorRT compatible |
| `torchscript` | PyTorch ecosystem | JIT compiled |

### ONNX Export Options

```json
{
  "export": {
    "format": "onnx",
    "half": true,
    "simplify": true
  }
}
```

- `half: true` - FP16 precision (smaller model, faster inference)
- `simplify: true` - Optimized graph (recommended for deployment)
