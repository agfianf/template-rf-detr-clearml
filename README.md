# RF-DETR with ClearML + CVAT

A template for training RF-DETR object detection and segmentation models with ClearML experiment tracking and CVAT/MinIO data integration.

## Features

- **RF-DETR Training** - Support for Base, Large, and Segmentation models
- **ClearML Integration** - Experiment tracking, parameter management, artifact storage
- **Multiple Data Sources** - CVAT, MinIO/S3, or local datasets
- **Pydantic Schemas** - Type-safe configuration management
- **CLI Interface** - Easy-to-use command line training

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd template-rf-detr

# Install dependencies with uv
uv sync
```

## Configuration

Copy the example environment file and configure your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# CVAT Configuration
CVAT_HOST=https://cvat.example.com
CVAT_USERNAME=your_username
CVAT_PASSWORD=your_password

# ClearML Configuration
CLEARML_WEB_HOST=https://app.clear.ml
CLEARML_API_HOST=https://api.clear.ml
CLEARML_FILES_HOST=https://files.clear.ml
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key
```

## Usage

### Train with Local Dataset

```bash
uv run src/run_training.py \
    --dataset-dir ./data/my-dataset \
    --model-type base \
    --epochs 100 \
    --batch-size 4 \
    --project-name "RF-DETR" \
    --task-name "experiment-1"
```

### Train with CVAT Data Source

```bash
uv run src/run_training.py \
    --cvat-task-ids 123 456 789 \
    --model-type seg_preview \
    --epochs 50 \
    --project-name "Segmentation"
```

### Train without ClearML

```bash
uv run src/run_training.py \
    --dataset-dir ./data/my-dataset \
    --no-clearml \
    --output-dir ./output
```

### Using JSON Config File

```bash
uv run src/run_training.py --config config.json
```

Example `config.json`:

```json
{
    "training": {
        "epochs": 100,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "lr": 0.0001,
        "resolution": 560,
        "early_stopping": true
    },
    "model": {
        "model_type": "base",
        "pretrain_weights": null
    },
    "data": {
        "source_type": "local",
        "local": {
            "path": "./data/my-dataset"
        }
    },
    "experiment": {
        "clearml": {
            "project_name": "RF-DETR",
            "task_name": "training-run"
        },
        "output_dir": "./output"
    }
}
```

## Dataset Format

RF-DETR expects datasets in COCO format:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    └── ...
```

## Model Types

| Model Type | Description | Training Support |
|------------|-------------|------------------|
| `base` | RF-DETR Base (default) | Yes |
| `large` | RF-DETR Large | Yes |
| `seg_preview` | RF-DETR Segmentation | Yes |
| `nano` | RF-DETR Nano | Inference only |
| `small` | RF-DETR Small | Inference only |
| `medium` | RF-DETR Medium | Inference only |

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-dir` | Path to local dataset | - |
| `--cvat-task-ids` | CVAT task IDs to download | - |
| `--model-type` | Model variant | `base` |
| `--epochs` | Training epochs | `100` |
| `--batch-size` | Batch size per GPU | `4` |
| `--grad-accum-steps` | Gradient accumulation | `4` |
| `--lr` | Learning rate | `1e-4` |
| `--resolution` | Input resolution | `560` |
| `--project-name` | ClearML project | `RF-DETR` |
| `--task-name` | ClearML task name | `training` |
| `--no-clearml` | Disable ClearML | `false` |
| `--early-stopping` | Enable early stopping | `false` |
| `--output-dir` | Output directory | `./output` |

## Project Structure

```
src/
├── config.py                 # Environment settings
├── configs/
│   ├── model_config.py       # Model configurations
│   └── params.py             # Default parameters
├── data/
│   └── downloader/
│       └── base_downloader.py
├── helpers/
│   ├── clearml_utils.py      # ClearML utilities
│   ├── data_handler.py       # Dataset orchestration
│   └── visualization.py      # Prediction visualization
├── integrations/
│   ├── clearml/
│   │   ├── callbacks.py      # Training callbacks
│   │   └── task_manager.py   # Task management
│   └── data/
│       ├── downloader_cvat.py
│       └── downloader_minio.py
├── schemas/
│   ├── data.py               # Data config schemas
│   ├── experiment.py         # Experiment schemas
│   ├── model.py              # Model schemas
│   └── training.py           # Training schemas
├── train.py                  # Main training script
└── run_training.py           # CLI entry point
```

## ClearML Features

When running with ClearML enabled:

- **Parameter Tracking** - All hyperparameters logged and editable from UI
- **Metric Logging** - Training loss, validation mAP, learning rate
- **Artifact Storage** - Best model checkpoints uploaded to S3/GCS
- **Prediction Visualization** - Sample predictions logged as images
- **Remote Execution** - Clone and run experiments on ClearML agents

## License

MIT
