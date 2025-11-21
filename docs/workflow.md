# RF-DETR Training Pipeline Workflow

This document describes the complete workflow of the RF-DETR training pipeline, including data flow, component interactions, and integration points.

## High-Level Overview

```mermaid
flowchart TB
    subgraph Entry["Entry Points"]
        CLI[run_training.py<br/>CLI Arguments]
        CONFIG[JSON Config File]
    end

    subgraph Core["Core Training"]
        TRAIN[train.py<br/>Main Orchestrator]
        MODEL[RF-DETR Model<br/>Base/Large/SegPreview]
    end

    subgraph Data["Data Sources"]
        CVAT[(CVAT<br/>Annotation Platform)]
        MINIO[(MinIO/S3<br/>Object Storage)]
        LOCAL[(Local<br/>COCO Dataset)]
    end

    subgraph Integration["Integrations"]
        CLEARML[ClearML<br/>Experiment Tracking]
        TB[TensorBoard<br/>Visualization]
    end

    subgraph Output["Outputs"]
        CKPT[Model Checkpoint<br/>.pth]
        EXPORT[Exported Model<br/>ONNX/TorchScript]
        VIZ[Prediction<br/>Visualizations]
    end

    CLI --> TRAIN
    CONFIG --> CLI
    CVAT --> TRAIN
    MINIO --> TRAIN
    LOCAL --> TRAIN
    TRAIN --> MODEL
    MODEL --> CKPT
    CKPT --> EXPORT
    TRAIN --> CLEARML
    TRAIN --> TB
    TRAIN --> VIZ
```

## Detailed Training Flow

```mermaid
flowchart TD
    START([Start]) --> PARSE[Parse CLI Arguments]
    PARSE --> BUILD[Build Configuration Schemas]
    BUILD --> INIT_CLEARML{ClearML<br/>Enabled?}

    INIT_CLEARML -->|Yes| TASK[Initialize ClearML Task]
    INIT_CLEARML -->|No| DATA
    TASK --> CONNECT[Connect Parameters to ClearML UI]
    CONNECT --> DATA

    DATA[Download/Prepare Dataset]
    DATA --> INFO[Extract Dataset Info<br/>Classes, Splits, Statistics]
    INFO --> LOG_DATA[Log Dataset Info]
    LOG_DATA --> INIT_MODEL[Initialize RF-DETR Model]

    INIT_MODEL --> TRAIN_LOOP[Training Loop]

    subgraph Training["Training Loop"]
        TRAIN_LOOP --> FORWARD[Forward Pass]
        FORWARD --> BACKWARD[Backward Pass]
        BACKWARD --> METRICS[Log Metrics]
        METRICS --> CKPT_CHECK{Checkpoint<br/>Interval?}
        CKPT_CHECK -->|Yes| SAVE_CKPT[Save Checkpoint]
        CKPT_CHECK -->|No| EARLY{Early Stopping<br/>Triggered?}
        SAVE_CKPT --> EARLY
        EARLY -->|No| EPOCH{More<br/>Epochs?}
        EARLY -->|Yes| BEST
        EPOCH -->|Yes| FORWARD
        EPOCH -->|No| BEST
    end

    BEST[Find Best Checkpoint]
    BEST --> LOG_MODEL[Log Model to ClearML]
    LOG_MODEL --> VIZ_CHECK{Save<br/>Predictions?}

    VIZ_CHECK -->|Yes| PREDICT[Generate Predictions]
    VIZ_CHECK -->|No| EXPORT_CHECK
    PREDICT --> ANNOTATE[Annotate Images]
    ANNOTATE --> LOG_VIZ[Log Visualization Grids]
    LOG_VIZ --> EXPORT_CHECK

    EXPORT_CHECK{Export<br/>Model?}
    EXPORT_CHECK -->|Yes| EXPORT[Export to ONNX/TorchScript]
    EXPORT_CHECK -->|No| CLOSE
    EXPORT --> CLOSE

    CLOSE[Close ClearML Task]
    CLOSE --> END([Return Best Checkpoint Path])
```

## Data Acquisition Flow

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        CVAT[(CVAT)]
        MINIO[(MinIO)]
        LOCAL[(Local)]
    end

    subgraph Handler["DataHandler"]
        ROUTE{Route by<br/>source_type}
        DL_CVAT[Download from CVAT]
        DL_MINIO[Download from MinIO]
        PREP_LOCAL[Validate Local Path]
        MERGE[Merge COCO Datasets]
    end

    subgraph Downloaders["Downloaders"]
        SDK[CVATSDKDownloader]
        HTTP1[CVATHTTPDownloaderV1]
        HTTP2[CVATHTTPDownloaderV2]
        MINIO_DL[MinioDatasetDownloader]
    end

    subgraph Output["Output Format"]
        COCO[COCO Dataset]
        TRAIN_SPLIT[train/<br/>_annotations.coco.json]
        VALID_SPLIT[valid/<br/>_annotations.coco.json]
        TEST_SPLIT[test/<br/>_annotations.coco.json]
    end

    CVAT --> ROUTE
    MINIO --> ROUTE
    LOCAL --> ROUTE

    ROUTE -->|CVAT| DL_CVAT
    ROUTE -->|MINIO| DL_MINIO
    ROUTE -->|LOCAL| PREP_LOCAL

    DL_CVAT --> SDK
    DL_CVAT --> HTTP1
    DL_CVAT --> HTTP2
    DL_MINIO --> MINIO_DL

    SDK --> MERGE
    HTTP1 --> MERGE
    HTTP2 --> MERGE
    MINIO_DL --> MERGE
    PREP_LOCAL --> COCO

    MERGE --> COCO
    COCO --> TRAIN_SPLIT
    COCO --> VALID_SPLIT
    COCO --> TEST_SPLIT
```

## ClearML Integration

```mermaid
flowchart TB
    subgraph TaskManager["TaskManager"]
        INIT[init_task]
        CONNECT_P[connect_params]
        GET_CFG[get_*_config]
        LOG_DS[log_dataset_info]
        LOG_M[log_model]
        CLOSE_T[close]
    end

    subgraph Callbacks["ClearMLCallback"]
        EPOCH_START[on_train_epoch_start]
        EPOCH_END[on_train_epoch_end]
        VAL_END[on_validation_end]
        BATCH_END[on_batch_end]
        LOG_IMG[log_image]
    end

    subgraph ClearML["ClearML Platform"]
        UI[Web UI<br/>Parameter Editor]
        DASHBOARD[Dashboard<br/>Metrics & Charts]
        ARTIFACTS[Artifacts<br/>Models & Data]
        DEBUG[Debug Samples<br/>Visualizations]
    end

    INIT --> UI
    CONNECT_P --> UI
    UI --> GET_CFG

    EPOCH_END --> DASHBOARD
    VAL_END --> DASHBOARD
    BATCH_END --> DASHBOARD

    LOG_DS --> ARTIFACTS
    LOG_M --> ARTIFACTS
    LOG_IMG --> DEBUG
```

## Configuration Schema Hierarchy

```mermaid
classDiagram
    class TrainingConfig {
        +int epochs
        +int batch_size
        +int grad_accum_steps
        +float lr
        +int resolution
        +float weight_decay
        +str device
        +bool use_ema
        +bool early_stopping
        +int early_stopping_patience
    }

    class ModelConfig {
        +ModelType model_type
        +str pretrain_weights
        +int num_classes
    }

    class DataConfig {
        +DataSourceType source_type
        +CVATDataSource cvat
        +MinIODataSource minio
        +LocalDataSource local
        +DataSplitConfig split
        +str output_dir
    }

    class ExperimentConfig {
        +ClearMLConfig clearml
        +LoggingConfig logging
        +str output_dir
    }

    class ValidationConfig {
        +float confidence_threshold
        +float iou_threshold
    }

    class ExportConfig {
        +str format
        +bool half
        +bool simplify
    }

    DataConfig --> CVATDataSource
    DataConfig --> MinIODataSource
    DataConfig --> LocalDataSource
    DataConfig --> DataSplitConfig
    ExperimentConfig --> ClearMLConfig
    ExperimentConfig --> LoggingConfig
```

## Model Types

```mermaid
flowchart LR
    subgraph Types["ModelType Enum"]
        BASE[BASE]
        LARGE[LARGE]
        SEG[SEG_PREVIEW]
        NANO[NANO]
        SMALL[SMALL]
        MEDIUM[MEDIUM]
    end

    subgraph Models["RF-DETR Classes"]
        RFBASE[RFDETRBase<br/>Trainable]
        RFLARGE[RFDETRLarge<br/>Trainable]
        RFSEG[RFDETRSegPreview<br/>Trainable]
    end

    BASE --> RFBASE
    LARGE --> RFLARGE
    SEG --> RFSEG
    NANO --> RFBASE
    SMALL --> RFBASE
    MEDIUM --> RFBASE
```

## Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| **CLI Entry** | `run_training.py` | Parse arguments, build schemas, invoke training |
| **Training Orchestrator** | `train.py` | Coordinate all training steps, model lifecycle |
| **Environment Config** | `config.py` | Load credentials from `.env` (CVAT, MinIO, ClearML) |
| **Data Handler** | `helpers/data_handler.py` | Download, merge, and prepare datasets |
| **CVAT Downloader** | `integrations/data/downloader_cvat.py` | Fetch datasets from CVAT (SDK/HTTP) |
| **MinIO Downloader** | `integrations/data/downloader_minio.py` | Fetch datasets from S3-compatible storage |
| **Task Manager** | `integrations/clearml/task_manager.py` | ClearML task lifecycle and parameter management |
| **Callbacks** | `integrations/clearml/callbacks.py` | Log training metrics and images to ClearML |
| **Visualization** | `helpers/visualization.py` | Generate prediction visualizations |
| **Schemas** | `schemas/*.py` | Type-safe Pydantic configuration models |
| **Default Params** | `configs/params.py` | Centralized default configuration values |

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Parameter Flow                               │
├─────────────────────────────────────────────────────────────────────┤
│  CLI args → Schemas → TaskManager.connect_params() → ClearML UI     │
│                            ↓                                         │
│              TaskManager.get_*_config() → Training                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                           Data Flow                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Source (CVAT/MinIO/Local) → Downloader → DataHandler → COCO        │
│                                              ↓                       │
│                                    model.train(dataset_dir)          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         Monitoring Flow                              │
├─────────────────────────────────────────────────────────────────────┤
│  model.train() → ClearMLCallback → Logger.report_* → ClearML UI     │
└─────────────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Basic Local Training

```bash
uv run src/run_training.py \
    --dataset-dir ./data/my-dataset \
    --epochs 100 \
    --batch-size 8
```

### Training with CVAT Data

```bash
uv run src/run_training.py \
    --cvat-task-ids 123 456 789 \
    --project-name "MyProject" \
    --model-type large \
    --epochs 150
```

### Training without ClearML

```bash
uv run src/run_training.py \
    --dataset-dir ./data/my-dataset \
    --no-clearml \
    --tensorboard
```

### Full Configuration

```bash
uv run src/run_training.py \
    --cvat-task-ids 100 \
    --project-name "ObjectDetection" \
    --task-name "experiment-v1" \
    --model-type base \
    --epochs 200 \
    --batch-size 4 \
    --lr 0.0001 \
    --resolution 640 \
    --early-stopping \
    --early-stopping-patience 20 \
    --export-format onnx \
    --tags "production" "v1.0"
```
