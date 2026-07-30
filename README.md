# HDA-Gait

Official code release for hierarchical domain-aligned gait pretraining across
moving-UAV and ground-view silhouette domains. The implementation is built on
[OpenGait](https://github.com/ShiqiYu/OpenGait).

This repository contains the training code and reproducible six-GPU
configuration only. Datasets, checkpoints, experiment logs, generated figures,
and server-specific files are intentionally excluded.

## Method overview

HDA-Gait aligns the two input domains at three representation depths:

- **Low level:** input-adaptive fusion of Instance Normalization and synchronized
  Batch Normalization in the ResNet stem.
- **Middle level:** Gaussian-kernel MMD on pooled stage-2 representations.
- **High level:** a gradient-reversal domain classifier on the 31 projected HPP
  parts.
- **Instance discrimination:** a global two-view supervised contrastive
  objective over UAV and ground tracklets.

The main experiment configuration is
[`configs/gaitssb/pretrain_v4_6V100.yaml`](configs/gaitssb/pretrain_v4_6V100.yaml).

## Repository layout

```text
.
├── configs/
│   ├── default.yaml
│   └── gaitssb/pretrain_v4_6V100.yaml
├── datasets/
│   └── partition.example.json
├── opengait/
│   ├── data/                         # multi-domain dataset, sampler, q/k views
│   ├── evaluation/                   # OpenGait evaluation utilities
│   ├── modeling/
│   │   ├── backbones/resnet_domain.py
│   │   ├── losses/                   # SupCon, MMD and adversarial losses
│   │   └── models/                   # V3 base and V4 HDA-Gait model
│   └── main.py
├── tests/
├── environment.yml
└── requirements.txt
```

## Environment

The reported run used:

- Python 3.10
- PyTorch 2.5.1
- CUDA 12.1
- 6 × NVIDIA V100

Create the environment with either Conda:

```bash
conda env create -f environment.yml
conda activate hda-gait
```

or pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

SAM 2 is not required for model training; it was present in the original
development environment only and is therefore omitted here.

## Data preparation

The pretraining loader expects one serialized silhouette tracklet under a
domain directory for every identity:

```text
CombinedDataset_Pretrain/
├── 0000001/
│   └── 0/
│       └── 0000001.pkl
├── 0000002/
│   └── 0/
│       └── 0000002.pkl
└── 0107922/
    └── 1/
        └── 0107922.pkl
```

Domain labels are:

- `0`: UAV
- `1`: ground view

Each PKL file contains a silhouette array shaped `[T, H, W]`. The main
experiment retains sequences with at least 32 valid frames. Dataset files are
not distributed in this repository.

Copy the partition template and populate it with the retained identity names:

```bash
cp datasets/partition.example.json datasets/partition.json
```

Then update these two fields in the main YAML:

```yaml
data_cfg:
  dataset_root: /path/to/CombinedDataset_Pretrain
  dataset_partition: ./datasets/partition.json
```

## q/k temporal sampling

For every tracklet, q and k are sampled independently. Each view contains 16
ordered frames selected without replacement from a 20-frame local candidate
window (`16 + frames_skip_num=4`). No explicit non-overlap constraint is
applied, so q and k may partially overlap.

## Pretraining

The six-GPU configuration uses 18 identities per rank, giving a global identity
batch of 108:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --nproc_per_node=6 --master_port=29501 \
opengait/main.py \
--cfgs ./configs/gaitssb/pretrain_v4_6V100.yaml \
--phase train \
--log_to_file
```

The principal optimization settings are:

| Component | Setting |
|---|---:|
| SupCon weight | 1.0 |
| MMD weight | 0.1 |
| Adversarial weight | 0.05 |
| Initial learning rate | 0.04 |
| LR milestones | 60k, 100k |
| GRL schedule | `(0, 0.0) → (60k, 0.5) → (100k, 1.0)` |
| Total iterations | 150k |

Checkpoints and logs are written below `output/`, which is ignored by Git.

## Reproducibility notes

- The configuration reproduces the reported six-V100 run; only machine-specific
  dataset paths and the stale historical save name were sanitized.
- Distributed ranks use deterministic rank-based seeds through the OpenGait
  training entry point.
- MMD and adversarial losses are computed from each rank's local features; DDP
  averages the resulting parameter gradients.
- The code uses synchronized BN in the backbone when `sync_BN: true`.

## Checkpoints and dataset release

No checkpoint or dataset is included. If released later, host large artifacts
on a dedicated service such as Hugging Face rather than committing them to Git.
Publish their checksums and download instructions in this README.

## Citation

The HDA-Gait citation will be added when the paper metadata is public. Please
also cite the OpenGait framework:

```bibtex
@inproceedings{fan2023opengait,
  title     = {OpenGait: Revisiting Gait Recognition Towards Better Practicality},
  author    = {Fan, Chao and Liang, Junhao and Shen, Chuanfu and Hou, Saihui
               and Huang, Yongzhen and Yu, Shiqi},
  booktitle = {CVPR},
  year      = {2023}
}
```

## Acknowledgement and terms

This implementation is derived from OpenGait. See
[`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md) before publishing or
redistributing the repository. No separate license is asserted here because the
release owner must first confirm the intended license and its compatibility with
the upstream academic-use terms.

