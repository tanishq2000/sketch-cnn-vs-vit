# sketch-cnn-vs-vit
Should a sketch app ship a CNN or a vision transformer? Frozen ResNet-50 vs DeiT-Small/Base on 20 classes of ImageNet-Sketch, comparing accuracy against latency and memory, plus calibration, failure analysis and confidence-based abstention. Duplicate-aware split, pinned checkpoints, bootstrap CIs.

<h1 align="center">CNN or Vision Transformer for Sketch Classification</h1>

<p align="center">
  <b>Should a sketch app ship a CNN or a vision transformer?</b><br>
  Three frozen ImageNet models, measured on accuracy <i>and</i> cost. Nothing is trained.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="timm" src="https://img.shields.io/badge/timm-1.0-0A7BBB">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-ImageNet--Sketch-6C757D">
  <img alt="Track" src="https://img.shields.io/badge/Task-Model%20evaluation-2EA44F">
</p>

---

## Table of contents

- [Overview](#overview)
- [Models and data](#models-and-data)
- [Results](#results)
- [How this is useful](#how-this-is-useful)
- [Setup](#setup)
- [Reproducibility](#reproducibility)
- [Repository structure](#repository-structure)
- [Limitations](#limitations)
- [Data and licences](#data-and-licences)
- [AI assistance](#ai-assistance)
- [References](#references)

---

## Overview

This project takes three pretrained classifiers from the `timm` library, runs them unchanged on hand-drawn sketches, and reports:

| What is measured | How |
|---|---|
| **Accuracy** | Overall, per-class precision, recall and F1, confusion matrices |
| **Cost** | Parameter count, milliseconds per image on GPU and CPU, peak GPU memory |
| **Confidence against correctness** | Reliability diagrams, expected calibration error (ECE), accuracy against coverage |
| **Failures** | The most confident errors and what they have in common |
| **Attention** | One attention map for the vision transformer |
| **Mitigation** | The model abstains ("not sure") below a confidence threshold |

It was built as Assignment 2 of the Computer Vision course (AIML ZG525) in the M.Tech AI/ML programme at BITS Pilani WILP.

## Models and data

| Role | Model | `timm` checkpoint | Parameters |
|---|---|---|---:|
| CNN baseline | ResNet-50 | `resnet50.a1_in1k` | 25.6 M |
| ViT baseline | DeiT-Small | `deit_small_patch16_224.fb_in1k` | 22.1 M |
| Intervention (larger ViT) | DeiT-Base | `deit_base_patch16_224.fb_in1k` | 86.6 M |

**Data:** 20 randomly chosen classes of [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch). That is 1,017 images, of which 804 form the test set.

## Results

| Model | Accuracy, 20 classes (95% CI) | GPU ms/image | CPU ms/image | Peak GPU MB |
|---|:---:|---:|---:|---:|
| ResNet-50 | 65.5% (61.1 to 69.8) | 3.18 | 31.7 | 115 |
| **DeiT-Small** | **68.4%** (64.0 to 72.5) | **3.13** | 35.9 | **89** |
| DeiT-Base | 66.4% (62.1 to 70.6) | 6.44 | 114.8 | 339 |

<sub>Timed at batch size 1 on an NVIDIA RTX 5060 Ti and an Intel i5-14400F.</sub>

### Key findings

1. **A bigger transformer did not help.** DeiT-Base costs two to four times more than DeiT-Small and is not more accurate (difference of -2.0 points, 95% CI -5.4 to +1.5).
2. **CNN against ViT is undecided.** DeiT-Small is 2.9 points ahead of ResNet-50, but the interval includes zero (-1.0 to +6.7), so the difference is not established at this sample size.
3. **The dataset contains copies.** 443 of the 1,017 images are flipped or rotated copies of another image. A plain random split would have leaked 185 test images into dev, so the split is made by drawing, not by image.
4. **Confidence is useful but is not a probability.** With abstention, DeiT-Small answers 58.8% of test images and is right on 88.2% of those, against 68.4% without abstaining.

> [!NOTE]
> The full analysis, with all figures, is in [`CV_Assignment2_Report.pdf`](CV_Assignment2_Report.pdf).

## How this is useful

- **A template for judging pretrained models.** To test other models, change the checkpoint ids in the model-loading cell. To test other ImageNet-labelled data, change the dataset cell. The same accuracy, cost and calibration measurements then run on them.
- **Honest comparison of two models.** The bootstrap confidence intervals show whether a gap between two models is real or just noise from a small test set.
- **Duplicate detection.** The split cell finds flipped and rotated copies of the same image and keeps them on one side of the split. This applies to any scraped image dataset.
- **Cost measurement done properly.** The timing code uses warm-up runs and GPU synchronisation, and measures peak memory per model.
- **Abstention.** The last code cell shows how to turn a confidence score into a "not sure" option, with the threshold chosen on dev and checked once on test.

## Setup

### Requirements

- Python 3.10 or newer (developed on Python 3.14)
- About 10 GB of free disk space (the dataset download is about 7 GB)
- A GPU is optional

> [!TIP]
> Without a GPU the notebook still runs, but it reports model weight size in place of peak GPU memory.

### Installation

**1. Get the code**

```bash
git clone https://github.com/tanishq2000/sketch-cnn-vs-vit.git
cd sketch-cnn-vs-vit
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**3. Install the libraries**

For GPU use, install PyTorch first with the command from [pytorch.org](https://pytorch.org).

```bash
pip install torch torchvision timm huggingface_hub numpy scipy scikit-learn pandas matplotlib pillow jupyter
```

**4. Run the notebook**

```bash
jupyter notebook assignment2.ipynb
```

Run all cells from top to bottom. The first run downloads the dataset and the three model checkpoints from the Hugging Face Hub and caches them, so later runs are fast.

<details>
<summary><b>Running on Google Colab</b></summary>

<br>

1. Upload `assignment2.ipynb` to Colab.
2. Select a GPU runtime (Runtime → Change runtime type).
3. Uncomment the `%pip install` line in the first cell.
4. Run all cells.

</details>

## Reproducibility

| Item | Value |
|---|---|
| **Seed** | 42, for the choice of classes, the dev/test split and the bootstrap |
| **Pinned versions** | The dataset and all three checkpoints are pinned to exact commit hashes in the code |
| **Deterministic** | Accuracy figures are the same on every run |
| **Not deterministic** | Timings change slightly between runs and depend on the hardware |

## Repository structure

```text
sketch-cnn-vs-vit/
├── main_sketch-cnn-vs-vit.ipynb            # the full pipeline, with saved outputs
├── report_sketch-cnn-vs-vit.pdf    # four-page report with the analysis and figures
└── README.md
```

## Limitations

- Only 20 classes and 460 unique test drawings, from one random draw of classes.
- The models choose among 20 classes, not 1,000, so these accuracies are not comparable with published ImageNet-Sketch numbers.
- Timings come from one machine.

## Data and licences

- **ImageNet-Sketch** (Wang et al., 2019): the repository is under the MIT licence.
- **Model checkpoints:** from `timm`, under the Apache-2.0 licence.

> [!IMPORTANT]
> The sketch images were scraped from the web and copyright stays with the artists. No images are included in this repository. The notebook downloads them for evaluation only.

## AI assistance

Claude (Anthropic) suggested the dataset and checkpoints, wrote most of the notebook code and drafted the text. I chose among the options, ran and checked everything, and reviewed the analysis and conclusions.

## References

1. Wang et al., *Learning Robust Global Representations by Penalizing Local Predictive Power*, NeurIPS 2019
2. Touvron et al., *Training data-efficient image transformers and distillation through attention*, ICML 2021
3. Wightman et al., *ResNet strikes back*, arXiv:2110.00476, 2021
4. Geirhos et al., *ImageNet-trained CNNs are biased towards texture*, ICLR 2019
5. Naseer et al., *Intriguing Properties of Vision Transformers*, NeurIPS 2021
