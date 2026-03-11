# Skin Cancer Image Classification Research

Binary classification of skin lesions (malignant vs benign) from dermoscopy images using DINOv2 and ResNet models on the MRA-MIDAS Stanford dataset.

### Dataset
Due to licensing restrictions and patient privacy, the data is not included in this repository. Researchers can obtain the dataset directly from the official source:

    Official Download: https://stanfordaimi.azurewebsites.net/datasets/f4c2020f-801a-42dd-a477-a1a8357ef2a5 

    Citation: Chiou, A. S., et al. "Multimodal Image Dataset for AI-based Skin Cancer (MIDAS) Benchmarking." medRxiv (2024). https://doi.org/10.1101/2024.06.27.24309562

## Setup

```bash
git clone <this repo>
cd SkinCancerImageClassificationResearch
pip install uv  # if you don't have it
uv sync
```

Requires Python 3.11+ and a CUDA GPU for training.