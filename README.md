# Reliability-Aware Dual-Graph Diffusion for Robust Social Recommendation

This anonymous repository contains the implementation and a representative end-to-end reproducibility pipeline for the proposed method.

##  ⚙️Environment

The primary reproducibility environment is:

- Ubuntu 20.04
- CUDA 11.8
- PyTorch 2.0.0
- torchvision 0.15.1
- torchaudio 2.0.1
- DGL 1.0.2 with CUDA 11.8

Create an isolated environment and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## 📖Datasets

### 🔥Raw data sources

The experiments use processed datasets derived from publicly available raw data. The raw files are not used directly by the model: every experimental dataset is produced by applying the preprocessing procedure.

- **Yelp.** The raw Yelp Open Dataset is available from the[Yelp Open Dataset portal](https://business.yelp.com/data/resources/open-dataset/).The preprocessing pipeline uses the review, user, and business JSON files.
- **Amazon.** The Amazon experimental data are derived from the public[Amazon Reviews 2024 collection](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).

Users must obtain the raw files from the corresponding providers and comply with their licenses and terms of use. All user and item indices in the experimental data are newly generated integer identifiers; the repository does not include raw user profiles or raw review files.

### ☀️Reproducibility scope

The anonymous artifact currently provides **Amazon-beauty example datasets only** as the representative end-to-end reproducibility dataset. Amazon-beauty covers the interaction graph, observed friendship graph, business metadata, and review-text inputs required by the model. The remaining experimental datasets are processed from their respective raw sources using the same general protocol, but they are not included in the current reproducibility package.

The Dataset includes:

```text
datasets/amazon-beauty/dataset.pkl
datasets/amazon-beauty/review_texts.csv
```

## 🚀Model Training

### 🔥Train on Dataset

All Amazon-beauty reproduction hyperparameters are stored as defaults in
[`utils/param.py`](utils/param.py). After preparing and validating the dataset,
the complete training run is therefore launched with the concise command:

```bash
python main.py --dataset amazon-beauty
```

Any value can still be overridden explicitly from the command line. For example:

```bash
python main.py --dataset amazon-beauty --cuda 0
```
