# Audio Spoofing Detection with Quantum & Classical Models

This repository provides a complete pipeline for **audio spoofing detection** using both **classical machine learning** and **quantum machine learning (QML)** approaches. It includes dataset preparation, embedding extraction using pretrained models, and evaluation using **SVM**, **MLP**, and **Quantum Support Vector Machines (QSVM)**.

---

## Overview

The **Quantum project** is designed to:

* Download and preprocess multiple audio spoofing datasets
* Extract high-dimensional embeddings using pretrained models (e.g., Wav2Vec2)
* Train and evaluate classical and quantum models on a shared feature space
* Provide detailed evaluation metrics and visualizations

---

## Project Structure

```
quantum/
│
├── dataset/                      # Dataset downloading and preparation scripts
│   └── dataset_05.py            # Downloads ASVspoof5 dataset from Hugging Face
│
├── embedding/                   # Audio feature extraction scripts
│   ├── embedding_asv19.py       # ASVspoof 2019 LA dataset embedding
│   ├── embedding_inthewild.py   # "In The Wild" dataset processing
│   ├── embedding_o5.py          # ASVspoof5 FLAC embedding extraction
│   └── embedd_add.py            # ADD2023 dataset embedding extraction
│
├── src/                         # Core modeling and evaluation logic
│   ├── run.py                   # Main training & evaluation (5-fold CV)
│   └── fpr.py                   # Cross-corpus evaluation & detailed metrics
│
└── README.md
```

---

##  Getting Started

### Data Preparation

Download and prepare datasets. Example for ASVspoof5:

```bash
python quantum/dataset/dataset_05.py
```

---

### Feature Extraction

Convert audio files into `.npz` embeddings using a pretrained Wav2Vec2 model:

```bash
python quantum/embedding/embedding_o5.py \
    --audio_dir /path/to/flac \
    --out_npz asv5_features.npz
```

---

###  Model Evaluation

Run experiments comparing classical and quantum models:

```bash
python quantum/src/run.py \
    --in_npz asv5_features.npz \
    --shared_dim 10 \
    --use_qsvm \
    --save_plot results.png
```

This pipeline includes:

* PCA dimensionality reduction
* 5-fold cross-validation
* Performance visualization

---

##  Key Features

### Quantum Machine Learning

* Uses **Qiskit** to implement QSVM (QSVC)
* Supports multiple feature maps:

  * Z Feature Map
  * ZZ Feature Map
  * Pauli Feature Map

###  Fair Model Comparison

* All models (SVM, MLP, QSVM) use the **same PCA-reduced feature space**

### Advanced Metrics

* Equal Error Rate (EER)
* Expected Calibration Error (ECE)
* AUC (Area Under Curve)
* TPR @ specific FPR thresholds

### Visualization

* ROC curves
* DET curves
* Calibration plots
* Score distribution histograms

###  Diagnostics

* Kernel diagnostics for QSVM
* Detects issues like **kernel collapse**

---

## Requirements

### Quantum Libraries

* qiskit
* qiskit-machine-learning

### Audio Processing

* torchaudio
* librosa
* soundfile

### Machine Learning

* scikit-learn
* torch
* transformers

### General

* numpy
* matplotlib
* seaborn
* pandas

---

##  Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install qiskit qiskit-machine-learning \
            torchaudio librosa soundfile \
            scikit-learn torch transformers \
            numpy matplotlib seaborn pandas
```

---

##  Example Workflow

```bash
# Step 1: Download dataset
python quantum/dataset/dataset_05.py

# Step 2: Extract embeddings
python quantum/embedding/embedding_o5.py \
    --audio_dir data/flac \
    --out_npz features.npz

# Step 3: Train & evaluate
python quantum/src/run.py \
    --in_npz features.npz \
    --shared_dim 10 \
    --use_qsvm \
    --save_plot results.png
```

---

##  Notes

* Use GPU for faster embedding extraction if available
* QSVM can be computationally expensive for high dimensions
* Recommended `shared_dim`: **8–12** for quantum models

---

##  License

  MIT 

---

##  Contributions

Contributions are welcome! You can:

* Open issues
* Submit pull requests
* Suggest improvements

---
 
---

##  Acknowledgements

* ASVspoof Challenges (2019, 2025)
* Hugging Face datasets
* Qiskit community
* PyTorch & Transformers ecosystem

---
