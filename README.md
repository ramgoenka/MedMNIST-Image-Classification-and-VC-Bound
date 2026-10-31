# MedMNIST Image Classification & VC Bound Analysis

**Benchmarking classical machine learning and deep learning for medical image classification, with an analysis of model generalization.**

A final project for **ECE 2372: Pattern Recognition** at the University of Pittsburgh.

## Overview

This project compares traditional machine-learning classifiers and deep-learning architectures on two datasets from [MedMNIST](https://medmnist.com/):

- **PneumoniaMNIST:** Binary classification of chest X-ray images (normal vs. pneumonia).
- **DermaMNIST:** Seven-class classification of dermatoscopic skin-lesion images.

The goal is to understand how feature representation, dimensionality reduction, model complexity, and hyperparameter selection affect classification performance and generalization.

## Key Results

| Dataset | Model | Test Accuracy |
| --- | --- | ---: |
| PneumoniaMNIST | SVM with RBF kernel + PCA | **86.06%** |
| PneumoniaMNIST | Custom CNN | 85.58% |
| PneumoniaMNIST | Fine-tuned ResNet18 | 85.42% |
| DermaMNIST | Random Forest + PCA | 70.42% |
| DermaMNIST | Custom CNN | 74.91% |
| DermaMNIST | Fine-tuned ResNet18 | **77.81%** |

**Takeaways:** An RBF SVM achieved the highest reported test accuracy on PneumoniaMNIST, while the fine-tuned ResNet18 performed best on DermaMNIST. The results illustrate that more complex models do not automatically outperform classical methods on every dataset.

*Results are from the saved notebook outputs. Model configurations were selected using validation performance; the percentages above are test-set accuracies, not validation accuracies.*

## Methods

### 1. Data exploration and preprocessing

- Loaded the official training, validation, and test splits for both MedMNIST datasets.
- Explored sample images and class distributions.
- Standardized flattened pixel features and applied **Principal Component Analysis (PCA)** for dimensionality reduction.
- Investigated handcrafted image features, including **Histogram of Oriented Gradients (HOG)** and **Local Binary Patterns (LBP)**.

### 2. Classical machine learning

Evaluated multiple classifiers, including:

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- k-Nearest Neighbors (kNN)
- Support Vector Machines (SVM, RBF kernel)
- Random Forest
- AdaBoost

Compared model performance across PCA dimensions and hyperparameter settings, selecting configurations by validation accuracy.

### 3. Deep learning

Trained and evaluated:

- **Custom CNN:** A convolutional neural network with batch normalization, pooling, and dropout.
- **ResNet18:** A fine-tuned residual network using image resizing and normalization appropriate for the pretrained architecture.

Used PyTorch to train the models and compared classification accuracy, ROC-AUC, F1-score, confusion matrices, and train/test generalization gaps.

### 4. VC-bound and generalization analysis

Examined the relationship between **VC dimension**, training-set size, and generalization. For linear classifiers after PCA, the analysis uses a VC-dimension estimate of `d + 1`, where `d` is the number of retained principal components.

The notebooks compare theoretical generalization bounds against observed train/test accuracy gaps for Logistic Regression and LDA, including cases where the bounds become vacuous (greater than 1).

## Repository Structure

```text
.
├── Classifiers.ipynb               # Classical ML, PCA, feature engineering, VC bounds
├── DeepLearning_benchmarking.ipynb # CNN and ResNet18 benchmarking
└── README.md
```

## Tech Stack

**Python**, NumPy, Pandas, Matplotlib, Scikit-learn, Scikit-image, PyTorch, Torchvision, MedMNIST, Jupyter Notebook / Google Colab.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/ramgoenka/MedMNIST-Image-Classification-and-VC-Bound.git
   cd MedMNIST-Image-Classification-and-VC-Bound
   ```

2. Install dependencies in a Python environment:

   ```bash
   python -m pip install numpy pandas matplotlib scikit-learn scikit-image torch torchvision medmnist notebook
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Run **`Classifiers.ipynb`** for classical ML experiments and VC-bound analysis, or **`DeepLearning_benchmarking.ipynb`** for the CNN and ResNet18 experiments. The notebooks download MedMNIST data as needed.

**Note:** Deep-learning training is computationally intensive; a GPU-enabled environment (such as Google Colab) is recommended. Full classical-model grid searches can also take time.

## Limitations

- **Class imbalance:** Both datasets have uneven class distributions, so accuracy alone does not fully characterize performance.
- **Generalization:** The observed train/test gaps are empirical; VC bounds are theoretical and may be loose or vacuous.
- **Reproducibility:** The results shown above come from the notebook's saved runs and may vary with random initialization, software versions, or hardware.
- **Intended use:** This is an educational model-comparison project, **not a clinical diagnostic tool**.

## References

- [MedMNIST dataset and benchmark](https://medmnist.com/)
- [MedMNIST GitHub repository](https://github.com/MedMNIST/MedMNIST)
