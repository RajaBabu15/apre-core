# Audio Pattern Recognition Engine (APRE)

A comprehensive audio analysis framework that combines advanced feature extraction, dimensionality reduction, clustering algorithms, and supervised learning for audio pattern recognition.

## Features

- **Advanced Audio Feature Extraction**: MFCC, spectral features, chroma, tonnetz, mel-spectrogram, wavelet features
- **Dimensionality Reduction**: PCA, t-SNE, UMAP, LDA
- **Clustering Algorithms**: K-Means, DBSCAN, Gaussian Mixture Models, Spectral Clustering, HDBSCAN
- **Deep Learning Models**: Autoencoders, CNN, Deep Embedded Clustering (DEC)
- **Supervised Learning**: MLP, XGBoost
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Hardware Acceleration**: GPU support for compatible algorithms

## Installation

### Option 1: Virtual Environment Setup

> **Note**: TensorFlow for macOS requires Python < 3.12. If you are on a newer Python version, consider using `conda` or installing a compatible Python version (e.g., 3.9-3.11).


1. Create and activate a virtual environment:
```bash
python -m venv apre_env
source apre_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Development Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

Run the complete APRE pipeline:

```bash
python main.py
```

### Ouput

```text
===== RESULTS SUMMARY =====
> "Performance benchmarks were conducted on a distinct subset of 5 classes (Dog, Chainsaw, Sea Waves, Clock Alarm, Church Bells) to demonstrate optimal separation capabilities."

Selected classes: ['chainsaw' 'church_bells' 'clock_alarm' 'dog' 'sea_waves']

🎯 CLUSTERING METHODS (Unsupervised):
                    Accuracy     ARI     NMI Silhouette
K-Means                0.545  0.3042  0.4059   0.056511
DBSCAN                 0.000  0.0000  0.0000         -1
GMM                    0.515  0.3001  0.4181        N/A
Spectral               0.755  0.4912  0.5702        N/A
HDBSCAN                0.000  0.0000  0.0000        0.0
UMAP+KMeans            0.815  0.6054  0.6468    0.55511
Autoencoder+KMeans     0.540  0.2795  0.3761    0.19262
DEC                    0.545  0.3042  0.4059        N/A

🤖 SUPERVISED METHODS:
         Accuracy  Precision  Recall  F1-Score
MLP         0.925     0.9278   0.925    0.9249
XGBoost     0.950     0.9600   0.950    0.9511
CNN         0.925     0.9350   0.925    0.9261

📈 BEST PERFORMERS BY CATEGORY:
• Best Clustering Method: UMAP+KMeans (81.5%)
• Best Supervised Method: XGBoost (95.0%)
```

## Configuration

Modify `config/config.py` to customize:

- Audio processing parameters
- Model hyperparameters
- File paths and output settings
- Visualization preferences

## Key Components

### Data Processing
- **ESC-50 Dataset**: Automatic download and preprocessing
- **Feature Extraction**: 280+ audio features per sample
- **Data Scaling**: StandardScaler normalization

### Machine Learning Models
- **Unsupervised**: K-Means, DBSCAN, GMM, Spectral, HDBSCAN
- **Deep Learning**: Autoencoders, CNN, DEC
- **Supervised**: MLP, XGBoost

### Evaluation Metrics
- **Clustering**: Accuracy, ARI, NMI, Silhouette Score
- **Classification**: Accuracy, Precision, Recall, F1-Score

### Visualizations
- Dimensionality reduction plots (PCA, t-SNE, UMAP)
- Clustering results visualization
- Confusion matrices
- Feature importance plots

## Hardware Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Optional but recommended for neural network training
- **Storage**: 2GB+ for dataset and results

## Dependencies

Key libraries:
- `librosa`: Audio processing
- `scikit-learn`: Machine learning algorithms
- `tensorflow`: Deep learning models
- `umap-learn`: UMAP dimensionality reduction
- `hdbscan`: HDBSCAN clustering
- `matplotlib/seaborn`: Visualizations

See `requirements.txt` for complete list.

## Output Files

The system generates:
- `clustering_visualizations.png`: Comprehensive clustering plots
- `confusion_matrix.png`: Best method confusion matrix
- `feature_importance.png`: XGBoost feature importance
- Console output with detailed performance metrics

### Sample Visualizations

#### Clustering Analysis
![Clustering Visualizations](clustering_visualizations.png)
*Comprehensive clustering results showing different algorithms' performance on dimensionally reduced audio features*

#### Classification Performance
![Confusion Matrix](confusion_matrix.png)
*Confusion matrix for the best performing supervised learning method*

#### Feature Analysis
![Feature Importance](feature_importance.png)
*XGBoost feature importance showing the most discriminative audio features*

## Performance Notes

- Uses sequential processing to avoid threading conflicts
- Implements fallback strategies for edge cases
- Optimized for stability over maximum performance
- Automatic parameter adjustment based on data size
