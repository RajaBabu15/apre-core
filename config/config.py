import os

ESC50_ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
OUT_DIR = "esc50_data"
REPO_ROOT = os.path.join(OUT_DIR, "ESC-50-master")
AUDIO_DIR = os.path.join(REPO_ROOT, "audio")
META_PATH = os.path.join(REPO_ROOT, "meta", "esc50.csv")

SAMPLE_RATE = 22050
N_MFCC = 20
TARGET_SECONDS = 5

EXPECTED_FEATURE_SIZE = 282

RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
AUTOENCODER_EPOCHS = 50
CNN_EPOCHS = 30
DEC_EPOCHS = 30
ENCODING_DIM = 32

OS_ENV_SETTINGS = {
    "OMP_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1"
}

CANDIDATE_CLASSES = ["glass", "door", "plate", "clock", "keyboard", "mouse", "water", "fire"]
MIN_SELECTED_CLASSES = 5

PCA_TARGET = 0.95
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = 200
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

KMEANS_N_INIT = 20
DBSCAN_PARAMS = {
    'eps': [0.3, 0.5, 0.7, 1.0],
    'min_samples': [3, 5, 7]
}
SPECTRAL_N_INIT = 10
GMM_N_INIT = 5

AUTOENCODER_LAYERS = [512, 256, 128]
CNN_FILTERS = [64, 128, 256]
CNN_KERNEL_SIZE = 3
DENSE_LAYERS = [256, 128, 64]
MLP_LAYERS = (256, 128, 64)
MLP_MAX_ITER = 500

XGB_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,
    "eval_metric": "mlogloss"
}

FIGURE_SIZE = (18, 12)
DPI = 300
COLORMAP = 'viridis'
ALPHA = 0.7

CLUSTERING_VIZ_FILE = 'clustering_visualizations.png'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png'
FEATURE_IMPORTANCE_FILE = 'feature_importance.png'

N_JOBS = 1
