import os
# MUST be set before importing numpy, sklearn, or umap
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import *
import joblib
import os
from apre.utils.accelerators import configure_accelerators, set_random_seeds
from apre.data.loader import download_esc50_data, load_metadata, select_classes, prepare_labels
from apre.features.extractor import extract_features_from_metadata
from apre.models.dimensionality_reduction import apply_all_reductions
from apre.clustering.algorithms import apply_all_clustering_methods, map_clusters_to_labels
from apre.models.neural_networks import (create_autoencoder_pipeline, build_dec_model,
                                           train_dec_model, create_cnn_pipeline)
from apre.models.supervised_learning import train_all_supervised_models
from apre.evaluation.metrics import (evaluate_all_clustering_results, print_results_summary,
                                       create_results_summary)
from apre.visualization.plots import save_all_plots

def main():
    print("Starting Audio Pattern Recognition Engine (APRE)")
    print("=" * 50)

    accel = configure_accelerators()
    set_random_seeds(RANDOM_SEED)

    print("\n1. Loading ESC-50 Dataset...")
    audio_dir, meta_path = download_esc50_data(ESC50_ZIP_URL, OUT_DIR)
    meta = load_metadata(meta_path)

    selected_classes, meta_sub = select_classes(meta, CANDIDATE_CLASSES, MIN_SELECTED_CLASSES)

    print("\n2. Extracting Advanced Audio Features...")
    X, labels, filepaths = extract_features_from_metadata(meta_sub, audio_dir)
    y_encoded, le, class_names = prepare_labels(labels)
    
    print("Saving preprocessing artifacts...")
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    joblib.dump(le, "saved_models/label_encoder.joblib")

    print("Feature matrix shape:", X.shape)
    print("Classes:", class_names)

    print("\n3. Applying Dimensionality Reduction...")
    reduction_config = {
        'pca_target': PCA_TARGET,
        'tsne_perplexity': TSNE_PERPLEXITY,
        'tsne_learning_rate': TSNE_LEARNING_RATE,
        'umap_n_neighbors': UMAP_N_NEIGHBORS,
        'umap_min_dist': UMAP_MIN_DIST
    }
    reduction_results = apply_all_reductions(X, y_encoded, reduction_config)
    joblib.dump(reduction_results['scaler'], "saved_models/scaler.joblib")

    print("\n4. Creating Autoencoder Representation...")
    X_encoded = create_autoencoder_pipeline(
        reduction_results['X_scaled'],
        encoding_dim=ENCODING_DIM,
        epochs=AUTOENCODER_EPOCHS
    )

    print("\n5. Applying Clustering Algorithms...")
    clustering_config = {
        'kmeans_n_init': KMEANS_N_INIT,
        'gmm_n_init': GMM_N_INIT,
        'spectral_n_init': SPECTRAL_N_INIT
    }
    clustering_results = apply_all_clustering_methods(
        reduction_results['X_scaled'],
        reduction_results['X_umap'],
        X_encoded,
        len(class_names),
        clustering_config
    )

    print("\n6. Applying Deep Embedded Clustering...")
    dec_model = build_dec_model(reduction_results['X_scaled'].shape[1], len(class_names))
    dec_proba = train_dec_model(
        dec_model,
        reduction_results['X_scaled'],
        clustering_results["K-Means"]["labels"],
        epochs=DEC_EPOCHS
    )
    dec_labels = np.argmax(dec_proba, axis=1)
    clustering_results["DEC"] = {
        "labels": dec_labels,
        "model": dec_model,
        "silhouette": "N/A"
    }

    print("\n7. Training Supervised Learning Models...")
    supervised_config = {
        'test_size': TEST_SIZE,
        'mlp_layers': MLP_LAYERS,
        'mlp_max_iter': MLP_MAX_ITER,
        'xgb_n_estimators': XGB_PARAMS['n_estimators']
    }
    supervised_results, best_xgb_model = train_all_supervised_models(
        reduction_results['X_scaled'], y_encoded, accel, supervised_config
    )
    best_xgb_model.save_model("saved_models/xgboost_model.json")
    print("Models saved successfully!")

    print("\n8. Training CNN Model...")
    cnn_results = create_cnn_pipeline(
        X_encoded, y_encoded,
        test_size=TEST_SIZE,
        epochs=CNN_EPOCHS
    )
    supervised_results["CNN"] = {
        "Accuracy": cnn_results["accuracy"],
        "Precision": cnn_results["precision"],
        "Recall": cnn_results["recall"],
        "F1-Score": cnn_results["f1_score"]
    }

    print("\n9. Evaluating Results...")
    clustering_evaluation = evaluate_all_clustering_results(clustering_results, y_encoded)

    print_results_summary(clustering_evaluation, supervised_results, class_names)

    print("\n10. Creating Visualizations...")
    viz_config = {
        'figure_size': FIGURE_SIZE,
        'dpi': DPI,
        'colormap': COLORMAP,
        'alpha': ALPHA,
        'clustering_viz_file': CLUSTERING_VIZ_FILE,
        'confusion_matrix_file': CONFUSION_MATRIX_FILE,
        'feature_importance_file': FEATURE_IMPORTANCE_FILE
    }
    save_all_plots(
        reduction_results, clustering_results, supervised_results,
        y_encoded, class_names, viz_config
    )

    summary = create_results_summary(clustering_evaluation, supervised_results, class_names)

    print("Analysis completed successfully!")
    print(f"Generated visualizations saved as PNG files in the current directory.")
    print("=" * 50)

    return summary

if __name__ == "__main__":
    import numpy as np
    summary = main()
