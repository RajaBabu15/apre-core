
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def apply_all_reductions(X, y_encoded, config):
    print("Applying dimensionality reduction...")
    
    results = {}
    
    # Scaling
    print("Standard Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    results['X_scaled'] = X_scaled
    
    # PCA
    target = config.get('pca_target', 0.95)
    print(f"Running PCA (target={target})...")
    pca = PCA(n_components=target)
    X_pca = pca.fit_transform(X_scaled)
    results['X_pca'] = X_pca
    print(f"PCA reduced feature space from {X_scaled.shape[1]} to {X_pca.shape[1]} components")
    
    # t-SNE
    print("Running t-SNE (2D)...")
    tsne = TSNE(
        n_components=2,
        perplexity=config.get('tsne_perplexity', 30),
        learning_rate=config.get('tsne_learning_rate', 200),
        random_state=42,
        init='pca',

    )
    X_tsne = tsne.fit_transform(X_scaled)
    results['X_tsne'] = X_tsne
    
    # UMAP
    print("Running UMAP (2D)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=config.get('umap_n_neighbors', 15),
        min_dist=config.get('umap_min_dist', 0.1),
        random_state=42
    )
    X_umap = reducer.fit_transform(X_scaled)
    results['X_umap'] = X_umap
    
    return results
