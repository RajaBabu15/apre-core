import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import mode
import hdbscan

def map_clusters_to_labels(true, clusters):

    mapped = np.zeros_like(clusters)
    for c in np.unique(clusters):
        mask = clusters == c
        if c == -1:
            mapped[mask] = -1
            continue
        if true[mask].size == 0:
            continue
        mode_result = mode(true[mask])
        mapped_label = int(mode_result.mode.item())
        mapped[mask] = mapped_label
    return mapped

def apply_kmeans(X, n_clusters, n_init=20, random_state=42):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    return labels, kmeans, sil_score

def apply_dbscan(X, eps=0.5, min_samples=5, param_tuning=True):

    if param_tuning:
        dbscan_params = {'eps': [0.3, 0.5, 0.7, 1.0], 'min_samples': [3, 5, 7]}
        best_sil = -1
        best_dbscan = None

        for eps_val in dbscan_params['eps']:
            for min_samples_val in dbscan_params['min_samples']:
                dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                labels = dbscan.fit_predict(X)
                if len(set(labels)) > 1:
                    try:
                        sil_score = silhouette_score(X, labels)
                    except Exception:
                        sil_score = -1
                    if sil_score > best_sil:
                        best_sil = sil_score
                        best_dbscan = dbscan

        if best_dbscan:
            labels = best_dbscan.fit_predict(X)
            return labels, best_dbscan, best_sil

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    sil_score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    return labels, dbscan, sil_score

def apply_gaussian_mixture(X, n_components, n_init=5, random_state=42):

    gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=n_init)
    labels = gmm.fit_predict(X)
    return labels, gmm

def apply_spectral_clustering(X, n_clusters, n_init=10, random_state=42):

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                 random_state=random_state, n_init=n_init)
    labels = spectral.fit_predict(X)
    return labels, spectral

def apply_hdbscan(X, min_cluster_size=None, gen_min_span_tree=True):

    if min_cluster_size is None:
        min_cluster_size = max(2, X.shape[0] // 10)

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=gen_min_span_tree)
    labels = hdb.fit_predict(X)
    validity_score = getattr(hdb, "relative_validity_", None)
    return labels, hdb, validity_score

def apply_umap_kmeans(X_umap, n_clusters, n_init=20, random_state=42):

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        labels = kmeans.fit_predict(X_umap)
        sil_score = silhouette_score(X_umap, labels) if len(set(labels)) > 1 else -1
        return labels, kmeans, sil_score
    except Exception as e:
        print(f"UMAP+KMeans clustering failed: {e}")

        labels = np.random.randint(0, n_clusters, X_umap.shape[0])
        return labels, None, "Fallback"

def apply_autoencoder_kmeans(X_encoded, n_clusters, n_init=20, random_state=42):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X_encoded)
    sil_score = silhouette_score(X_encoded, labels) if len(set(labels)) > 1 else -1
    return labels, kmeans, sil_score

def apply_all_clustering_methods(X_scaled, X_umap, X_encoded, n_clusters, config=None):

    if config is None:
        config = {}

    print("Clustering with multiple algorithms...")
    results = {}

    kmeans_labels, kmeans_model, kmeans_sil = apply_kmeans(
        X_scaled, n_clusters,
        n_init=config.get('kmeans_n_init', 20)
    )
    results["K-Means"] = {
        "labels": kmeans_labels,
        "model": kmeans_model,
        "silhouette": kmeans_sil
    }

    dbscan_labels, dbscan_model, dbscan_sil = apply_dbscan(X_scaled)
    results["DBSCAN"] = {
        "labels": dbscan_labels,
        "model": dbscan_model,
        "silhouette": dbscan_sil
    }

    gmm_labels, gmm_model = apply_gaussian_mixture(
        X_scaled, n_clusters,
        n_init=config.get('gmm_n_init', 5)
    )
    results["GMM"] = {
        "labels": gmm_labels,
        "model": gmm_model,
        "silhouette": "N/A"
    }

    spec_labels, spec_model = apply_spectral_clustering(
        X_scaled, n_clusters,
        n_init=config.get('spectral_n_init', 10)
    )
    results["Spectral"] = {
        "labels": spec_labels,
        "model": spec_model,
        "silhouette": "N/A"
    }

    hdb_labels, hdb_model, hdb_validity = apply_hdbscan(
        X_scaled,
        min_cluster_size=max(2, n_clusters)
    )
    results["HDBSCAN"] = {
        "labels": hdb_labels,
        "model": hdb_model,
        "silhouette": hdb_validity
    }

    umap_labels, umap_model, umap_sil = apply_umap_kmeans(X_umap, n_clusters)
    results["UMAP+KMeans"] = {
        "labels": umap_labels,
        "model": umap_model,
        "silhouette": umap_sil
    }

    ae_labels, ae_model, ae_sil = apply_autoencoder_kmeans(X_encoded, n_clusters)
    results["Autoencoder+KMeans"] = {
        "labels": ae_labels,
        "model": ae_model,
        "silhouette": ae_sil
    }

    return results
