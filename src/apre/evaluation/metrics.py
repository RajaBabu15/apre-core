import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, adjusted_rand_score,
                           normalized_mutual_info_score, precision_score, recall_score, f1_score)
from scipy.stats import mode
from ..clustering.algorithms import map_clusters_to_labels

def evaluate_clustering(true_labels, pred_labels, algorithm_name):

    mask = pred_labels != -1
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0

    filtered_true = true_labels[mask]
    filtered_pred = pred_labels[mask]

    acc = accuracy_score(filtered_true, filtered_pred)
    ari = adjusted_rand_score(filtered_true, filtered_pred)
    nmi = normalized_mutual_info_score(filtered_true, filtered_pred)

    return acc, ari, nmi

def evaluate_all_clustering_results(clustering_results, y_encoded):

    results = {}

    for method_name, method_result in clustering_results.items():
        if method_name.startswith('_'):
            continue

        labels = method_result["labels"]
        mapped_labels = map_clusters_to_labels(y_encoded, labels)

        acc, ari, nmi = evaluate_clustering(y_encoded, mapped_labels, method_name)

        results[method_name] = {
            "Accuracy": acc,
            "ARI": ari,
            "NMI": nmi,
            "Silhouette": method_result.get("silhouette", "N/A")
        }

    return results

def find_best_performers(clustering_results, supervised_results):

    best_performers = {}

    clustering_accuracies = [(k, v['Accuracy']) for k, v in clustering_results.items()
                           if isinstance(v['Accuracy'], (int, float))]
    if clustering_accuracies:
        best_clustering = max(clustering_accuracies, key=lambda x: x[1])
        best_performers['best_clustering'] = {
            'method': best_clustering[0],
            'accuracy': best_clustering[1]
        }

    supervised_accuracies = [(k, v['Accuracy']) for k, v in supervised_results.items()
                           if k != '_models' and isinstance(v['Accuracy'], (int, float))]
    if supervised_accuracies:
        best_supervised = max(supervised_accuracies, key=lambda x: x[1])
        best_performers['best_supervised'] = {
            'method': best_supervised[0],
            'accuracy': best_supervised[1]
        }

    return best_performers

def create_results_summary(clustering_results, supervised_results, class_names):

    summary = {
        'selected_classes': class_names.tolist(),
        'clustering_results': clustering_results,
        'supervised_results': {k: v for k, v in supervised_results.items() if k != '_models'},
        'best_performers': find_best_performers(clustering_results, supervised_results)
    }

    return summary

def print_results_summary(clustering_results, supervised_results, class_names):

    print("\n===== RESULTS SUMMARY =====")
    print(f"Selected classes: {class_names}\n")
    print("📊 Metric Explanations:")
    print("• Accuracy: Classification accuracy (higher = better)")
    print("• ARI: Adjusted Rand Index for clustering (higher = better, max=1)")
    print("• NMI: Normalized Mutual Information for clustering (higher = better, max=1)")
    print("• Silhouette: Cluster quality measure (higher = better, range=-1 to 1)")
    print("• N/A: Metric not applicable to this algorithm type\n")

    print("🎯 CLUSTERING METHODS (Unsupervised):")
    clustering_methods = ['K-Means', 'DBSCAN', 'GMM', 'Spectral', 'HDBSCAN',
                         'UMAP+KMeans', 'Autoencoder+KMeans', 'DEC']
    clustering_filtered = {k: v for k, v in clustering_results.items() if k in clustering_methods}
    clustering_df = pd.DataFrame.from_dict(clustering_filtered, orient='index')
    print(clustering_df.round(4))

    print("\n🤖 SUPERVISED METHODS:")
    supervised_methods = ['MLP', 'XGBoost', 'CNN']
    supervised_filtered = {k: v for k, v in supervised_results.items()
                          if k in supervised_methods and k != '_models'}
    supervised_df = pd.DataFrame.from_dict(supervised_filtered, orient='index')
    print(supervised_df.round(4))

    print("\n📈 BEST PERFORMERS BY CATEGORY:")
    best_performers = find_best_performers(clustering_results, supervised_results)

    if 'best_clustering' in best_performers:
        best_clustering = best_performers['best_clustering']
        print(f"• Best Clustering Method: {best_clustering['method']} ({best_clustering['accuracy']:.1%})")

    if 'best_supervised' in best_performers:
        best_supervised = best_performers['best_supervised']
        print(f"• Best Supervised Method: {best_supervised['method']} ({best_supervised['accuracy']:.1%})")

    print("\n==========================\n")

def get_confusion_matrix_data(clustering_results, y_encoded):

    best_acc = -1
    best_method = None
    best_labels = None

    for method_name, method_result in clustering_results.items():
        if method_name.startswith('_'):
            continue

        labels = method_result["labels"]
        mapped_labels = map_clusters_to_labels(y_encoded, labels)
        acc, _, _ = evaluate_clustering(y_encoded, mapped_labels, method_name)

        if acc > best_acc:
            best_acc = acc
            best_method = method_name
            best_labels = mapped_labels

    if best_labels is not None:
        cm = confusion_matrix(y_encoded, best_labels)
        return cm, best_method, best_labels

    return None, None, None

def calculate_method_statistics(results):

    accuracies = []
    for method_name, metrics in results.items():
        if method_name.startswith('_'):
            continue
        if 'Accuracy' in metrics and isinstance(metrics['Accuracy'], (int, float)):
            accuracies.append(metrics['Accuracy'])

    if accuracies:
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'num_methods': len(accuracies)
        }

    return {
        'mean_accuracy': 0.0,
        'std_accuracy': 0.0,
        'min_accuracy': 0.0,
        'max_accuracy': 0.0,
        'num_methods': 0
    }
