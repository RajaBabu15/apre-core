import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..evaluation.metrics import get_confusion_matrix_data

def create_clustering_visualizations(reduction_results, clustering_results, y_encoded,
                                   class_names, config=None):

    if config is None:
        config = {}

    figure_size = config.get('figure_size', (18, 12))
    colormap = config.get('colormap', 'viridis')
    alpha = config.get('alpha', 0.7)
    dpi = config.get('dpi', 300)

    X_pca = reduction_results['X_pca']
    X_tsne = reduction_results['X_tsne']
    X_umap = reduction_results['X_umap']

    best_acc = -1
    best_method = None
    best_labels = None

    for method_name, method_result in clustering_results.items():
        if method_name.startswith('_'):
            continue

        labels = method_result["labels"]

        unique_labels = len(set(labels))
        if unique_labels > 1 and unique_labels > best_acc:
            best_acc = unique_labels
            best_method = method_name
            best_labels = labels

    if best_labels is None:
        best_labels = np.zeros(len(y_encoded))
        best_method = "Default"

    fig, axes = plt.subplots(2, 3, figsize=figure_size)
    fig.suptitle('Audio Signal Clustering Visualizations', fontsize=16)

    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap=colormap, alpha=alpha)
    axes[0, 0].set_title('PCA - True Labels')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[0, 0])

    scatter2 = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap=colormap, alpha=alpha)
    axes[0, 1].set_title('t-SNE - True Labels')
    axes[0, 1].set_xlabel('t-SNE1')
    axes[0, 1].set_ylabel('t-SNE2')
    plt.colorbar(scatter2, ax=axes[0, 1])

    scatter3 = axes[0, 2].scatter(X_umap[:, 0], X_umap[:, 1], c=y_encoded, cmap=colormap, alpha=alpha)
    axes[0, 2].set_title('UMAP - True Labels')
    axes[0, 2].set_xlabel('UMAP1')
    axes[0, 2].set_ylabel('UMAP2')
    plt.colorbar(scatter3, ax=axes[0, 2])

    scatter4 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap=colormap, alpha=alpha)
    axes[1, 0].set_title(f'PCA - {best_method}')
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    plt.colorbar(scatter4, ax=axes[1, 0])

    scatter5 = axes[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=best_labels, cmap=colormap, alpha=alpha)
    axes[1, 1].set_title(f't-SNE - {best_method}')
    axes[1, 1].set_xlabel('t-SNE1')
    axes[1, 1].set_ylabel('t-SNE2')
    plt.colorbar(scatter5, ax=axes[1, 1])

    scatter6 = axes[1, 2].scatter(X_umap[:, 0], X_umap[:, 1], c=best_labels, cmap=colormap, alpha=alpha)
    axes[1, 2].set_title(f'UMAP - {best_method}')
    axes[1, 2].set_xlabel('UMAP1')
    axes[1, 2].set_ylabel('UMAP2')
    plt.colorbar(scatter6, ax=axes[1, 2])

    plt.tight_layout()
    return fig

def create_confusion_matrix_plot(clustering_results, y_encoded, class_names, config=None):

    if config is None:
        config = {}

    dpi = config.get('dpi', 300)

    cm, best_method, mapped_labels = get_confusion_matrix_data(clustering_results, y_encoded)

    if cm is None:

        cm = np.eye(len(class_names))
        best_method = "No Results"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix for {best_method}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()

    return fig

def create_feature_importance_plot(supervised_results, config=None):

    if config is None:
        config = {}

    dpi = config.get('dpi', 300)

    try:
        xgboost_results = supervised_results.get('_models', {}).get('xgboost', {})
        feature_importance = xgboost_results.get('feature_importance')

        if feature_importance is not None:
            top_idx = np.argsort(feature_importance)[-10:][::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_idx)), feature_importance[top_idx])
            ax.set_yticks(range(len(top_idx)))
            ax.set_yticklabels([f'Feature {i}' for i in top_idx])
            ax.set_title('Top 10 Important Features (XGBoost)')
            ax.set_xlabel('Importance Score')
            plt.tight_layout()

            return fig
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

    return None

def save_all_plots(reduction_results, clustering_results, supervised_results,
                  y_encoded, class_names, config=None):

    if config is None:
        config = {}

    print("Creating visualizations...")

    clustering_fig = create_clustering_visualizations(
        reduction_results, clustering_results, y_encoded, class_names, config)
    clustering_file = config.get('clustering_viz_file', 'clustering_visualizations.png')
    clustering_fig.savefig(clustering_file, dpi=config.get('dpi', 300), bbox_inches='tight')
    print(f"Saved clustering visualizations to {clustering_file}")

    cm_fig = create_confusion_matrix_plot(clustering_results, y_encoded, class_names, config)
    cm_file = config.get('confusion_matrix_file', 'confusion_matrix.png')
    cm_fig.savefig(cm_file, dpi=config.get('dpi', 300), bbox_inches='tight')
    print(f"Saved confusion matrix to {cm_file}")

    fi_fig = create_feature_importance_plot(supervised_results, config)
    if fi_fig is not None:
        fi_file = config.get('feature_importance_file', 'feature_importance.png')
        fi_fig.savefig(fi_file, dpi=config.get('dpi', 300), bbox_inches='tight')
        print(f"Saved feature importance plot to {fi_file}")

    try:
        plt.show()
    except Exception:

        pass

    plt.close('all')

def create_performance_comparison_plot(clustering_results, supervised_results):

    all_results = {}

    for method_name, metrics in clustering_results.items():
        if isinstance(metrics.get('Accuracy'), (int, float)):
            all_results[method_name] = metrics['Accuracy']

    for method_name, metrics in supervised_results.items():
        if method_name != '_models' and isinstance(metrics.get('Accuracy'), (int, float)):
            all_results[method_name] = metrics['Accuracy']

    if not all_results:
        return None

    methods = list(all_results.keys())
    accuracies = list(all_results.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(methods, accuracies, alpha=0.7)
    ax.set_title('Performance Comparison Across All Methods')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Method')
    plt.xticks(rotation=45, ha='right')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    return fig
