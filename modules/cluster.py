import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.metrics import silhouette_score
import umap

def load_cate_files(file_dict):
    """
    file_dict: dict like {"mrs_binary": "path/to/file.csv", ...}
    """
    cates = {}
    for name, path in file_dict.items():
        df = pd.read_csv(path)
        vec = df['treatment_effect'].values
        if name in ['infarct_dch', 'shunt_180', 'infection_dch', 'vs_clin']:
            vec = -1 * vec  # Invert CATEs where 1 = bad outcome
        cates[name] = vec
    return cates

def generate_cate_matrix(X, cate_file_dict):
    """
    Given raw feature matrix X and dict of cate csvs, return n x m matrix.
    """
    cates = load_cate_files(cate_file_dict)
    # Check alignment
    for name, vec in cates.items():
        assert len(vec) == len(X), f"{name} length mismatch!"
    matrix = np.column_stack([cates[k] for k in sorted(cates.keys())])
    return matrix

def umap_cluster_cates(cate_matrix, n_clusters=4, n_components=3):
    scaler = Normalizer()
    matrix_scaled = scaler.fit_transform(cate_matrix)

    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_umap = reducer.fit_transform(matrix_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_umap)

    score = silhouette_score(X_umap, labels)
    print(f"📏 Silhouette Score (UMAP): {score:.4f}")

    return X_umap, labels

def visualize_clusters(X_embed, labels, method="UMAP"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_embed[:, 0], y=X_embed[:, 1], hue=labels, palette='Set2', s=60)
    plt.title(f"Individual Treatment Effects across Outcomes")
    plt.xlabel(f"{method} 1")
    plt.ylabel(f"{method} 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_clusters_3d_plotly(X_embed, labels, kmeans_model=None, method="UMAP"):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.spatial import ConvexHull
    import seaborn as sns

    fig = go.Figure()
    unique_labels = np.unique(labels)
    palette = sns.color_palette("Set2", n_colors=len(unique_labels)).as_hex()

    for i, label in enumerate(unique_labels):
        cluster_points = X_embed[labels == label]
        color = palette[i]

        # Add cluster points
        fig.add_trace(go.Scatter3d(
            x=cluster_points[:, 0], y=cluster_points[:, 1], z=cluster_points[:, 2],
            mode='markers',
            marker=dict(size=5, color=color),
            name=f"Cluster {label}",
            opacity=0.75
        ))

        # Add centroid
        if kmeans_model:
            centroid = kmeans_model.cluster_centers_[label]
            fig.add_trace(go.Scatter3d(
                x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                mode='markers+text',
                marker=dict(size=8, color='black', symbol='x'),
                text=[f'C{label}'],
                name=f"Centroid {label}",
                textposition="top center"
            ))

        # Add convex hull (if > 3 points)
        if cluster_points.shape[0] >= 4:
            hull = ConvexHull(cluster_points)
            simplices = hull.simplices
            for simplex in simplices:
                tri = cluster_points[simplex]
                fig.add_trace(go.Mesh3d(
                    x=tri[:, 0], y=tri[:, 1], z=tri[:, 2],
                    color=color,
                    opacity=0.2,
                    showscale=False,
                    name=f"Hull {label}",
                    hoverinfo='skip'
                ))

    fig.update_layout(
        title=f"🧠 3D UMAP Clustering of CATEs",
        width=1000,         # 👈 wider canvas
        height=800,         # 👈 taller height
        scene=dict(
            xaxis_title=f"{method} 1",
            yaxis_title=f"{method} 2",
            zaxis_title=f"{method} 3",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        legend=dict(itemsizing='constant'),
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()


def plot_cate_tradeoff(cate_matrix, outcome1, outcome2, clusters=None):
    """
    Scatterplot showing tradeoff between two outcomes' CATEs.
    Optionally color-coded by cluster.
    """
    if outcome1 not in cate_matrix.columns or outcome2 not in cate_matrix.columns:
        raise ValueError("❌ Selected outcomes not in CATE matrix.")

    plt.figure(figsize=(7, 6))
    if clusters is not None:
        sns.scatterplot(
            x=cate_matrix[outcome1], y=cate_matrix[outcome2],
            hue=clusters, palette='Set2', s=60
        )
    else:
        sns.scatterplot(
            x=cate_matrix[outcome1], y=cate_matrix[outcome2],
            color='darkblue', alpha=0.6, s=50
        )

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title(f"Trade-off: {outcome1} vs {outcome2}")
    plt.xlabel(f"CATE for {outcome1}")
    plt.ylabel(f"CATE for {outcome2}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import plotly.graph_objects as go
import seaborn as sns

def plot_patient_overlay_3d(X_embed, labels, patient_point, method="UMAP", kmeans_model=None):
    fig = go.Figure()
    unique_labels = np.unique(labels)
    palette = sns.color_palette("Set2", n_colors=len(unique_labels)).as_hex()

    for i, label in enumerate(unique_labels):
        cluster_points = X_embed[labels == label]
        color = palette[i]

        fig.add_trace(go.Scatter3d(
            x=cluster_points[:, 0], y=cluster_points[:, 1], z=cluster_points[:, 2],
            mode='markers',
            marker=dict(size=4, color=color),
            name=f"Cluster {label}",
            opacity=0.7
        ))

    # Overlay new patient point as 'X'
    fig.add_trace(go.Scatter3d(
        x=[patient_point[0]], y=[patient_point[1]], z=[patient_point[2]],
        mode='markers+text',
        marker=dict(size=8, color='black', symbol='x'),
        text=["You"],
        name="New Patient",
        textposition="top center"
    ))

    fig.update_layout(
        title="🧠 3D UMAP Clustering of CATEs with New Patient Overlay",
        scene=dict(
            xaxis_title=f"{method} 1",
            yaxis_title=f"{method} 2",
            zaxis_title=f"{method} 3"
        ),
        template="plotly_white",
        height=800,
        width=1000,
        showlegend=True
    )
    return fig

