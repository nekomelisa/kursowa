from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import (
    SpectralClustering,
    KMeans,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.manifold import MDS
from collections import Counter, defaultdict

from compute_similarity import load_data, compute_p_statistic

sigma = 0.1
g = 3

clustering_methods = {
    'SpectralClustering': SpectralClustering(
        n_clusters=2,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=0
    ),
    'AgglomerativeClustering': AgglomerativeClustering(
        n_clusters=2,
        metric='precomputed',
        linkage='average'
    ),
    'KMeans': KMeans(n_clusters=2, random_state=0),
    'GaussianMixture': GaussianMixture(n_components=2, covariance_type='full', random_state=0)
}

if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE_DIR / "data"
    print(f"Using data directory: {DATA_ROOT}")

    data = load_data(DATA_ROOT)
    channels = sorted({ch for grp in data.values() for ch in next(iter(grp.values())).keys()})

    output_pdf = BASE_DIR / 'all_clusters.pdf'
    with PdfPages(output_pdf) as pdf:
        for ch in channels:
            bc_items = [(pid, data['BC'][pid][ch]) for pid in data['BC'] if ch in data['BC'][pid]]
            ctrl_items = [(pid, data['Control'][pid][ch]) for pid in data['Control'] if ch in data['Control'][pid]]

            bc_list = [arr for pid, arr in bc_items]
            ctrl_list = [arr for pid, arr in ctrl_items]
            n_bc, m_ctrl = len(bc_list), len(ctrl_list)
            group_tags = ['BC'] * n_bc + ['Control'] * m_ctrl

            S_bc = np.zeros((n_bc, n_bc), float)
            S_ctrl = np.zeros((m_ctrl, m_ctrl), float)
            S_cross = np.zeros((n_bc, m_ctrl), float)
            for i in range(n_bc):
                for j in range(n_bc):
                    S_bc[i, j] = compute_p_statistic(bc_list[i], bc_list[j], g=g)
            for i in range(m_ctrl):
                for j in range(m_ctrl):
                    S_ctrl[i, j] = compute_p_statistic(ctrl_list[i], ctrl_list[j], g=g)
            for i in range(n_bc):
                for j in range(m_ctrl):
                    S_cross[i, j] = compute_p_statistic(bc_list[i], ctrl_list[j], g=g)

            top = np.hstack([S_bc, S_cross])
            bottom = np.hstack([S_cross.T, S_ctrl])
            S_all = np.vstack([top, bottom])
            S_all = (S_all + S_all.T) / 2
            d_all = 1.0 - S_all
            affinity_all = np.exp(-(d_all ** 2) / (2 * sigma ** 2))

            mds_model = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
            X_mds = mds_model.fit_transform(d_all)

            print(f"\n===== CHANNEL: {ch} =====")
            for name, algo in clustering_methods.items():
                if name == 'SpectralClustering':
                    labels = algo.fit_predict(affinity_all)
                elif name in ('AgglomerativeClustering',):
                    labels = algo.fit_predict(d_all)
                else:
                    labels = algo.fit_predict(X_mds)

                counts = defaultdict(Counter)
                for lbl, tag in zip(labels, group_tags):
                    counts[lbl][tag] += 1
                print(f"\n{name} — channel={ch}")
                for cid in sorted(counts):
                    print(f" Cluster {cid}: BC={counts[cid]['BC']:3d}, Control={counts[cid]['Control']:3d}")

                if len(counts) == 2:
                    cids = sorted(counts)
                    pos_c = max(cids, key=lambda c: counts[c]['BC'])
                    neg_c = min(cids) if max(cids) == pos_c else [c for c in cids if c != pos_c][0]

                    TP = counts[pos_c]['BC']
                    FN = counts[neg_c]['BC']
                    TN = counts[neg_c]['Control']
                    FP = counts[pos_c]['Control']
                    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
                    specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
                    print(f" Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")


                bc_counts = [counts[c]['BC'] for c in sorted(counts)]
                ctrl_counts = [counts[c]['Control'] for c in sorted(counts)]
                x = np.arange(len(bc_counts))
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(x - 0.2, bc_counts, 0.4, label='BC')
                ax.bar(x + 0.2, ctrl_counts, 0.4, label='Control')
                ax.set_xticks(x)
                ax.set_xticklabels(sorted(counts))
                ax.set_xlabel('Cluster ID')
                ax.set_ylabel('Count')
                ax.set_title(f"{name} — channel={ch}")
                ax.legend()
                plt.tight_layout()

                pdf.savefig(fig)
                plt.close(fig)

    print(f"All cluster plots with sensitivity/specificity saved to {output_pdf}")
