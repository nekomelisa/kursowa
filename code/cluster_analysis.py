import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix

def print_and_plot(labels, group_tags, method_name):
    # Count BC/Control in each cluster
    counts = defaultdict(Counter)
    for lbl, tag in zip(labels, group_tags):
        counts[lbl][tag] += 1

    print(f'\n=== {method_name} ===')
    for cid in sorted(counts):
        bc_c, ctrl_c = counts[cid]['BC'], counts[cid]['Control']
        print(f'Cluster {cid}:  BC={bc_c:3d},  Control={ctrl_c:3d}')

    # true labels
    y_true = np.array([1 if tag=='BC' else 0 for tag in group_tags])

    # try both possible mappings of cluster IDs → {0,1}
    best_mapping = None
    best_bal_acc = -1
    best_metrics = None

    for mapping in [{0:0,1:1}, {0:1,1:0}]:
        y_pred = np.array([mapping[l] for l in labels])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp/(tp+fn) if (tp+fn)>0 else 0
        specificity = tn/(tn+fp) if (tn+fp)>0 else 0
        bal_acc = 0.5*(sensitivity + specificity)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_mapping = mapping
            best_metrics = (sensitivity, specificity)

    sensitivity, specificity = best_metrics
    print(f'Chosen mapping: cluster→BC for those mapped to 1: { [c for c,m in best_mapping.items() if m==1] }')
    print(f'Sensitivity (TPR): {sensitivity:.3f}')
    print(f'Specificity (TNR): {specificity:.3f}')

    # plot
    clusters   = sorted(counts)
    bc_counts   = [counts[c]['BC']    for c in clusters]
    ctrl_counts = [counts[c]['Control'] for c in clusters]

    x = np.arange(len(clusters))
    w = 0.4
    plt.figure(figsize=(6,4))
    plt.bar(x - w/2, bc_counts,   w, label='BC')
    plt.bar(x + w/2, ctrl_counts, w, label='Control')
    plt.xticks(x, clusters)
    plt.xlabel('Cluster ID')
    plt.ylabel('Count')
    plt.title(f'{method_name}\nSens={sensitivity:.2f}, Spec={specificity:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    with open('sim_matrices_3.pkl','rb') as f:
        data = pickle.load(f)

    S_bc, S_ctrl, S_cross = data['S_bc'], data['S_ctrl'], data['S_cross']
    bc_labels, ctrl_labels = data['bc_labels'], data['ctrl_labels']

    # build distance & affinity
    S_all = np.vstack([
        np.hstack([S_bc,    S_cross]),
        np.hstack([S_cross.T, S_ctrl])
    ])
    d_all = 1 - S_all
    d_all = (d_all + d_all.T)/2
    sigma = 0.1
    affinity = np.exp(-(d_all**2)/(2*sigma**2))

    group_tags = ['BC']*len(bc_labels) + ['Control']*len(ctrl_labels)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    X_mds = mds.fit_transform(d_all)

    clustering_methods = {
        'SpectralClustering (σ=0.1)': SpectralClustering(2, affinity='precomputed',
                                                        assign_labels='kmeans',
                                                        random_state=0),
        'AgglomerativeClustering (avg linkage)': AgglomerativeClustering(
            n_clusters=2, metric='precomputed', linkage='average'
        ),
        'KMeans (k=2)': KMeans(n_clusters=2, random_state=0),
        'GaussianMixture (n=2)': GaussianMixture(n_components=2,
                                                 covariance_type='full',
                                                 random_state=0)
    }

    for name, algo in clustering_methods.items():
        if 'SpectralClustering' in name:
            labels = algo.fit_predict(affinity)
        elif 'AgglomerativeClustering' in name:
            labels = algo.fit_predict(d_all)
        else:
            labels = algo.fit_predict(X_mds)
        print_and_plot(labels, group_tags, name)