"""Build correlation-based adjacency matrices from training flow data."""

import os
import numpy as np


def build_correlation_adj(flow_train: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Build a sparse adjacency matrix based on Pearson correlation.

    Steps:
        1. Dead nodes (std=0) get zero correlation with all others
        2. Pearson correlation on active nodes
        3. Clip negative values to 0
        4. Zero diagonal (no self-loops)
        5. Top-K sparsification per node (deterministic via stable argsort)
        6. Symmetrize: A = max(A, A^T)
    """
    T, N = flow_train.shape
    node_std = flow_train.std(axis=0)
    active_mask = node_std > 1e-8
    n_active = int(active_mask.sum())
    print(f"  Nodes: {N} total, {n_active} active, {N - n_active} dead")

    corr = np.zeros((N, N), dtype=np.float64)
    if n_active > 1:
        active_idx = np.where(active_mask)[0]
        active_flow = flow_train[:, active_idx]
        mean = active_flow.mean(axis=0, keepdims=True)
        std = active_flow.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        normed = (active_flow - mean) / std
        sub_corr = (normed.T @ normed) / T
        np.clip(sub_corr, 0, None, out=sub_corr)
        np.fill_diagonal(sub_corr, 0)
        corr[np.ix_(active_idx, active_idx)] = sub_corr

    # Top-K sparsification: deterministic via stable descending argsort
    adj = np.zeros_like(corr)
    for i in range(N):
        row = corr[i]
        nonzero_count = int((row > 0).sum())
        if nonzero_count == 0:
            continue
        k = min(top_k, nonzero_count)
        top_indices = np.argsort(-row, kind="stable")[:k]
        adj[i, top_indices] = row[top_indices]

    adj = np.maximum(adj, adj.T)

    n_edges = int((adj > 0).sum())
    density = n_edges / (N * N) * 100
    print(f"  Edges: {n_edges}, density: {density:.2f}%")
    return adj


def main():
    from data_preprocess.dataset import load_flow, split_data

    flow_dir = "data/processed_data/traffic_flow"
    save_dir = "data/processed_data/adjacency"
    os.makedirs(save_dir, exist_ok=True)

    start_date, end_date, freq_min = "2014-10-01", "2017-09-30", 60
    train_steps = 15336
    top_k = 10

    for city in ["NYC", "CHI"]:
        print(f"\n=== Building adjacency for {city} ===")
        flow = load_flow(city, flow_dir, start_date, end_date, freq_min)
        T, N = flow.shape
        (train_s, train_e), _, _ = split_data(T, train_steps, 5160)
        flow_train = flow[train_s:train_e].astype(np.float64)

        adj = build_correlation_adj(flow_train, top_k=top_k)

        save_path = os.path.join(save_dir, f"adj_{city}_corr_k{top_k}.npy")
        np.save(save_path, adj)
        print(f"  Saved to {save_path} | shape={adj.shape}")

        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        n_components, labels = connected_components(csr_matrix(adj > 0), directed=False)
        sizes = np.bincount(labels)
        print(f"  Connected components: {n_components} "
              f"(largest: {sizes.max()} nodes, isolated: {(sizes == 1).sum()} nodes)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
