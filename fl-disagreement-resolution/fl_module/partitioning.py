"""Label-skewed (non-IID) data partitioning that uses every sample.

The Dirichlet partition is the standard non-IID benchmark for federated learning
(Hsu et al. 2019): for each class, client proportions are drawn from a symmetric
Dirichlet(alpha). Small alpha -> strong skew; large alpha -> near-IID. Every index
is assigned to exactly one client, so the full dataset is used (matching the IID
setup, where the only difference is how the data is split, not how much).
"""
import numpy as np


def dirichlet_partition(labels, num_clients, alpha=0.5, seed=42, min_per_client=10):
    """Partition sample indices across clients with Dirichlet label skew.

    Args:
        labels: 1-D array of integer class labels, one per sample.
        num_clients: number of clients.
        alpha: Dirichlet concentration; lower = more skew (0.1 strong, 0.5 moderate).
        seed: reproducible draw.
        min_per_client: redraw until every client gets at least this many samples,
            so no client is left empty/degenerate.

    Returns:
        list of np.ndarray, one index array per client (covering all samples, disjoint).
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)

    while True:
        client_idx = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # cut points along this class's samples, proportional to the draw
            cuts = (np.cumsum(proportions)[:-1] * len(idx_c)).astype(int)
            for cid, chunk in enumerate(np.split(idx_c, cuts)):
                client_idx[cid].extend(chunk.tolist())
        sizes = [len(c) for c in client_idx]
        if min(sizes) >= min_per_client:
            break
        seed += 1
        rng = np.random.default_rng(seed)

    return [np.array(sorted(c)) for c in client_idx]
