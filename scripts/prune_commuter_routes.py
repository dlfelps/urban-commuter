"""
Prune Commuter Transit Routes Graph

Filters the commuter_transit_routes graph to only include edges with
weight greater than a specified threshold.
"""

import pickle
import networkx as nx
from pathlib import Path

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"

# Pruning threshold
THRESHOLD = 10


def main():
    """Prune commuter routes graph"""
    print("=" * 80)
    print("PRUNING COMMUTER TRANSIT ROUTES GRAPH")
    print("=" * 80)

    # Load commuter routes graph
    routes_path = GRAPHS_DIR / "commuter_transit_routes.pkl"

    if not routes_path.exists():
        print(f"\n[ERROR] Commuter routes file not found: {routes_path}")
        print("Run 'uv run scripts/identify_commuter_routes.py' first")
        return

    print(f"\nLoading {routes_path.name}...")
    with open(routes_path, 'rb') as f:
        G = pickle.load(f)

    print(f"Original graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Calculate statistics before pruning
    weights_before = [data['weight'] for _, _, data in G.edges(data=True)]
    print(f"\nEdge weight statistics (before pruning):")
    print(f"  Min: {min(weights_before):.6f}")
    print(f"  Max: {max(weights_before):.6f}")
    print(f"  Mean: {sum(weights_before) / len(weights_before):.6f}")

    # Prune edges below threshold
    edges_to_remove = []
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if weight <= THRESHOLD:
            edges_to_remove.append((u, v))

    print(f"\nPruning edges with weight <= {THRESHOLD}...")
    print(f"Edges to remove: {len(edges_to_remove):,}")

    G.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    print(f"Isolated nodes after edge removal: {len(isolated):,}")
    G.remove_nodes_from(isolated)

    print(f"\nPruned graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Calculate statistics after pruning
    if G.number_of_edges() > 0:
        weights_after = [data['weight'] for _, _, data in G.edges(data=True)]
        total_weight_before = sum(weights_before)
        total_weight_after = sum(weights_after)

        print(f"\nEdge weight statistics (after pruning):")
        print(f"  Min: {min(weights_after):.6f}")
        print(f"  Max: {max(weights_after):.6f}")
        print(f"  Mean: {sum(weights_after) / len(weights_after):.6f}")

        print(f"\nWeight distribution:")
        print(f"  Total weight before: {total_weight_before:,.0f}")
        print(f"  Total weight after: {total_weight_after:,.0f}")
        print(f"  Flow retained: {(total_weight_after / total_weight_before * 100):.1f}%")

    # Save pruned graph
    output_path = GRAPHS_DIR / f"commuter_transit_routes_pruned_{THRESHOLD}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)

    print(f"\nSaved: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("PRUNING SUMMARY")
    print("=" * 80)
    print(f"Threshold: {THRESHOLD}")
    print(f"Edges removed: {len(edges_to_remove):,}")
    print(f"Nodes removed: {len(isolated):,}")
    print(f"Final graph size: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")


if __name__ == "__main__":
    main()
