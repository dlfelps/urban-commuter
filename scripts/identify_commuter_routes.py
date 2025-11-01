"""
Identify Commuter Transit Routes

Uses morning and afternoon commute graphs to identify routes that show
distinctive directional patterns between the two time periods.

Algorithm:
1. Convert directed graphs to undirected with signed weights
   (reflects net directional flow between node pairs)
2. Find common edges and multiply their signed values
3. Keep edges where values differ in sign (showing opposite flow directions)
4. Extract square root of absolute values to normalize
"""

import pickle
import networkx as nx
from pathlib import Path
import numpy as np
import math

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"


def node_to_index(node, grid_size=200):
    """Convert (x, y) coordinate to absolute index using x + grid_size*y"""
    x, y = node
    return x + grid_size * y


def index_to_node(index, grid_size=200):
    """Convert absolute index back to (x, y) coordinate"""
    y = index // grid_size
    x = index % grid_size
    return (x, y)


def convert_to_undirected_signed(G, graph_name="Graph"):
    """
    Convert directed graph to undirected graph with signed weights.

    For each pair of nodes (a, b) where a has lower index than b:
    - Get weight from a->b (weight_forward)
    - Get weight from b->a (weight_backward)
    - New edge weight = weight_forward - weight_backward
    (Positive: more flow a->b, Negative: more flow b->a)
    """
    print(f"\nConverting {graph_name} to undirected signed graph...")
    G_undirected = nx.Graph()

    # Collect all node pairs and their edge weights
    edge_pairs = {}

    # First, add all forward edges
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0)
        idx_u = node_to_index(u)
        idx_v = node_to_index(v)

        # Ensure consistent ordering (smaller index first)
        if idx_u < idx_v:
            key = (u, v)
        else:
            key = (v, u)

        if key not in edge_pairs:
            edge_pairs[key] = {'forward': 0, 'backward': 0}

        if idx_u < idx_v:
            edge_pairs[key]['forward'] = weight
        else:
            edge_pairs[key]['backward'] = weight

    # Now create undirected edges with signed weights
    for (u, v), weights in edge_pairs.items():
        idx_u = node_to_index(u)
        idx_v = node_to_index(v)

        # Calculate signed weight: smaller_index -> larger_index minus larger -> smaller
        if idx_u < idx_v:
            signed_weight = weights['forward'] - weights['backward']
        else:
            signed_weight = weights['backward'] - weights['forward']

        G_undirected.add_edge(u, v, weight=signed_weight)

    print(f"  Nodes: {G_undirected.number_of_nodes():,}")
    print(f"  Edges: {G_undirected.number_of_edges():,}")

    # Statistics
    weights = [data['weight'] for _, _, data in G_undirected.edges(data=True)]
    positive_edges = sum(1 for w in weights if w > 0)
    negative_edges = sum(1 for w in weights if w < 0)
    zero_edges = sum(1 for w in weights if w == 0)

    print(f"  Weight distribution:")
    print(f"    Positive (u->v): {positive_edges:,}")
    print(f"    Negative (v->u): {negative_edges:,}")
    print(f"    Zero: {zero_edges:,}")

    return G_undirected


def identify_commuter_routes(G_morning, G_afternoon):
    """
    Identify commuter transit routes by comparing morning and afternoon graphs.

    Algorithm:
    1. Find edges common to both graphs
    2. Multiply their signed weights
    3. Keep edges where product is negative (indicating opposite flow directions)
    4. Convert to positive and take square root
    """
    print("\n" + "="*80)
    print("PHASE 2: IDENTIFYING COMMUTER ROUTES")
    print("="*80)

    # Build commuter route graph
    G_commuter = nx.Graph()

    # Get all edges from morning graph
    morning_edges = set()
    for u, v in G_morning.edges():
        idx_u = node_to_index(u)
        idx_v = node_to_index(v)
        # Canonical form: smaller index first
        edge_key = tuple(sorted([(u, idx_u), (v, idx_v)], key=lambda x: x[1]))
        morning_edges.add((edge_key[0][0], edge_key[1][0]))

    # Get all edges from afternoon graph
    afternoon_edges = set()
    for u, v in G_afternoon.edges():
        idx_u = node_to_index(u)
        idx_v = node_to_index(v)
        # Canonical form: smaller index first
        edge_key = tuple(sorted([(u, idx_u), (v, idx_v)], key=lambda x: x[1]))
        afternoon_edges.add((edge_key[0][0], edge_key[1][0]))

    # Find common edges
    common_edges = morning_edges & afternoon_edges
    print(f"\nCommon edges between morning and afternoon: {len(common_edges):,}")

    # Process common edges
    negative_product_edges = 0
    total_weight = 0

    for u, v in common_edges:
        # Get signed weights from both graphs
        weight_morning = G_morning[u][v]['weight']
        weight_afternoon = G_afternoon[u][v]['weight']

        # Multiply the values
        product = weight_morning * weight_afternoon

        # Keep only negative products (opposite flow directions)
        if product < 0:
            negative_product_edges += 1
            # Convert to positive and take square root
            abs_weight = abs(product)
            sqrt_weight = math.sqrt(abs_weight)

            G_commuter.add_edge(u, v, weight=sqrt_weight)
            total_weight += sqrt_weight

    print(f"Edges with negative product (opposite flows): {negative_product_edges:,}")
    print(f"Commuter route edges identified: {G_commuter.number_of_edges():,}")
    print(f"Total commuter route weight: {total_weight:,.0f}")

    if G_commuter.number_of_edges() > 0:
        avg_weight = total_weight / G_commuter.number_of_edges()
        print(f"Average route weight: {avg_weight:,.1f}")

        # Weight statistics
        weights = [data['weight'] for _, _, data in G_commuter.edges(data=True)]
        print(f"Weight stats:")
        print(f"  Min: {min(weights):,.1f}")
        print(f"  Max: {max(weights):,.1f}")
        print(f"  Mean: {np.mean(weights):,.1f}")
        print(f"  Median: {np.median(weights):,.1f}")
        print(f"  Std Dev: {np.std(weights):,.1f}")

    return G_commuter


def main():
    """Run full commuter route identification algorithm"""
    print("="*80)
    print("COMMUTER TRANSIT ROUTE IDENTIFICATION")
    print("="*80)

    # Load input graphs
    print("\nLoading commute graphs...")
    morning_path = GRAPHS_DIR / "morning_commute_weekday_v2.pkl"
    afternoon_path = GRAPHS_DIR / "afternoon_commute_weekday_v2.pkl"

    if not morning_path.exists() or not afternoon_path.exists():
        print(f"[ERROR] Input graphs not found:")
        print(f"  Morning: {morning_path}")
        print(f"  Afternoon: {afternoon_path}")
        return

    with open(morning_path, 'rb') as f:
        G_morning_directed = pickle.load(f)
    with open(afternoon_path, 'rb') as f:
        G_afternoon_directed = pickle.load(f)

    print(f"  Morning (directed): {G_morning_directed.number_of_nodes():,} nodes, "
          f"{G_morning_directed.number_of_edges():,} edges")
    print(f"  Afternoon (directed): {G_afternoon_directed.number_of_nodes():,} nodes, "
          f"{G_afternoon_directed.number_of_edges():,} edges")

    # Phase 1: Convert to undirected with signed weights
    print("\n" + "="*80)
    print("PHASE 1: CONVERTING TO UNDIRECTED SIGNED GRAPHS")
    print("="*80)

    G_morning_signed = convert_to_undirected_signed(G_morning_directed, "Morning Commute")
    G_afternoon_signed = convert_to_undirected_signed(G_afternoon_directed, "Afternoon Commute")

    # Save phase 1 graphs
    morning_phase1_path = GRAPHS_DIR / "morning_transit_phase1.pkl"
    afternoon_phase1_path = GRAPHS_DIR / "afternoon_transit_phase1.pkl"

    with open(morning_phase1_path, 'wb') as f:
        pickle.dump(G_morning_signed, f)
    print(f"\nSaved: {morning_phase1_path}")

    with open(afternoon_phase1_path, 'wb') as f:
        pickle.dump(G_afternoon_signed, f)
    print(f"Saved: {afternoon_phase1_path}")

    # Phase 2: Identify commuter routes
    G_commuter = identify_commuter_routes(G_morning_signed, G_afternoon_signed)

    # Save commuter routes
    commuter_path = GRAPHS_DIR / "commuter_transit_routes.pkl"
    with open(commuter_path, 'wb') as f:
        pickle.dump(G_commuter, f)
    print(f"\nSaved: {commuter_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nPhase 1 outputs (undirected signed graphs):")
    print(f"  - {morning_phase1_path.name}")
    print(f"  - {afternoon_phase1_path.name}")
    print(f"\nPhase 2 output (commuter transit routes):")
    print(f"  - {commuter_path.name}")
    print(f"\nRoute statistics:")
    print(f"  Nodes: {G_commuter.number_of_nodes():,}")
    print(f"  Edges: {G_commuter.number_of_edges():,}")
    print(f"\nThese edges represent transit routes with asymmetric morning/afternoon flows")
    print("(likely showing directional commute patterns).")


if __name__ == "__main__":
    main()
