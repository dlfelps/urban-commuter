"""
Prune Graph Edges by Flow Threshold

Removes edges from commute graphs with weight below a specified threshold.
Saves pruned graphs with new filenames indicating the threshold used.

Usage:
    uv run scripts/prune_graphs.py                 # Use default threshold of 20
    uv run scripts/prune_graphs.py 50              # Use threshold of 50
    uv run scripts/prune_graphs.py 100             # Use threshold of 100
"""

import pickle
import networkx as nx
from pathlib import Path
from typing import Optional
import sys

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
DEFAULT_THRESHOLD = 20


def prune_graph(
    graph: nx.DiGraph,
    threshold: int,
    graph_name: str = "graph"
) -> nx.DiGraph:
    """
    Prune edges from a directed graph keeping only edges with weight >= threshold.

    Args:
        graph: NetworkX directed graph with 'weight' attribute on edges
        threshold: Minimum edge weight to keep
        graph_name: Name of graph for logging

    Returns:
        Pruned graph (modifies and returns original)
    """
    # Count edges before pruning
    edges_before = graph.number_of_edges()
    nodes_before = graph.number_of_nodes()
    flow_before = sum(data.get('weight', 0) for _, _, data in graph.edges(data=True))

    # Identify edges to remove
    edges_to_remove = []
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 0)
        if weight < threshold:
            edges_to_remove.append((u, v))

    # Remove edges
    graph.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)

    # Count edges after pruning
    edges_after = graph.number_of_edges()
    nodes_after = graph.number_of_nodes()
    flow_after = sum(data.get('weight', 0) for _, _, data in graph.edges(data=True))

    # Print statistics
    edges_removed = edges_before - edges_after
    nodes_removed = nodes_before - nodes_after
    flow_removed = flow_before - flow_after
    flow_retained_pct = (flow_after / flow_before * 100) if flow_before > 0 else 0

    print(f"\n{graph_name}:")
    print(f"  Edges: {edges_before:,} -> {edges_after:,} ({edges_removed:,} removed)")
    print(f"  Nodes: {nodes_before:,} -> {nodes_after:,} ({nodes_removed:,} isolated nodes removed)")
    print(f"  Total Flow: {flow_before:,} -> {flow_after:,} ({flow_removed:,} removed)")
    print(f"  Flow Retained: {flow_retained_pct:.1f}%")

    return graph


def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load graph from pickle file."""
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def save_graph(graph: nx.DiGraph, output_path: Path) -> None:
    """Save graph to pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)


def prune_graph_file(
    input_path: Path,
    threshold: int,
    output_path: Optional[Path] = None
) -> Path:
    """
    Load, prune, and save a graph file.

    Args:
        input_path: Path to input graph pickle file
        threshold: Minimum edge weight to keep
        output_path: Path to save pruned graph (if None, inferred from input)

    Returns:
        Path to pruned graph file
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Graph file not found: {input_path}")

    # Infer output path if not provided
    if output_path is None:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_pruned_{threshold}{suffix}"

    # Load, prune, and save
    graph_name = input_path.stem
    print(f"\nProcessing {graph_name}...")
    graph = load_graph(input_path)
    pruned_graph = prune_graph(graph, threshold, graph_name)
    save_graph(pruned_graph, output_path)

    print(f"  Saved: {output_path}")

    return output_path


def main():
    """Prune all graph files in the graphs directory."""
    # Parse command-line arguments
    threshold = DEFAULT_THRESHOLD
    if len(sys.argv) > 1:
        try:
            threshold = int(sys.argv[1])
        except ValueError:
            print(f"[ERROR] Invalid threshold value: {sys.argv[1]}")
            print(f"Usage: uv run scripts/prune_graphs.py [threshold]")
            print(f"Example: uv run scripts/prune_graphs.py 50")
            sys.exit(1)

    print("="*80)
    print("GRAPH EDGE PRUNING")
    print("="*80)
    print(f"\nThreshold: edges with flow >= {threshold}")
    print(f"Graphs directory: {GRAPHS_DIR}")

    if not GRAPHS_DIR.exists():
        print(f"\n[ERROR] Graphs directory not found: {GRAPHS_DIR}")
        return

    # Find all original (unpruned) graph files
    # Avoid pruning already-pruned graphs
    graph_files = sorted([
        f for f in GRAPHS_DIR.glob("*.pkl")
        if "_pruned_" not in f.name
    ])

    if not graph_files:
        print(f"\n[ERROR] No original graph files found in {GRAPHS_DIR}")
        print("(Looking for .pkl files without '_pruned_' in the name)")
        return

    print(f"\nFound {len(graph_files)} original graph file(s)")

    # Prune each graph
    pruned_files = []
    for graph_path in graph_files:
        try:
            pruned_path = prune_graph_file(graph_path, threshold)
            pruned_files.append(pruned_path)
        except Exception as e:
            print(f"  [ERROR] Failed to prune {graph_path}: {e}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nProcessed: {len(pruned_files)} graph(s)")
    print("\nPruned graphs saved:")
    for pruned_path in pruned_files:
        print(f"  - {pruned_path.name}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review the pruning statistics above")
    print("2. Visualize the pruned graphs: uv run scripts/visualize_pruned_graphs.py")
    print(f"3. To try a different threshold, run:")
    print(f"   uv run scripts/prune_graphs.py 100  # (or any other threshold)")


if __name__ == "__main__":
    main()
