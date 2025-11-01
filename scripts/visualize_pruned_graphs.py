"""
Visualize Pruned Commute Graphs

Creates spatial visualizations of pruned morning and afternoon commute networks
with nodes positioned at their actual grid coordinates (x, y).

Usage:
    uv run scripts/visualize_pruned_graphs.py                  # Visualize latest pruned graphs
    uv run scripts/visualize_pruned_graphs.py 50               # Visualize graphs pruned at threshold 50
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
import sys

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

INTERMEDIATE_DIR = Path(__file__).parent.parent / "intermediate"


def load_graph(graph_path):
    """Load graph from pickle file"""
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def create_spatial_visualization(G, graph_name, output_path, figsize=(14, 12), sample_edges=True):
    """
    Create spatial visualization of a commute graph

    Args:
        G: NetworkX DiGraph
        graph_name: Name of the graph (for title)
        output_path: Path to save the figure
        figsize: Figure size
        sample_edges: If True, sample high-weight edges for faster rendering
    """
    print(f"\nCreating visualization for {graph_name}...")

    fig, ax = plt.subplots(figsize=figsize)

    # Create position dictionary using node coordinates
    # Nodes are tuples (x, y)
    pos = {node: node for node in G.nodes()}

    # Draw edges with varying alpha based on weight
    # Normalize edge weights for visualization
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights)
    min_weight = min(weights)

    # Sample edges if too many (render only higher-weight edges)
    edges_to_draw = list(G.edges())
    if sample_edges and len(edges_to_draw) > 50000:
        # Sort by weight and keep top 50K edges
        edges_to_draw = sorted(edges_to_draw, key=lambda e: G[e[0]][e[1]]['weight'], reverse=True)[:50000]
        print(f"  Sampling {len(edges_to_draw):,} highest-weight edges for faster rendering")

    # Create edge list with normalized widths and alphas
    for u, v in edges_to_draw:
        weight = G[u][v]['weight']
        # Normalize weight to 0-1
        norm_weight = (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5

        # Line width based on weight (thicker = more flow)
        line_width = 0.1 + norm_weight * 1.5

        # Alpha (transparency) based on weight (more opaque = more flow)
        alpha = 0.1 + norm_weight * 0.4

        x = [u[0], v[0]]
        y = [u[1], v[1]]
        ax.plot(x, y, 'b-', linewidth=line_width, alpha=alpha, zorder=1)

    # Draw nodes
    node_x = [node[0] for node in G.nodes()]
    node_y = [node[1] for node in G.nodes()]

    # Color nodes by degree (in-degree)
    node_colors = [G.in_degree(node) for node in G.nodes()]

    nodes = ax.scatter(node_x, node_y, c=node_colors, cmap='YlOrRd',
                       s=50, alpha=0.8, edgecolors='black', linewidth=0.5,
                       zorder=2)

    # Add colorbar
    cbar = plt.colorbar(nodes, ax=ax, label='In-Degree (incoming flows)')

    # Labels and formatting
    ax.set_xlabel('X Coordinate (grid cell)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate (grid cell)', fontsize=12, fontweight='bold')
    ax.set_title(f'{graph_name}\nSpatial Network Visualization',
                 fontsize=14, fontweight='bold')

    # Set grid
    ax.grid(True, alpha=0.2, linestyle='--')

    # Set equal aspect ratio to preserve grid coordinates
    ax.set_aspect('equal')

    # Add statistics as text
    stats_text = (
        f"Nodes: {G.number_of_nodes():,}\n"
        f"Edges: {G.number_of_edges():,}\n"
        f"Total Flow: {sum(data['weight'] for _, _, data in G.edges(data=True)):,}\n"
        f"Avg Edge Weight: {sum(data['weight'] for _, _, data in G.edges(data=True)) / G.number_of_edges():.1f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_comparison_visualization(morning_G, afternoon_G, output_path, figsize=(18, 8)):
    """
    Create side-by-side comparison visualization of pruned graphs

    Args:
        morning_G: Morning commute graph
        afternoon_G: Afternoon commute graph
        output_path: Path to save the figure
        figsize: Figure size
    """
    print(f"\nCreating side-by-side comparison visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Morning visualization
    weights_morning = [morning_G[u][v]['weight'] for u, v in morning_G.edges()]
    max_weight_morning = max(weights_morning)
    min_weight_morning = min(weights_morning)

    # Sample edges for morning
    morning_edges = list(morning_G.edges())
    if len(morning_edges) > 50000:
        morning_edges = sorted(morning_edges, key=lambda e: morning_G[e[0]][e[1]]['weight'], reverse=True)[:50000]

    for u, v in morning_edges:
        weight = morning_G[u][v]['weight']
        norm_weight = (weight - min_weight_morning) / (max_weight_morning - min_weight_morning) if max_weight_morning > min_weight_morning else 0.5
        line_width = 0.1 + norm_weight * 1.5
        alpha = 0.1 + norm_weight * 0.4
        x = [u[0], v[0]]
        y = [u[1], v[1]]
        ax1.plot(x, y, 'b-', linewidth=line_width, alpha=alpha, zorder=1)

    node_x_morning = [node[0] for node in morning_G.nodes()]
    node_y_morning = [node[1] for node in morning_G.nodes()]
    node_colors_morning = [morning_G.in_degree(node) for node in morning_G.nodes()]

    nodes1 = ax1.scatter(node_x_morning, node_y_morning, c=node_colors_morning,
                         cmap='YlOrRd', s=50, alpha=0.8, edgecolors='black',
                         linewidth=0.5, zorder=2)

    ax1.set_xlabel('X Coordinate (grid cell)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y Coordinate (grid cell)', fontsize=11, fontweight='bold')
    ax1.set_title('Morning Commute (7:00-11:00 AM)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_aspect('equal')

    cbar1 = plt.colorbar(nodes1, ax=ax1, label='In-Degree')

    # Afternoon visualization
    weights_afternoon = [afternoon_G[u][v]['weight'] for u, v in afternoon_G.edges()]
    max_weight_afternoon = max(weights_afternoon)
    min_weight_afternoon = min(weights_afternoon)

    # Sample edges for afternoon
    afternoon_edges = list(afternoon_G.edges())
    if len(afternoon_edges) > 50000:
        afternoon_edges = sorted(afternoon_edges, key=lambda e: afternoon_G[e[0]][e[1]]['weight'], reverse=True)[:50000]

    for u, v in afternoon_edges:
        weight = afternoon_G[u][v]['weight']
        norm_weight = (weight - min_weight_afternoon) / (max_weight_afternoon - min_weight_afternoon) if max_weight_afternoon > min_weight_afternoon else 0.5
        line_width = 0.1 + norm_weight * 1.5
        alpha = 0.1 + norm_weight * 0.4
        x = [u[0], v[0]]
        y = [u[1], v[1]]
        ax2.plot(x, y, 'b-', linewidth=line_width, alpha=alpha, zorder=1)

    node_x_afternoon = [node[0] for node in afternoon_G.nodes()]
    node_y_afternoon = [node[1] for node in afternoon_G.nodes()]
    node_colors_afternoon = [afternoon_G.in_degree(node) for node in afternoon_G.nodes()]

    nodes2 = ax2.scatter(node_x_afternoon, node_y_afternoon, c=node_colors_afternoon,
                         cmap='YlOrRd', s=50, alpha=0.8, edgecolors='black',
                         linewidth=0.5, zorder=2)

    ax2.set_xlabel('X Coordinate (grid cell)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y Coordinate (grid cell)', fontsize=11, fontweight='bold')
    ax2.set_title('Afternoon Commute (3:00-7:00 PM)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_aspect('equal')

    cbar2 = plt.colorbar(nodes2, ax=ax2, label='In-Degree')

    fig.suptitle('Pruned Weekday Commute Network Comparison (Flow >= 20)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_flow_heatmap(G, graph_name, output_path, figsize=(12, 10)):
    """
    Create heatmap of flow intensities on the grid

    Args:
        G: NetworkX DiGraph
        graph_name: Name of the graph
        output_path: Path to save the figure
        figsize: Figure size
    """
    print(f"\nCreating heatmap for {graph_name}...")

    # Create a grid to accumulate flows
    grid_max = 200
    grid = np.zeros((grid_max + 1, grid_max + 1))

    # Accumulate edge weights on the grid (using source node position)
    for u, v in G.edges():
        weight = G[u][v]['weight']
        x, y = u
        if 1 <= x <= grid_max and 1 <= y <= grid_max:
            grid[y, x] += weight

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap (note: imshow has y-axis inverted by default)
    im = ax.imshow(grid, cmap='hot', origin='lower', aspect='auto',
                   extent=[1, grid_max, 1, grid_max], interpolation='bilinear')

    ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax.set_title(f'{graph_name}\nFlow Intensity Heatmap (by source cell)',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='Total Flow Intensity')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def find_pruned_graphs(threshold=None):
    """
    Find pruned graph files.

    Args:
        threshold: Specific threshold to look for (e.g., 50). If None, find latest.

    Returns:
        Tuple of (morning_path, afternoon_path, threshold_used)
    """
    if threshold is not None:
        # Look for specific threshold
        morning_path = GRAPHS_DIR / f"morning_commute_weekday_v2_pruned_{threshold}.pkl"
        afternoon_path = GRAPHS_DIR / f"afternoon_commute_weekday_v2_pruned_{threshold}.pkl"
        if morning_path.exists() and afternoon_path.exists():
            return morning_path, afternoon_path, threshold
        else:
            return None, None, None

    # Find all pruned graphs and get the highest threshold
    pruned_files = list(GRAPHS_DIR.glob("*_pruned_*.pkl"))
    if not pruned_files:
        return None, None, None

    # Extract thresholds from filenames
    thresholds = set()
    for f in pruned_files:
        # Extract number from *_pruned_N.pkl
        try:
            threshold_str = f.stem.split("_pruned_")[-1]
            thresholds.add(int(threshold_str))
        except (ValueError, IndexError):
            pass

    if not thresholds:
        return None, None, None

    # Use highest threshold found
    threshold = max(thresholds)
    morning_path = GRAPHS_DIR / f"morning_commute_weekday_v2_pruned_{threshold}.pkl"
    afternoon_path = GRAPHS_DIR / f"afternoon_commute_weekday_v2_pruned_{threshold}.pkl"

    if morning_path.exists() and afternoon_path.exists():
        return morning_path, afternoon_path, threshold

    return None, None, None


def main():
    """Create all visualizations of pruned graphs"""
    print("=" * 70)
    print("VISUALIZING PRUNED COMMUTE GRAPHS")
    print("=" * 70)

    # Parse command-line arguments
    threshold = None
    if len(sys.argv) > 1:
        try:
            threshold = int(sys.argv[1])
        except ValueError:
            print(f"[ERROR] Invalid threshold value: {sys.argv[1]}")
            print(f"Usage: uv run scripts/visualize_pruned_graphs.py [threshold]")
            print(f"Example: uv run scripts/visualize_pruned_graphs.py 50")
            sys.exit(1)

    # Find pruned graphs
    morning_path, afternoon_path, threshold_used = find_pruned_graphs(threshold)

    if morning_path is None or afternoon_path is None:
        print(f"\n[ERROR] Pruned graph files not found.")
        if threshold is not None:
            print(f"  Could not find graphs pruned at threshold {threshold}")
        else:
            print(f"  No pruned graphs found in {GRAPHS_DIR}")
        print(f"\nRun 'uv run scripts/prune_graphs.py [threshold]' first to generate pruned graphs.")
        print(f"Example: uv run scripts/prune_graphs.py 50")
        return

    print(f"\nLoading pruned graphs (threshold={threshold_used})...")
    morning_G = load_graph(morning_path)
    afternoon_G = load_graph(afternoon_path)
    print(f"  Morning: {morning_G.number_of_nodes():,} nodes, {morning_G.number_of_edges():,} edges")
    print(f"  Afternoon: {afternoon_G.number_of_nodes():,} nodes, {afternoon_G.number_of_edges():,} edges")

    # Create individual visualizations
    create_spatial_visualization(
        morning_G,
        f"Morning Commute - Pruned (Flow >= {threshold_used})",
        OUTPUT_DIR / f"morning_commute_pruned_{threshold_used}_network.png"
    )

    create_spatial_visualization(
        afternoon_G,
        f"Afternoon Commute - Pruned (Flow >= {threshold_used})",
        OUTPUT_DIR / f"afternoon_commute_pruned_{threshold_used}_network.png"
    )

    # Create comparison visualization
    create_comparison_visualization(
        morning_G,
        afternoon_G,
        OUTPUT_DIR / f"commute_pruned_{threshold_used}_comparison.png"
    )

    # Create heatmaps
    create_flow_heatmap(
        morning_G,
        f"Morning Commute Flow Intensity - Pruned (>= {threshold_used})",
        OUTPUT_DIR / f"morning_commute_pruned_{threshold_used}_heatmap.png"
    )

    create_flow_heatmap(
        afternoon_G,
        f"Afternoon Commute Flow Intensity - Pruned (>= {threshold_used})",
        OUTPUT_DIR / f"afternoon_commute_pruned_{threshold_used}_heatmap.png"
    )

    print("\n" + "=" * 70)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated visualizations (threshold={threshold_used}):")
    print(f"  - {OUTPUT_DIR / f'morning_commute_pruned_{threshold_used}_network.png'}")
    print(f"  - {OUTPUT_DIR / f'afternoon_commute_pruned_{threshold_used}_network.png'}")
    print(f"  - {OUTPUT_DIR / f'commute_pruned_{threshold_used}_comparison.png'}")
    print(f"  - {OUTPUT_DIR / f'morning_commute_pruned_{threshold_used}_heatmap.png'}")
    print(f"  - {OUTPUT_DIR / f'afternoon_commute_pruned_{threshold_used}_heatmap.png'}")


if __name__ == "__main__":
    main()
