"""
Calculate Shortest Path Distances from Source Node

Using the commuter_transit_routes graph, calculates shortest path distance from
node (135,77) to all other nodes. Uses inverse weights (1/x) as edge costs.
Unreachable nodes are set to NaN.
"""

import pickle
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Source node for shortest path analysis
SOURCE_NODE = (135, 77)
GRID_SIZE = 200


def main():
    """Calculate shortest paths and visualize"""
    print("=" * 80)
    print("SHORTEST COMMUTE PATH ANALYSIS")
    print("=" * 80)

    # Load commuter routes graph
    routes_path = GRAPHS_DIR / "commuter_transit_routes.pkl"

    if not routes_path.exists():
        print(f"\n[ERROR] Commuter routes file not found: {routes_path}")
        print("Run 'uv run scripts/identify_commuter_routes.py' first")
        return

    print(f"\nLoading {routes_path.name}...")
    with open(routes_path, 'rb') as f:
        G_commuter = pickle.load(f)

    print(f"Graph loaded: {G_commuter.number_of_nodes():,} nodes, {G_commuter.number_of_edges():,} edges")

    # Check if source node exists
    if SOURCE_NODE not in G_commuter:
        print(f"\n[ERROR] Source node {SOURCE_NODE} not in graph")
        return

    print(f"\nSource node: {SOURCE_NODE}")

    # Create a copy of the graph with inverse weights
    G_inverse = G_commuter.copy()

    # Convert weights to inverse (1/weight)
    for u, v, data in G_inverse.edges(data=True):
        weight = data['weight']
        if weight > 0:
            G_inverse[u][v]['weight'] = 1.0 / weight
        else:
            # Handle edge case of zero weight (shouldn't happen in our data)
            G_inverse[u][v]['weight'] = float('inf')

    print("\nCalculating shortest paths using inverse weights (1/x)...")

    # Calculate shortest paths from source to all other nodes
    try:
        # Use Dijkstra's algorithm with inverse weights
        distances = nx.single_source_dijkstra_path_length(
            G_inverse, SOURCE_NODE, weight='weight'
        )
    except nx.NetworkXError as e:
        print(f"[ERROR] Failed to compute shortest paths: {e}")
        return

    print(f"Reachable nodes: {len(distances):,}")

    # Create a dictionary for all nodes with NaN for unreachable
    all_nodes = set(G_commuter.nodes())
    distance_map = {}

    for node in all_nodes:
        if node in distances:
            distance_map[node] = distances[node]
        else:
            distance_map[node] = np.nan

    # Statistics
    reachable_distances = [d for d in distance_map.values() if not np.isnan(d)]
    unreachable_count = sum(1 for d in distance_map.values() if np.isnan(d))

    print(f"\nDistance statistics (reachable nodes only):")
    print(f"  Count: {len(reachable_distances):,}")
    print(f"  Min: {np.nanmin(list(reachable_distances)):.6f}")
    print(f"  Max: {np.nanmax(list(reachable_distances)):.6f}")
    print(f"  Mean: {np.nanmean(list(reachable_distances)):.6f}")
    print(f"  Median: {np.nanmedian(list(reachable_distances)):.6f}")
    print(f"  Std Dev: {np.nanstd(list(reachable_distances)):.6f}")
    print(f"\nUnreachable nodes: {unreachable_count:,}")

    # Save distances to disk
    distances_df = pd.DataFrame(
        [(node[0], node[1], dist) for node, dist in distance_map.items()],
        columns=['x', 'y', 'distance']
    )

    distances_path = OUTPUT_DIR / "shortest_paths_from_source.csv"
    distances_df.to_csv(distances_path, index=False)
    print(f"\nSaved: {distances_path}")

    # Also save as pickle for faster loading
    distances_pkl_path = OUTPUT_DIR / "shortest_paths_from_source.pkl"
    with open(distances_pkl_path, 'wb') as f:
        pickle.dump(distance_map, f)
    print(f"Saved: {distances_pkl_path}")

    # Create visualization
    print("\nCreating visualization...")
    visualize_shortest_paths(distance_map)


def visualize_shortest_paths(distance_map):
    """Create grid visualization of shortest path distances"""

    # Create grid arrays
    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)

    # Fill grid with distances
    for (x, y), distance in distance_map.items():
        # Adjust for 1-indexed grid
        if 1 <= x <= GRID_SIZE and 1 <= y <= GRID_SIZE:
            grid[y - 1, x - 1] = distance

    # Get statistics for visualization
    reachable_distances = grid[~np.isnan(grid)]

    # Create figure with multiple visualizations
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Linear distance gradient
    ax1 = plt.subplot(2, 3, 1)

    # Create a masked array to handle NaN values
    masked_grid = np.ma.masked_invalid(grid)

    im1 = ax1.imshow(masked_grid, cmap='viridis', origin='lower',
                      extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                      interpolation='bilinear')

    # Mark source node
    ax1.plot(SOURCE_NODE[0], SOURCE_NODE[1], 'r*', markersize=20,
             label=f'Source {SOURCE_NODE}')

    ax1.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax1.set_title('Shortest Path Distances (Linear Scale)\nUsing Inverse Weights (1/flow)',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    cbar1 = plt.colorbar(im1, ax=ax1, label='Distance')

    # Plot 2: Log scale distance gradient
    ax2 = plt.subplot(2, 3, 2)

    # Add small epsilon to avoid log(0)
    log_grid = np.log10(masked_grid + 1e-10)
    im2 = ax2.imshow(log_grid, cmap='plasma', origin='lower',
                      extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                      interpolation='bilinear')

    ax2.plot(SOURCE_NODE[0], SOURCE_NODE[1], 'r*', markersize=20,
             label=f'Source {SOURCE_NODE}')

    ax2.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax2.set_title('Shortest Path Distances (Log Scale)\nUsing Inverse Weights (1/flow)',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    cbar2 = plt.colorbar(im2, ax=ax2, label='Log10(Distance)')

    # Plot 3: Distance distribution histogram
    ax3 = plt.subplot(2, 3, 3)

    ax3.hist(reachable_distances, bins=100, color='steelblue',
             edgecolor='black', alpha=0.7)
    ax3.axvline(np.nanmean(reachable_distances), color='red',
                linestyle='--', linewidth=2,
                label=f'Mean: {np.nanmean(reachable_distances):.6f}')
    ax3.axvline(np.nanmedian(reachable_distances), color='green',
                linestyle='--', linewidth=2,
                label=f'Median: {np.nanmedian(reachable_distances):.6f}')

    ax3.set_xlabel('Distance', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Shortest Path Distances', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Reachability map (binary)
    ax4 = plt.subplot(2, 3, 4)

    reachability = np.where(np.isnan(grid), 0, 1)
    im4 = ax4.imshow(reachability, cmap='RdYlGn', origin='lower',
                      extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                      interpolation='nearest')

    ax4.plot(SOURCE_NODE[0], SOURCE_NODE[1], 'b*', markersize=20)
    ax4.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax4.set_title('Node Reachability Map\n(Red=Unreachable, Green=Reachable)',
                  fontsize=12, fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=ax4, label='Reachable')

    # Plot 5: Contour plot of distances
    ax5 = plt.subplot(2, 3, 5)

    x = np.arange(1, GRID_SIZE + 1)
    y = np.arange(1, GRID_SIZE + 1)
    X, Y = np.meshgrid(x, y)

    levels = np.nanpercentile(reachable_distances, np.linspace(0, 100, 20))
    contour = ax5.contourf(X, Y, masked_grid, levels=levels, cmap='viridis')
    ax5.plot(SOURCE_NODE[0], SOURCE_NODE[1], 'r*', markersize=20)

    ax5.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax5.set_title('Shortest Path Distance Contours', fontsize=12, fontweight='bold')
    ax5.set_aspect('equal')
    cbar5 = plt.colorbar(contour, ax=ax5, label='Distance')

    # Plot 6: Statistics text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
SHORTEST PATH ANALYSIS SUMMARY

Source Node: {SOURCE_NODE}
Weight Type: Inverse (1/commuter_weight)

REACHABILITY:
  Total nodes in graph: {np.sum(~np.isnan(list(distance_map.values()))):,}
  Reachable from source: {len(reachable_distances):,}
  Unreachable: {np.sum(np.isnan(list(distance_map.values()))):,}

DISTANCE STATISTICS:
  Min distance: {np.nanmin(reachable_distances):.6f}
  Max distance: {np.nanmax(reachable_distances):.6f}
  Mean distance: {np.nanmean(reachable_distances):.6f}
  Median distance: {np.nanmedian(reachable_distances):.6f}
  Std Dev: {np.nanstd(reachable_distances):.6f}

  Q1 (25%): {np.nanpercentile(reachable_distances, 25):.6f}
  Q3 (75%): {np.nanpercentile(reachable_distances, 75):.6f}
  Q90: {np.nanpercentile(reachable_distances, 90):.6f}
  Q99: {np.nanpercentile(reachable_distances, 99):.6f}
"""

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Shortest Commute Path Analysis from Source Node',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "shortest_paths_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Create a large detailed heatmap (linear scale)
    print("Creating detailed heatmap (linear scale)...")

    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(masked_grid, cmap='viridis', origin='lower',
                    extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                    interpolation='bilinear')

    ax.plot(SOURCE_NODE[0], SOURCE_NODE[1], 'r*', markersize=30,
            label=f'Source Node {SOURCE_NODE}', markeredgewidth=2, markeredgecolor='white')

    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title('Shortest Commute Distances from {}\n(200x200 grid, Using Inverse Weight Distance, Linear Scale)',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.2)

    cbar = plt.colorbar(im, ax=ax, label='Shortest Path Distance', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path2 = OUTPUT_DIR / "shortest_paths_detailed.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()

    # Create a large detailed heatmap (log scale)
    print("Creating detailed heatmap (log scale)...")

    fig, ax = plt.subplots(figsize=(16, 14))

    # Convert to log scale with small epsilon to avoid log(0)
    log_grid = np.log10(masked_grid + 1e-10)

    im = ax.imshow(log_grid, cmap='plasma', origin='lower',
                    extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                    interpolation='bilinear')

    ax.plot(SOURCE_NODE[0], SOURCE_NODE[1], 'r*', markersize=30,
            label=f'Source Node {SOURCE_NODE}', markeredgewidth=2, markeredgecolor='white')

    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title('Shortest Commute Distances from {}\n(200x200 grid, Using Inverse Weight Distance, Log Scale)',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.2)

    cbar = plt.colorbar(im, ax=ax, label='Log10(Shortest Path Distance)', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path3 = OUTPUT_DIR / "shortest_paths_detailed_logscale.png"
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path3}")
    plt.close()


if __name__ == "__main__":
    main()
