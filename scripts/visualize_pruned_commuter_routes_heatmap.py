"""
Visualize Pruned Commuter Routes as Heatmap

Creates heatmap visualization of pruned commuter transit routes graph,
where each grid cell's value is the sum of all edge weights connected to that node.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

GRID_SIZE = 200
THRESHOLD = 10


def main():
    """Create heatmap of pruned commuter routes"""
    print("=" * 80)
    print("VISUALIZING PRUNED COMMUTER ROUTES - HEATMAP")
    print("=" * 80)

    # Load pruned commuter routes graph
    routes_path = GRAPHS_DIR / f"commuter_transit_routes.pkl"

    if not routes_path.exists():
        print(f"\n[ERROR] Pruned commuter routes file not found: {routes_path}")
        return

    print(f"\nLoading {routes_path.name}...")
    with open(routes_path, 'rb') as f:
        G = pickle.load(f)

    print(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Calculate node importance (sum of connected edge weights)
    print("\nCalculating node importance...")
    node_weights = {}
    for node in G.nodes():
        total_weight = 0
        for neighbor in G.neighbors(node):
            total_weight += G[node][neighbor]['weight']
        node_weights[node] = total_weight

    # Create grid
    print("Creating grid...")
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    for node, weight in node_weights.items():
        x, y = node
        # Adjust for 1-indexed grid
        if 1 <= x <= GRID_SIZE and 1 <= y <= GRID_SIZE:
            grid[y - 1, x - 1] = weight

    # Find statistics
    nonzero_values = grid[grid > 0]
    print(f"\nGrid statistics:")
    print(f"  Cells with data: {len(nonzero_values):,}")
    print(f"  Min weight: {nonzero_values.min():.2f}")
    print(f"  Max weight: {nonzero_values.max():.2f}")
    print(f"  Mean weight: {nonzero_values.mean():.2f}")
    print(f"  Median weight: {np.median(nonzero_values):.2f}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Figure 1: Linear scale heatmap
    fig, ax = plt.subplots(figsize=(16, 14))

    masked_grid = np.ma.masked_where(grid == 0, grid)

    im = ax.imshow(masked_grid, cmap='YlOrRd', origin='lower',
                    extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                    interpolation='bilinear')

    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title('Pruned Commuter Routes - Node Importance Heatmap (Linear Scale)\n(Value = Sum of Connected Edge Weights, Threshold > 10)',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.2)

    cbar = plt.colorbar(im, ax=ax, label='Node Importance (Sum of Edge Weights)', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path1 = OUTPUT_DIR / "pruned_commuter_routes_heatmap_linear.png"
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path1}")
    plt.close()

    # Figure 2: Log scale heatmap
    fig, ax = plt.subplots(figsize=(16, 14))

    log_grid = np.log10(masked_grid + 1e-10)

    im = ax.imshow(log_grid, cmap='plasma', origin='lower',
                    extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                    interpolation='bilinear')

    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title('Pruned Commuter Routes - Node Importance Heatmap (Log Scale)\n(Value = Log10(Sum of Connected Edge Weights), Threshold > 10)',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.2)

    cbar = plt.colorbar(im, ax=ax, label='Log10(Node Importance)', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path2 = OUTPUT_DIR / "pruned_commuter_routes_heatmap_logscale.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()

    # Figure 3: Multi-panel comparison
    fig = plt.figure(figsize=(20, 10))

    # Linear scale
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(masked_grid, cmap='YlOrRd', origin='lower',
                      extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                      interpolation='bilinear')
    ax1.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Scale', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Node Importance')

    # Log scale
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(log_grid, cmap='plasma', origin='lower',
                      extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                      interpolation='bilinear')
    ax2.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax2.set_title('Log Scale', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Log10(Node Importance)')

    plt.suptitle('Pruned Commuter Routes - Node Importance Comparison (Threshold > 10)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path3 = OUTPUT_DIR / "pruned_commuter_routes_heatmap_comparison.png"
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path3}")
    plt.close()

    # Figure 4: Distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram (linear)
    ax = axes[0, 0]
    ax.hist(nonzero_values, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(nonzero_values.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {nonzero_values.mean():.2f}')
    ax.axvline(np.median(nonzero_values), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(nonzero_values):.2f}')
    ax.set_xlabel('Node Importance', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Node Importance (Linear)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Histogram (log y-axis)
    ax = axes[0, 1]
    ax.hist(nonzero_values, bins=100, color='coral', edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Node Importance', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency (log)', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Node Importance (Log Y)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Cumulative distribution
    ax = axes[1, 0]
    sorted_vals = np.sort(nonzero_values)
    cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cumulative, linewidth=2, color='darkgreen')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax.set_xlabel('Node Importance', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Percentile breakdown
    ax = axes[1, 1]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(nonzero_values, p) for p in percentiles]
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    bars = ax.barh([f'{p}th' for p in percentiles], percentile_values, color=colors_bar,
                    edgecolor='black')
    ax.set_xlabel('Node Importance', fontsize=11, fontweight='bold')
    ax.set_title('Percentile Breakdown', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, percentile_values)):
        ax.text(val, i, f' {val:.1f}', va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Node Importance Statistics - Pruned Commuter Routes (Threshold > 10)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path4 = OUTPUT_DIR / "pruned_commuter_routes_statistics.png"
    plt.savefig(output_path4, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path4}")
    plt.close()

    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION SUMMARY")
    print("=" * 80)
    print(f"Threshold: >{THRESHOLD}")
    print(f"Grid size: {GRID_SIZE}×{GRID_SIZE}")
    print(f"Nodes visualized: {len(nonzero_values):,}")
    print(f"Max node importance: {nonzero_values.max():.2f} at coordinates", end=" ")
    max_node = max(node_weights.items(), key=lambda x: x[1])
    print(f"{max_node[0]}")


if __name__ == "__main__":
    main()
