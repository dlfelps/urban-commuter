"""
Visualize Commuter Transit Routes - Node Importance by Connected Flow

Plots nodes from the commuter transit routes graph with circle sizes
proportional to the sum of edge weights connected to each node.
This shows the importance/centrality of each location in commuter flows.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    """Load commuter routes and visualize node importance"""
    print("="*80)
    print("COMMUTER TRANSIT ROUTES - NODE IMPORTANCE VISUALIZATION")
    print("="*80)

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

    # Calculate node importance (sum of connected edge weights)
    node_weights = {}
    for node in G_commuter.nodes():
        total_weight = 0
        # Sum weights of all edges connected to this node
        # For undirected graphs, each edge appears once in neighbors
        for neighbor in G_commuter.neighbors(node):
            total_weight += G_commuter[node][neighbor]['weight']
        node_weights[node] = total_weight

    # Extract coordinates and weights
    nodes = list(G_commuter.nodes())
    x_coords = np.array([node[0] for node in nodes])
    y_coords = np.array([node[1] for node in nodes])
    weights = np.array([node_weights[node] for node in nodes])

    # Statistics
    print(f"\nNode weight statistics:")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")
    print(f"  Mean: {weights.mean():.3f}")
    print(f"  Median: {np.median(weights):.3f}")
    print(f"  Std Dev: {weights.std():.3f}")

    # Find top 10 nodes by weight
    top_indices = np.argsort(weights)[-10:][::-1]
    print(f"\nTop 10 Most Important Nodes (by connected flow):")
    for rank, idx in enumerate(top_indices, 1):
        node = nodes[idx]
        weight = weights[idx]
        print(f"  {rank:2d}. Node {node}: {weight:8.2f}")

    # Create multiple visualizations
    fig = plt.figure(figsize=(18, 14))

    # Plot 1: Full scatter with proportional sizes
    ax1 = plt.subplot(2, 3, 1)

    # Scale weights for circle sizes (area proportional to weight)
    # Use sqrt for visual scaling since area = π*r^2
    sizes = np.sqrt(weights) * 3  # Scale factor for visibility

    scatter = ax1.scatter(x_coords, y_coords, s=sizes, alpha=0.6,
                         c=weights, cmap='YlOrRd', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax1.set_title('Node Importance by Connected Flow\n(Circle area proportional to edge weight sum)',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(scatter, ax=ax1, label='Total Connected Weight')

    # Plot 2: Log scale visualization (better for dynamic range)
    ax2 = plt.subplot(2, 3, 2)

    # Use log scale for better visibility of smaller nodes
    log_weights = np.log10(weights + 1)  # +1 to avoid log(0)
    sizes_log = np.sqrt(log_weights) * 5

    scatter2 = ax2.scatter(x_coords, y_coords, s=sizes_log, alpha=0.6,
                          c=log_weights, cmap='viridis', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax2.set_title('Node Importance (Log Scale)\n(Better for visualizing dynamic range)',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Log10(Total Weight + 1)')

    # Plot 3: Density heatmap
    ax3 = plt.subplot(2, 3, 3)

    # Create 2D histogram of weights
    h = ax3.hist2d(x_coords, y_coords, bins=40, weights=weights, cmap='hot')
    ax3.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax3.set_title('Flow Intensity Heatmap\n(Aggregated by grid cell)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(h[3], ax=ax3, label='Aggregated Weight')
    ax3.set_aspect('equal')

    # Plot 4: Weight distribution histogram
    ax4 = plt.subplot(2, 3, 4)

    ax4.hist(weights, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(weights.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {weights.mean():.2f}')
    ax4.axvline(np.median(weights), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(weights):.2f}')
    ax4.set_xlabel('Total Connected Weight', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Node Importance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()

    # Plot 5: Scatter with size legend
    ax5 = plt.subplot(2, 3, 5)

    scatter5 = ax5.scatter(x_coords, y_coords, s=sizes, alpha=0.6,
                          c=weights, cmap='plasma', edgecolors='black', linewidth=0.5)
    ax5.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax5.set_title('Node Importance (Plasma colormap)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')

    # Add size legend
    legend_sizes = [weights.min(), weights.mean(), weights.max()]
    legend_labels = [f'{s:.1f}' for s in legend_sizes]
    legend_handles = []
    for size, label in zip(legend_sizes, legend_labels):
        legend_handles.append(plt.scatter([], [], s=np.sqrt(size)*3, c='gray',
                                         edgecolors='black', linewidth=0.5, label=label))
    ax5.legend(legend_handles, legend_labels, scatterpoints=1,
              loc='upper left', frameon=True, fontsize=9, title='Weight')
    cbar5 = plt.colorbar(scatter5, ax=ax5, label='Total Connected Weight')

    # Plot 6: Top nodes highlighted
    ax6 = plt.subplot(2, 3, 6)

    # Plot all nodes small
    ax6.scatter(x_coords, y_coords, s=sizes*0.3, alpha=0.2,
               c='lightgray', edgecolors='none')

    # Plot top 20 nodes large
    top_20_indices = np.argsort(weights)[-20:]
    top_x = x_coords[top_20_indices]
    top_y = y_coords[top_20_indices]
    top_weights = weights[top_20_indices]
    top_sizes = np.sqrt(top_weights) * 3

    scatter6 = ax6.scatter(top_x, top_y, s=top_sizes, alpha=0.8,
                          c=top_weights, cmap='Reds', edgecolors='darkred', linewidth=1)

    ax6.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax6.set_title('Top 20 Most Important Nodes\n(All others shown in light gray)',
                  fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    cbar6 = plt.colorbar(scatter6, ax=ax6, label='Total Connected Weight')

    plt.suptitle('Commuter Transit Routes - Node Importance Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "commuter_routes_node_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Create additional focused plot with better size scaling
    print("\nCreating additional detailed visualization...")

    fig, ax = plt.subplots(figsize=(16, 14))

    # Better size scaling using square root of weights
    sizes_detailed = (np.sqrt(weights) / np.sqrt(weights.max())) * 500 + 10

    scatter = ax.scatter(x_coords, y_coords, s=sizes_detailed, alpha=0.6,
                        c=weights, cmap='coolwarm', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title('Commuter Transit Routes - Node Importance Map\n(Circle size = total connected edge weight)',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    cbar = plt.colorbar(scatter, ax=ax, label='Total Connected Weight', shrink=0.8)

    # Annotate top 5 nodes
    top_5_indices = np.argsort(weights)[-5:][::-1]
    for rank, idx in enumerate(top_5_indices, 1):
        node = nodes[idx]
        weight = weights[idx]
        ax.annotate(f'#{rank}\n({node[0]},{node[1]})\nW:{weight:.1f}',
                   xy=(x_coords[idx], y_coords[idx]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    output_path2 = OUTPUT_DIR / "commuter_routes_node_importance_detailed.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()

    # Statistics by weight ranges
    print("\n" + "="*80)
    print("NODE DISTRIBUTION BY IMPORTANCE")
    print("="*80)

    ranges = [
        (0, 1),
        (1, 2),
        (2, 5),
        (5, 10),
        (10, 20),
        (20, 50),
        (50, 100),
        (100, 1000)
    ]

    for low, high in ranges:
        count = sum(1 for w in weights if low <= w < high)
        pct = (count / len(weights)) * 100
        label = f"[{low:4d}, {high:4d})" if high != 1000 else f"[{low:4d}, inf)"
        print(f"{label}: {count:6,} nodes ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
