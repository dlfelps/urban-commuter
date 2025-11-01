"""
Plot Histogram of Commuter Transit Route Edge Weights

Visualizes the distribution of edge weights in the commuter_transit_routes graph.
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
    """Load commuter routes and plot weight distribution"""
    print("="*80)
    print("COMMUTER TRANSIT ROUTES - EDGE WEIGHT HISTOGRAM")
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

    # Extract edge weights
    weights = [data['weight'] for _, _, data in G_commuter.edges(data=True)]

    print(f"\nEdge weight statistics:")
    print(f"  Min: {min(weights):.3f}")
    print(f"  Max: {max(weights):.3f}")
    print(f"  Mean: {np.mean(weights):.3f}")
    print(f"  Median: {np.median(weights):.3f}")
    print(f"  Std Dev: {np.std(weights):.3f}")
    print(f"  Q1 (25th percentile): {np.percentile(weights, 25):.3f}")
    print(f"  Q3 (75th percentile): {np.percentile(weights, 75):.3f}")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Linear histogram with all data
    ax1 = plt.subplot(2, 3, 1)
    counts, bins, patches = ax1.hist(weights, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Edge Weight', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Edge Weight Distribution (Linear)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0.98, 0.97, f'n={len(weights):,}\nMean={np.mean(weights):.2f}\nMedian={np.median(weights):.2f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: Log scale histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(weights, bins=100, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Edge Weight', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Edge Weight Distribution (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Cumulative distribution
    ax3 = plt.subplot(2, 3, 3)
    sorted_weights = np.sort(weights)
    cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
    ax3.plot(sorted_weights, cumulative, linewidth=2, color='darkgreen')
    ax3.set_xlabel('Edge Weight', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax3.legend()

    # Plot 4: Box plot with quartiles
    ax4 = plt.subplot(2, 3, 4)
    bp = ax4.boxplot(weights, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    ax4.set_ylabel('Edge Weight', fontsize=12, fontweight='bold')
    ax4.set_title('Box Plot of Edge Weights', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add text with quartile info
    q1, q3 = np.percentile(weights, [25, 75])
    median = np.median(weights)
    ax4.text(1.15, q3, f'Q3: {q3:.2f}', fontsize=10, verticalalignment='center')
    ax4.text(1.15, median, f'Median: {median:.2f}', fontsize=10, verticalalignment='center')
    ax4.text(1.15, q1, f'Q1: {q1:.2f}', fontsize=10, verticalalignment='center')

    # Plot 5: Histogram with log bins
    ax5 = plt.subplot(2, 3, 5)
    # Use log-spaced bins for better view of long tail
    bins_log = np.logspace(np.log10(min(weights)), np.log10(max(weights)), 50)
    ax5.hist(weights, bins=bins_log, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax5.set_xscale('log')
    ax5.set_xlabel('Edge Weight (log scale)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('Edge Weight Distribution (Log-Spaced Bins)', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y', which='both')

    # Plot 6: Percentile breakdown
    ax6 = plt.subplot(2, 3, 6)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(weights, p) for p in percentiles]
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    bars = ax6.barh([f'{p}th' for p in percentiles], percentile_values, color=colors_bar, edgecolor='black')
    ax6.set_xlabel('Edge Weight', fontsize=12, fontweight='bold')
    ax6.set_title('Percentile Breakdown', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, percentile_values)):
        ax6.text(val, i, f' {val:.2f}', va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Commuter Transit Routes - Edge Weight Analysis', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "commuter_routes_weight_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*80)
    print("PERCENTILE DISTRIBUTION")
    print("="*80)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(weights, p)
        print(f"{p:3d}th percentile: {val:8.3f}")

    # Count edges by weight ranges
    print("\n" + "="*80)
    print("EDGE COUNT BY WEIGHT RANGE")
    print("="*80)
    ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 50), (50, 100), (100, 500), (500, 10000)]
    for low, high in ranges:
        count = sum(1 for w in weights if low <= w < high)
        pct = (count / len(weights)) * 100
        if high == 10000:
            label = f"[{low:4d}, inf)"
        else:
            label = f"[{low:4d}, {high:4d})"
        print(f"{label}: {count:8,} edges ({pct:6.2f}%)")


if __name__ == "__main__":
    main()
