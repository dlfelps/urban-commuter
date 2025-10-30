"""
Analyze Edge Weight Distribution

Analyzes the edge weight distribution in commute graphs to determine
optimal trimming thresholds that capture most flow with fewer edges.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_graph(graph_path):
    """Load graph from pickle file"""
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def analyze_edge_weights(G, graph_name):
    """Analyze edge weight distribution"""
    print(f"\n{'='*70}")
    print(f"{graph_name.upper()} EDGE WEIGHT ANALYSIS")
    print(f"{'='*70}")

    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    weights_sorted = sorted(weights, reverse=True)

    print(f"\nEdge Count: {len(weights):,}")
    print(f"Total Flow: {sum(weights):,}")
    print(f"\nWeight Statistics:")
    print(f"  Min: {min(weights)}")
    print(f"  Max: {max(weights)}")
    print(f"  Mean: {np.mean(weights):.2f}")
    print(f"  Median: {np.median(weights):.2f}")
    print(f"  Std Dev: {np.std(weights):.2f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(weights, p)
        print(f"  {p}th percentile: {val:.0f}")

    # Calculate cumulative flow by threshold
    print(f"\n{'Threshold':<12} {'Edges Kept':<15} {'Flow Kept':<15} {'% of Flow':<12} {'% of Edges':<12}")
    print(f"{'-'*70}")

    cumulative_flow = 0
    thresholds = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100]

    results = []

    for threshold in thresholds:
        edges_above = sum(1 for w in weights if w >= threshold)
        flow_above = sum(w for w in weights if w >= threshold)
        pct_flow = (flow_above / sum(weights)) * 100
        pct_edges = (edges_above / len(weights)) * 100

        results.append({
            'threshold': threshold,
            'edges': edges_above,
            'flow': flow_above,
            'pct_flow': pct_flow,
            'pct_edges': pct_edges
        })

        print(f"{threshold:<12} {edges_above:<15,} {flow_above:<15,} {pct_flow:<11.1f}% {pct_edges:<11.1f}%")

    return weights_sorted, results


def create_weight_distribution_plot(weights, graph_name, output_path):
    """Create histogram and cumulative distribution plot"""
    print(f"\nCreating weight distribution visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram (log scale)
    ax1.hist(weights, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Edge Weight', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'{graph_name}\nEdge Weight Distribution (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_weights = sorted(weights, reverse=True)
    cumsum = np.cumsum(sorted_weights)
    cumsum_pct = (cumsum / cumsum[-1]) * 100

    ax2.plot(range(len(sorted_weights)), cumsum_pct, linewidth=2, color='darkred')
    ax2.axhline(y=95, color='green', linestyle='--', linewidth=2, label='95% flow')
    ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% flow')
    ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% flow')
    ax2.set_xlabel('Edge Rank (sorted by weight)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Flow %', fontsize=12, fontweight='bold')
    ax2.set_title(f'{graph_name}\nCumulative Flow Distribution', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def find_optimal_threshold(results, target_flow_pct=95):
    """Find threshold that captures target % of flow with fewest edges"""
    print(f"\n{'='*70}")
    print(f"OPTIMAL THRESHOLD ANALYSIS (Target: {target_flow_pct}% flow)")
    print(f"{'='*70}")

    # Find smallest threshold that achieves target flow
    for r in results:
        if r['pct_flow'] >= target_flow_pct:
            print(f"\nTo capture {target_flow_pct}% of flow:")
            print(f"  Threshold: {r['threshold']}")
            print(f"  Edges to keep: {r['edges']:,} ({r['pct_edges']:.1f}% of total)")
            print(f"  Flow captured: {r['flow']:,} ({r['pct_flow']:.1f}%)")
            print(f"  Edges to remove: {results[0]['edges'] - r['edges']:,}")
            return r['threshold']

    return None


def main():
    """Analyze edge weights for both commute graphs"""
    print("="*70)
    print("COMMUTE GRAPH EDGE WEIGHT ANALYSIS")
    print("="*70)

    # Load graphs (v2 versions have full 200x200 grid)
    morning_path = GRAPHS_DIR / "morning_commute_weekday_v2.pkl"
    afternoon_path = GRAPHS_DIR / "afternoon_commute_weekday_v2.pkl"

    print("\nLoading graphs...")
    morning_G = load_graph(morning_path)
    afternoon_G = load_graph(afternoon_path)

    # Analyze morning
    morning_weights, morning_results = analyze_edge_weights(morning_G, "Morning Commute")
    morning_threshold = find_optimal_threshold(morning_results, target_flow_pct=95)

    # Analyze afternoon
    afternoon_weights, afternoon_results = analyze_edge_weights(afternoon_G, "Afternoon Commute")
    afternoon_threshold = find_optimal_threshold(afternoon_results, target_flow_pct=95)

    # Create visualizations
    create_weight_distribution_plot(
        morning_weights,
        "Morning Commute (7:00-11:00 AM)",
        OUTPUT_DIR / "morning_edge_weight_distribution.png"
    )

    create_weight_distribution_plot(
        afternoon_weights,
        "Afternoon Commute (3:00-7:00 PM)",
        OUTPUT_DIR / "afternoon_edge_weight_distribution.png"
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*70}")

    print(f"\nMorning Commute:")
    print(f"  Suggested threshold (95% flow): {morning_threshold}")
    print(f"  This will reduce edges by ~{100 - (morning_results[[r['threshold'] for r in morning_results].index(morning_threshold)]['pct_edges'])}% while retaining 95% of flow")

    print(f"\nAfternoon Commute:")
    print(f"  Suggested threshold (95% flow): {afternoon_threshold}")
    print(f"  This will reduce edges by ~{100 - (afternoon_results[[r['threshold'] for r in afternoon_results].index(afternoon_threshold)]['pct_edges'])}% while retaining 95% of flow")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"\nTo create trimmed graphs:")
    print(f"  1. Review the edge weight distribution plots")
    print(f"  2. Create trimmed versions with selected threshold")
    print(f"  3. Regenerate visualizations with trimmed graphs")
    print(f"\nSuggested command:")
    print(f"  uv run scripts/create_trimmed_graphs.py --morning-threshold {morning_threshold} --afternoon-threshold {afternoon_threshold}")


if __name__ == "__main__":
    main()
