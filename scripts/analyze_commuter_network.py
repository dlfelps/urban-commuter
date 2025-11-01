"""
Comprehensive Network Analysis of Commuter Transit Routes

Performs multiple centrality measures, community detection, and flow analysis
to identify critical hubs, communities, and congestion corridors in the
commuter transit network.
"""

import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

MORNING_GRAPH = "morning_commute_weekday_full.pkl"
AFTERNOON_GRAPH = "afternoon_commute_weekday_full.pkl"
COMMUTER_GRAPH = "commuter_transit_routes.pkl"

print("=" * 80)
print("COMPREHENSIVE NETWORK ANALYSIS OF COMMUTER TRANSIT ROUTES")
print("=" * 80)

# Load commuter transit routes graph
commuter_path = GRAPHS_DIR / COMMUTER_GRAPH
if not commuter_path.exists():
    print(f"\n[ERROR] Commuter routes graph not found: {commuter_path}")
    exit(1)

print(f"\nLoading {commuter_path.name}...")
with open(commuter_path, 'rb') as f:
    G = pickle.load(f)

print(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Convert to undirected for some analyses
G_undirected = G.to_undirected()
print(f"Undirected version: {G_undirected.number_of_nodes():,} nodes, {G_undirected.number_of_edges():,} edges")

# ============================================================================
# 1. CENTRALITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("1. CENTRALITY ANALYSIS")
print("=" * 80)

centrality_results = {}

# Degree Centrality (undirected)
print("\nCalculating degree centrality...")
degree_cent = dict(G_undirected.degree(weight='weight'))
centrality_results['degree'] = degree_cent

# Weighted degree (strength)
print("Calculating weighted degree centrality...")
weighted_degree = dict(G_undirected.degree(weight='weight'))

# Largest connected component for algorithms that require it
largest_cc = max(nx.connected_components(G_undirected), key=len)
G_largest = G_undirected.subgraph(largest_cc).copy()
print(f"Largest connected component: {len(largest_cc):,} nodes ({len(largest_cc)/G_undirected.number_of_nodes()*100:.1f}%)")

# Eigenvector Centrality (undirected) - on largest component
print("Calculating eigenvector centrality...")
try:
    eigenvector_cent = nx.eigenvector_centrality_numpy(G_largest, weight='weight', max_iter=500)
    eigenvector_full = {node: eigenvector_cent.get(node, 0) for node in G_undirected.nodes()}
except:
    print("  (Eigenvector centrality failed)")
    eigenvector_full = {node: 0 for node in G_undirected.nodes()}
centrality_results['eigenvector'] = eigenvector_full

# PageRank (works on directed or undirected)
print("Calculating PageRank...")
pagerank = nx.pagerank(G, weight='weight')
centrality_results['pagerank'] = pagerank

# Find top nodes for each centrality measure
print("\n" + "-" * 80)
print("TOP 10 NODES BY CENTRALITY MEASURE:")
print("-" * 80)

for measure_name, centrality_dict in centrality_results.items():
    top_10 = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n{measure_name.upper()}:")
    for rank, (node, value) in enumerate(top_10, 1):
        print(f"  {rank:2d}. {node}: {value:.6f}")

# Save centrality results to CSV
centrality_df = pd.DataFrame(centrality_results)
centrality_csv = OUTPUT_DIR / "centrality_analysis.csv"
centrality_df.to_csv(centrality_csv)
print(f"\n[OK] Saved centrality analysis to {centrality_csv.name}")

# ============================================================================
# 2. COMMUNITY DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("2. COMMUNITY DETECTION (LOUVAIN ALGORITHM)")
print("=" * 80)

print("\nDetecting communities...")
import networkx.algorithms.community as nx_community

try:
    # Louvain community detection
    communities = list(nx_community.louvain_communities(G_undirected, weight='weight', seed=42))
    print(f"Found {len(communities)} communities")

    # Create community mapping
    node_to_community = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = comm_id

    print(f"\nCommunity size distribution:")
    community_sizes = [len(c) for c in communities]
    for i, size in enumerate(sorted(community_sizes, reverse=True)[:10]):
        print(f"  Community {i+1}: {size} nodes")

    # Calculate community-level statistics
    print(f"\nCommunity-level statistics:")
    community_stats = []
    for comm_id, comm in enumerate(communities):
        subgraph = G_undirected.subgraph(comm)
        edges = subgraph.number_of_edges()
        total_weight = sum(d['weight'] for u, v, d in subgraph.edges(data=True)) if edges > 0 else 0
        degrees = [d for n, d in subgraph.degree(weight='weight')]
        avg_degree = sum(degrees) / len(comm) if len(comm) > 0 else 0

        community_stats.append({
            'community_id': comm_id,
            'num_nodes': len(comm),
            'num_edges': edges,
            'total_weight': total_weight,
            'avg_degree': avg_degree,
            'density': nx.density(subgraph)
        })

    community_stats_df = pd.DataFrame(community_stats)
    community_stats_csv = OUTPUT_DIR / "community_statistics.csv"
    community_stats_df.to_csv(community_stats_csv, index=False)
    print(f"\n[OK] Saved community statistics to {community_stats_csv.name}")

    # Save node-to-community mapping
    node_community_df = pd.DataFrame(list(node_to_community.items()), columns=['node', 'community_id'])
    node_community_csv = OUTPUT_DIR / "node_community_mapping.csv"
    node_community_df.to_csv(node_community_csv, index=False)
    print(f"[OK] Saved node-community mapping to {node_community_csv.name}")

except Exception as e:
    print(f"[ERROR] Community detection failed: {e}")
    communities = []
    node_to_community = {}

# ============================================================================
# 3. FLOW AND CORRIDOR ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. FLOW AND CORRIDOR ANALYSIS")
print("=" * 80)

print("\nAnalyzing edge weights and flows...")

# Analyze edge weights
edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
edge_weights = np.array(edge_weights)

print(f"\nEdge weight statistics:")
print(f"  Count: {len(edge_weights):,}")
print(f"  Total flow: {edge_weights.sum():,.0f}")
print(f"  Min: {edge_weights.min():.2f}")
print(f"  Max: {edge_weights.max():.2f}")
print(f"  Mean: {edge_weights.mean():.2f}")
print(f"  Median: {np.median(edge_weights):.2f}")
print(f"  Std Dev: {edge_weights.std():.2f}")
print(f"  90th percentile: {np.percentile(edge_weights, 90):.2f}")
print(f"  95th percentile: {np.percentile(edge_weights, 95):.2f}")
print(f"  99th percentile: {np.percentile(edge_weights, 99):.2f}")

# Identify high-flow corridors (top edges by weight)
print(f"\nTop 20 highest-flow corridors:")
edges_by_weight = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
for rank, (source, target, attr) in enumerate(edges_by_weight[:20], 1):
    print(f"  {rank:2d}. {source} -> {target}: {attr['weight']:.0f}")

# Create corridor dataframe
corridors_data = []
for source, target, attr in G.edges(data=True):
    distance = np.sqrt((source[0] - target[0])**2 + (source[1] - target[1])**2)
    corridors_data.append({
        'source_x': source[0],
        'source_y': source[1],
        'target_x': target[0],
        'target_y': target[1],
        'flow': attr['weight'],
        'distance': distance,
        'flow_per_unit_distance': attr['weight'] / distance if distance > 0 else 0
    })

corridors_df = pd.DataFrame(corridors_data)
corridors_csv = OUTPUT_DIR / "corridors_analysis.csv"
corridors_df.to_csv(corridors_csv, index=False)
print(f"\n[OK] Saved corridor analysis to {corridors_csv.name}")

# ============================================================================
# 4. CONNECTIVITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. CONNECTIVITY ANALYSIS")
print("=" * 80)

# Connected components
components = list(nx.connected_components(G_undirected))
print(f"\nConnected components: {len(components)}")
print(f"Largest component size: {len(max(components, key=len)):,} nodes ({len(max(components, key=len))/G.number_of_nodes()*100:.1f}%)")
print(f"Smallest component size: {len(min(components, key=len))} nodes")

# Average clustering coefficient
clustering = nx.average_clustering(G_undirected, weight='weight')
print(f"\nAverage clustering coefficient: {clustering:.6f}")

# Transitivity (global clustering coefficient)
transitivity = nx.transitivity(G_undirected)
print(f"Transitivity (global clustering): {transitivity:.6f}")

# ============================================================================
# 5. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("5. CREATING VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Centrality distributions
print("\nCreating centrality distribution plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Centrality Measure Distributions', fontsize=16, fontweight='bold')

centrality_measures = ['degree', 'eigenvector', 'pagerank']
for idx, (ax, measure) in enumerate(zip(axes.flat, centrality_measures)):
    values = list(centrality_results[measure].values())
    values = [v for v in values if v > 0]  # Filter zeros

    ax.hist(values, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Centrality Value', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(measure.upper(), fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
viz_path1 = OUTPUT_DIR / "centrality_distributions.png"
plt.savefig(viz_path1, dpi=150, bbox_inches='tight')
print(f"Saved: {viz_path1.name}")
plt.close()

# Visualization 2: Top nodes comparison
print("Creating top nodes comparison plot...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Top 15 Nodes by Centrality Measure', fontsize=16, fontweight='bold')

for idx, (ax, measure) in enumerate(zip(axes.flat, centrality_measures)):
    top_15 = sorted(centrality_results[measure].items(), key=lambda x: x[1], reverse=True)[:15]
    nodes_labels = [f"{n[0][0]},{n[0][1]}" for n in top_15]
    values = [n[1] for n in top_15]

    ax.barh(range(len(nodes_labels)), values, color='coral', edgecolor='black')
    ax.set_yticks(range(len(nodes_labels)))
    ax.set_yticklabels(nodes_labels, fontsize=8)
    ax.set_xlabel('Centrality Value', fontsize=10, fontweight='bold')
    ax.set_title(measure.upper(), fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

plt.tight_layout()
viz_path2 = OUTPUT_DIR / "top_nodes_comparison.png"
plt.savefig(viz_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {viz_path2.name}")
plt.close()

# Visualization 3: Edge weight distribution
print("Creating edge weight analysis plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Edge Weight Distribution Analysis', fontsize=16, fontweight='bold')

# Linear histogram
ax = axes[0, 0]
ax.hist(edge_weights, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Edge Weight', fontsize=10, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax.set_title('Linear Scale', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Log-scale histogram
ax = axes[0, 1]
ax.hist(edge_weights, bins=100, color='coral', alpha=0.7, edgecolor='black')
ax.set_yscale('log')
ax.set_xlabel('Edge Weight', fontsize=10, fontweight='bold')
ax.set_ylabel('Frequency (log)', fontsize=10, fontweight='bold')
ax.set_title('Log Scale', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Cumulative distribution
ax = axes[1, 0]
sorted_weights = np.sort(edge_weights)
cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
ax.plot(sorted_weights, cumulative, linewidth=2, color='darkgreen')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
ax.set_xlabel('Edge Weight', fontsize=10, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
ax.set_title('Cumulative Distribution', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Percentile breakdown
ax = axes[1, 1]
percentiles = [10, 25, 50, 75, 90, 95, 99]
percentile_values = [np.percentile(edge_weights, p) for p in percentiles]
colors_bar = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
bars = ax.barh([f'{p}th' for p in percentiles], percentile_values, color=colors_bar, edgecolor='black')
ax.set_xlabel('Edge Weight', fontsize=10, fontweight='bold')
ax.set_title('Percentile Breakdown', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, percentile_values):
    ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.1f}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
viz_path3 = OUTPUT_DIR / "edge_weight_analysis.png"
plt.savefig(viz_path3, dpi=150, bbox_inches='tight')
print(f"Saved: {viz_path3.name}")
plt.close()

# Visualization 4: Community visualization (if communities detected)
if communities:
    print("Creating community statistics plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Community Detection Results', fontsize=16, fontweight='bold')

    # Community sizes
    ax = axes[0, 0]
    community_sizes = sorted([len(c) for c in communities], reverse=True)
    ax.bar(range(min(20, len(community_sizes))), community_sizes[:20], color='steelblue', edgecolor='black')
    ax.set_xlabel('Community Rank', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Nodes', fontsize=10, fontweight='bold')
    ax.set_title('Community Sizes (Top 20)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Community total weight
    ax = axes[0, 1]
    community_weights = []
    for comm in communities:
        subgraph = G_undirected.subgraph(comm)
        total_weight = sum(d['weight'] for u, v, d in subgraph.edges(data=True))
        community_weights.append(total_weight)

    community_weights_sorted = sorted(community_weights, reverse=True)
    ax.bar(range(min(20, len(community_weights_sorted))), community_weights_sorted[:20],
           color='coral', edgecolor='black')
    ax.set_xlabel('Community Rank', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total Edge Weight', fontsize=10, fontweight='bold')
    ax.set_title('Total Community Flow (Top 20)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Community density
    ax = axes[1, 0]
    community_densities = []
    for comm in communities:
        subgraph = G_undirected.subgraph(comm)
        density = nx.density(subgraph)
        community_densities.append(density)

    ax.hist(community_densities, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Density', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Community Density Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Number of communities
    ax = axes[1, 1]
    ax.text(0.5, 0.7, f'Total Communities: {len(communities)}', ha='center', va='center',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.55, f'Avg Community Size: {np.mean(community_sizes):.0f}', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.40, f'Avg Density: {np.mean(community_densities):.4f}', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.25, f'Modularity: {nx_community.modularity(G_undirected, communities, weight="weight"):.4f}',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.axis('off')

    plt.tight_layout()
    viz_path4 = OUTPUT_DIR / "community_analysis.png"
    plt.savefig(viz_path4, dpi=150, bbox_inches='tight')
    print(f"Saved: {viz_path4.name}")
    plt.close()

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print(f"\nGraph Statistics:")
print(f"  Nodes: {G.number_of_nodes():,}")
print(f"  Edges: {G.number_of_edges():,}")
print(f"  Total Flow: {edge_weights.sum():,.0f}")
print(f"  Average Degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
print(f"  Density: {nx.density(G_undirected):.6f}")

if communities:
    print(f"\nCommunity Detection:")
    print(f"  Communities Found: {len(communities)}")
    print(f"  Largest Community: {len(max(communities, key=len)):,} nodes")
    print(f"  Smallest Community: {len(min(communities, key=len))} nodes")
    modularity = nx_community.modularity(G_undirected, communities, weight='weight')
    print(f"  Modularity Score: {modularity:.4f}")

print(f"\nTop Hub (by PageRank):")
top_hub = max(pagerank.items(), key=lambda x: x[1])
print(f"  Node: {top_hub[0]}")
print(f"  PageRank: {top_hub[1]:.6f}")

print(f"\nTop Corridor (by flow):")
top_corridor = edges_by_weight[0]
print(f"  {top_corridor[0]} -> {top_corridor[1]}: {top_corridor[2]['weight']:.0f}")

print(f"\n[OK] Network analysis complete!")
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print(f"  - centrality_analysis.csv")
print(f"  - community_statistics.csv")
print(f"  - node_community_mapping.csv")
print(f"  - corridors_analysis.csv")
print(f"  - centrality_distributions.png")
print(f"  - top_nodes_comparison.png")
print(f"  - edge_weight_analysis.png")
if communities:
    print(f"  - community_analysis.png")
