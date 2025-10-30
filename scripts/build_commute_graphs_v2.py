"""
Build Commute Graphs (Version 2)

Creates NetworkX graphs for morning and afternoon commutes.
This version preserves the full 200x200 grid and all flows,
treating each (source, target, timeslot) combination separately
or aggregating them properly.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
import pickle

# Configuration
INTERMEDIATE_DIR = Path(__file__).parent.parent / "intermediate"
OUTPUT_DIR = Path(__file__).parent.parent / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Timeslot definitions
MORNING_TIMESLOTS = list(range(14, 22))  # 7:00-11:00 (timeslots 14-21)
AFTERNOON_TIMESLOTS = list(range(30, 38))  # 15:00-19:00 (timeslots 30-37)


def load_flows():
    """Load aggregated flows from intermediate data"""
    print("Loading cell flows...")
    flows = pd.read_parquet(INTERMEDIATE_DIR / "cell_flows.parquet")
    print(f"Loaded {len(flows):,} flows")
    return flows


def filter_flows_for_period(flows, timeslots, day_type='weekday'):
    """Filter flows for a specific time period and day type"""
    filtered = flows[
        (flows['timeslot'].isin(timeslots)) &
        (flows['day_type'] == day_type)
    ].copy()
    return filtered


def build_graph_from_flows(flows, graph_name="graph"):
    """
    Build a NetworkX directed graph from flows.

    Each row in flows represents (source, target, timeslot, day_type, flow_count).
    We aggregate across timeslots to create a single graph showing total flow
    between each (source, target) pair across all timeslots.
    """
    print(f"\nBuilding {graph_name}...")

    # First, let's aggregate: sum flow_count for each unique (source, target) pair
    # This treats the graph as day-level, ignoring timeslots
    aggregated = flows.groupby(['source_cell_x', 'source_cell_y',
                                'target_cell_x', 'target_cell_y'])['flow_count'].sum().reset_index()

    print(f"  Unique (source, target) pairs: {len(aggregated):,}")

    # Create directed graph
    G = nx.DiGraph()

    # Add edges with aggregated flow as weight
    for _, row in aggregated.iterrows():
        source = (int(row['source_cell_x']), int(row['source_cell_y']))
        target = (int(row['target_cell_x']), int(row['target_cell_y']))
        flow_weight = int(row['flow_count'])

        G.add_edge(source, target, weight=flow_weight)

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Total flow: {sum(data['weight'] for _, _, data in G.edges(data=True)):,}")

    # Verify grid size
    all_nodes = list(G.nodes())
    x_coords = [n[0] for n in all_nodes]
    y_coords = [n[1] for n in all_nodes]
    print(f"  Grid coverage: X [{min(x_coords)}-{max(x_coords)}] Y [{min(y_coords)}-{max(y_coords)}]")

    return G


def print_graph_stats(G, graph_name="graph"):
    """Print statistics about a graph"""
    print(f"\n{graph_name.upper()} STATISTICS")
    print("=" * 70)
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    print(f"Density: {nx.density(G):.6f}")

    # Get weight statistics
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    print(f"\nEdge Weight Statistics:")
    print(f"  Min: {min(weights)}")
    print(f"  Max: {max(weights)}")
    print(f"  Mean: {sum(weights) / len(weights):.1f}")
    print(f"  Total: {sum(weights):,}")

    # In/out degree statistics
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]

    print(f"\nDegree Statistics:")
    print(f"  Avg in-degree: {sum(in_degrees) / len(in_degrees):.1f}")
    print(f"  Avg out-degree: {sum(out_degrees) / len(out_degrees):.1f}")

    # Check connectivity
    if nx.is_weakly_connected(G):
        print(f"  Weakly connected: Yes")
    else:
        num_components = nx.number_weakly_connected_components(G)
        print(f"  Weakly connected: No ({num_components} components)")

    if nx.is_strongly_connected(G):
        print(f"  Strongly connected: Yes")
    else:
        num_components = nx.number_strongly_connected_components(G)
        print(f"  Strongly connected: No ({num_components} components)")


def main():
    """Build commute graphs"""
    print("=" * 70)
    print("BUILDING COMMUTE GRAPHS (v2 - Proper Aggregation)")
    print("=" * 70)

    # Load flows
    flows = load_flows()

    print(f"\nTimeslot Definitions:")
    print(f"  Morning commute: timeslots {MORNING_TIMESLOTS} (7:00-11:00 AM)")
    print(f"  Afternoon commute: timeslots {AFTERNOON_TIMESLOTS} (3:00-7:00 PM)")

    # Build morning commute graph (weekday)
    print("\n" + "=" * 70)
    print("MORNING COMMUTE (WEEKDAY)")
    print("=" * 70)

    morning_flows = filter_flows_for_period(flows, MORNING_TIMESLOTS, 'weekday')
    print(f"Input flows: {len(morning_flows):,}")
    print(f"Total flow count: {morning_flows['flow_count'].sum():,}")

    morning_graph = build_graph_from_flows(morning_flows, "morning_commute")
    print_graph_stats(morning_graph, "morning_commute")

    # Save morning graph
    morning_path = OUTPUT_DIR / "morning_commute_weekday_v2.pkl"
    with open(morning_path, 'wb') as f:
        pickle.dump(morning_graph, f)
    print(f"\nSaved: {morning_path}")

    # Build afternoon commute graph (weekday)
    print("\n" + "=" * 70)
    print("AFTERNOON COMMUTE (WEEKDAY)")
    print("=" * 70)

    afternoon_flows = filter_flows_for_period(flows, AFTERNOON_TIMESLOTS, 'weekday')
    print(f"Input flows: {len(afternoon_flows):,}")
    print(f"Total flow count: {afternoon_flows['flow_count'].sum():,}")

    afternoon_graph = build_graph_from_flows(afternoon_flows, "afternoon_commute")
    print_graph_stats(afternoon_graph, "afternoon_commute")

    # Save afternoon graph
    afternoon_path = OUTPUT_DIR / "afternoon_commute_weekday_v2.pkl"
    with open(afternoon_path, 'wb') as f:
        pickle.dump(afternoon_graph, f)
    print(f"\nSaved: {afternoon_path}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nMorning vs Afternoon:")
    print(f"  Nodes: {morning_graph.number_of_nodes():,} vs {afternoon_graph.number_of_nodes():,}")
    print(f"  Edges: {morning_graph.number_of_edges():,} vs {afternoon_graph.number_of_edges():,}")

    morning_total = sum(data['weight'] for _, _, data in morning_graph.edges(data=True))
    afternoon_total = sum(data['weight'] for _, _, data in afternoon_graph.edges(data=True))
    print(f"  Total flow: {morning_total:,} vs {afternoon_total:,}")
    print(f"  Ratio (morning/afternoon): {morning_total/afternoon_total:.2f}x")


if __name__ == "__main__":
    main()
