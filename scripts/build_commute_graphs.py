"""
Build Commute Graphs

Creates NetworkX graphs for:
- Morning commute (weekday): specific timeslots
- Afternoon commute (weekday): specific timeslots

Each graph represents cell-to-cell movement patterns during that period.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
import pickle

# Configuration
INTERMEDIATE_DIR = Path(__file__).parent.parent / "intermediate"
OUTPUT_DIR = Path(__file__).parent.parent / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Timeslot definitions (30-minute bins, 0-47 covering 48 hours)
# 0-47 timeslots = 0:00 to 24:00 (24 hours)
# Each timeslot is 30 minutes
# Timeslot 0: 00:00-00:30
# Timeslot 1: 00:30-01:00
# ...
# Timeslot 14: 07:00-07:30 (morning start)
# Timeslot 15: 07:30-08:00
# Timeslot 16: 08:00-08:30
# Timeslot 17: 08:30-09:00
# ...
# Timeslot 22: 11:00-11:30 (morning end)
# Timeslot 23: 11:30-12:00 (noon)
# ...
# Timeslot 30: 15:00-15:30 (afternoon start)
# Timeslot 31: 15:30-16:00
# ...
# Timeslot 35: 17:30-18:00 (afternoon end)

# Morning commute: 7:00 AM - 11:00 AM (4 hours = 8 timeslots)
MORNING_TIMESLOTS = list(range(14, 22))  # 7:00-11:00 (timeslots 14-21)

# Afternoon commute: 3:00 PM - 7:00 PM (4 hours = 8 timeslots)
AFTERNOON_TIMESLOTS = list(range(30, 38))  # 15:00-19:00 (timeslots 30-37)


def load_flows():
    """Load aggregated flows from intermediate data"""
    print("Loading cell flows...")
    flows = pd.read_parquet(INTERMEDIATE_DIR / "cell_flows.parquet")
    print(f"Loaded {len(flows):,} flows")
    return flows


def filter_flows_for_period(flows, timeslots, day_type='weekday'):
    """
    Filter flows for a specific time period and day type

    Args:
        flows: DataFrame from cell_flows.parquet
        timeslots: List of timeslot indices
        day_type: 'weekday', 'weekend', or 'all'

    Returns:
        Filtered DataFrame
    """
    filtered = flows[
        (flows['timeslot'].isin(timeslots)) &
        (flows['day_type'] == day_type)
    ].copy()

    return filtered


def build_graph_from_flows(flows, graph_name="graph"):
    """
    Build a NetworkX directed graph from flows

    Args:
        flows: DataFrame with columns: source_cell_x, source_cell_y,
               target_cell_x, target_cell_y, flow_count
        graph_name: Name for the graph

    Returns:
        NetworkX DiGraph
    """
    print(f"\nBuilding {graph_name}...")

    # Create directed graph
    G = nx.DiGraph()

    # Add edges with flow_count as weight
    for _, row in flows.iterrows():
        source = (int(row['source_cell_x']), int(row['source_cell_y']))
        target = (int(row['target_cell_x']), int(row['target_cell_y']))
        flow_count = int(row['flow_count'])

        # Add edge with weight
        if G.has_edge(source, target):
            # If edge exists, add to the weight
            G[source][target]['weight'] += flow_count
        else:
            G.add_edge(source, target, weight=flow_count)

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Total flow: {sum(data['weight'] for _, _, data in G.edges(data=True)):,}")

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

    # Check if graph is connected
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
    print("BUILDING COMMUTE GRAPHS")
    print("=" * 70)

    # Load flows
    flows = load_flows()

    # Define timeslot ranges
    print(f"\nTimeslot Definitions:")
    print(f"  Morning commute (7:00-11:30 AM): timeslots {MORNING_TIMESLOTS}")
    print(f"  Afternoon commute (3:00-6:00 PM): timeslots {AFTERNOON_TIMESLOTS}")

    # Build morning commute graph (weekday)
    print("\n" + "=" * 70)
    print("MORNING COMMUTE (WEEKDAY)")
    print("=" * 70)

    morning_flows = filter_flows_for_period(flows, MORNING_TIMESLOTS, 'weekday')
    print(f"Aggregated flows: {len(morning_flows):,}")
    print(f"Total flow count: {morning_flows['flow_count'].sum():,}")

    morning_graph = build_graph_from_flows(morning_flows, "morning_commute")
    print_graph_stats(morning_graph, "morning_commute")

    # Save morning graph
    morning_path = OUTPUT_DIR / "morning_commute_weekday.pkl"
    with open(morning_path, 'wb') as f:
        pickle.dump(morning_graph, f)
    print(f"\nSaved: {morning_path}")

    # Build afternoon commute graph (weekday)
    print("\n" + "=" * 70)
    print("AFTERNOON COMMUTE (WEEKDAY)")
    print("=" * 70)

    afternoon_flows = filter_flows_for_period(flows, AFTERNOON_TIMESLOTS, 'weekday')
    print(f"Aggregated flows: {len(afternoon_flows):,}")
    print(f"Total flow count: {afternoon_flows['flow_count'].sum():,}")

    afternoon_graph = build_graph_from_flows(afternoon_flows, "afternoon_commute")
    print_graph_stats(afternoon_graph, "afternoon_commute")

    # Save afternoon graph
    afternoon_path = OUTPUT_DIR / "afternoon_commute_weekday.pkl"
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
