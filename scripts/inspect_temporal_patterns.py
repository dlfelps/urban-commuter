"""
Temporal Pattern Inspection Script

Analyzes daily and hourly patterns in mobility data to help identify weekdays vs weekends.
Generates visualizations and statistics to manually classify days.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Timeslot to hour mapping (30-minute bins)
# timeslot 0 = 00:00-00:30, timeslot 1 = 00:30-01:00, etc.
def timeslot_to_hour(timeslot):
    """Convert timeslot (0-47) to hour of day"""
    return (timeslot * 30) // 60

def load_data():
    """Load and validate cityA dataset"""
    print("Loading cityA-dataset.csv...")
    df = pd.read_csv(DATA_DIR / "cityA-dataset.csv")
    print(f"Loaded {len(df):,} records")
    print(f"Days: {df['d'].min()}-{df['d'].max()}")
    print(f"Timeslots: {df['t'].min()}-{df['t'].max()}")
    print(f"Users: {df['uid'].nunique():,}")
    print(f"Grid cells: {df[['x', 'y']].drop_duplicates().shape[0]:,}")
    return df

def analyze_daily_patterns(df):
    """Analyze aggregate statistics per day"""
    print("\nAnalyzing daily patterns...")

    daily_stats = df.groupby('d').agg({
        'uid': 'count',  # total pings
        'uid': 'nunique',  # unique users
    }).rename(columns={'uid': 'total_pings'})

    # Recalculate to get both metrics
    daily_stats = df.groupby('d').agg(
        total_pings=('uid', 'count'),
        unique_users=('uid', 'nunique'),
        unique_cells=('x', 'nunique')
    ).reset_index()

    # Flag day 27 as anomaly
    daily_stats['is_anomaly'] = daily_stats['d'] == 27

    return daily_stats

def analyze_timeslot_patterns(df):
    """Analyze patterns across timeslots (hours of day)"""
    print("Analyzing timeslot patterns...")

    timeslot_stats = df.groupby('t').agg(
        total_pings=('uid', 'count'),
        unique_users=('uid', 'nunique')
    ).reset_index()

    timeslot_stats['hour'] = timeslot_stats['t'].apply(timeslot_to_hour)

    return timeslot_stats

def analyze_day_and_timeslot_patterns(df):
    """Analyze combined day and timeslot patterns"""
    print("Analyzing day-timeslot patterns...")

    day_timeslot = df.groupby(['d', 't']).agg(
        total_pings=('uid', 'count'),
        unique_users=('uid', 'nunique')
    ).reset_index()

    day_timeslot['hour'] = day_timeslot['t'].apply(timeslot_to_hour)

    return day_timeslot

def plot_daily_trends(daily_stats):
    """Visualize daily trends to identify patterns"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Total pings per day
    ax = axes[0]
    ax.plot(daily_stats['d'], daily_stats['total_pings'], marker='o', linewidth=2, markersize=4)
    ax.axvline(x=27, color='red', linestyle='--', alpha=0.5, label='Day 27 (anomaly)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Total Pings')
    ax.set_title('Daily Ping Count - Look for 7-day cycles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Unique users per day
    ax = axes[1]
    ax.plot(daily_stats['d'], daily_stats['unique_users'], marker='o', linewidth=2, markersize=4, color='orange')
    ax.axvline(x=27, color='red', linestyle='--', alpha=0.5, label='Day 27 (anomaly)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Unique Users')
    ax.set_title('Daily Unique Users - Look for weekly patterns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Normalized pings to see pattern more clearly
    ax = axes[2]
    normalized = (daily_stats['total_pings'] - daily_stats['total_pings'].mean()) / daily_stats['total_pings'].std()
    ax.plot(daily_stats['d'], normalized, marker='o', linewidth=2, markersize=4, color='green')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=27, color='red', linestyle='--', alpha=0.5, label='Day 27 (anomaly)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Normalized Pings (z-score)')
    ax.set_title('Normalized Daily Pattern - Clearer view of cycles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'daily_trends.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'daily_trends.png'}")
    plt.close()

def plot_timeslot_patterns(timeslot_stats):
    """Visualize hourly patterns"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.plot(timeslot_stats['hour'], timeslot_stats['total_pings'], marker='o', linewidth=2)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Total Pings')
    ax.set_title('Average Hourly Pattern Across All Days')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(timeslot_stats['hour'], timeslot_stats['unique_users'], marker='o', linewidth=2, color='orange')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Unique Users')
    ax.set_title('Average Unique Users by Hour')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'timeslot_patterns.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'timeslot_patterns.png'}")
    plt.close()

def plot_heatmap_day_timeslot(day_timeslot):
    """Visualize day-timeslot heatmap"""
    # Pivot to create matrix
    pivot = day_timeslot.pivot_table(
        values='total_pings',
        index='d',
        columns='hour',
        aggfunc='sum'
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Total Pings'})
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day')
    ax.set_title('Daily-Hourly Activity Heatmap - Identify weekly patterns')

    # Mark day 27
    ax.axhline(y=27, color='cyan', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'day_timeslot_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'day_timeslot_heatmap.png'}")
    plt.close()

def print_daily_statistics(daily_stats):
    """Print detailed daily statistics"""
    print("\n" + "="*80)
    print("DAILY STATISTICS")
    print("="*80)
    print(daily_stats.to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Mean pings/day: {daily_stats[~daily_stats['is_anomaly']]['total_pings'].mean():.0f}")
    print(f"Std dev pings/day: {daily_stats[~daily_stats['is_anomaly']]['total_pings'].std():.0f}")
    print(f"Min pings/day (excl. day 27): {daily_stats[~daily_stats['is_anomaly']]['total_pings'].min():.0f}")
    print(f"Max pings/day (excl. day 27): {daily_stats[~daily_stats['is_anomaly']]['total_pings'].max():.0f}")

    # Look for 7-day patterns
    print("\n" + "="*80)
    print("GROUPING BY DAY OF WEEK (assuming 7-day cycle)")
    print("="*80)
    daily_stats['day_of_week'] = daily_stats['d'] % 7
    day_of_week_stats = daily_stats[~daily_stats['is_anomaly']].groupby('day_of_week').agg({
        'total_pings': ['mean', 'std'],
        'unique_users': ['mean', 'std']
    }).round(0)
    print(day_of_week_stats)

    return daily_stats

def main():
    """Main inspection workflow"""
    print("="*80)
    print("URBAN MOBILITY TEMPORAL PATTERN INSPECTION")
    print("="*80)

    # Load data
    df = load_data()

    # Analyze patterns
    daily_stats = analyze_daily_patterns(df)
    timeslot_stats = analyze_timeslot_patterns(df)
    day_timeslot = analyze_day_and_timeslot_patterns(df)

    # Print statistics
    daily_stats = print_daily_statistics(daily_stats)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_daily_trends(daily_stats)
    plot_timeslot_patterns(timeslot_stats)
    plot_heatmap_day_timeslot(day_timeslot)

    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)
    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
    print("\nNEXT STEPS:")
    print("1. Review the generated plots in ./analysis/")
    print("2. Look for 7-day cycles or other patterns")
    print("3. Identify which days appear to be weekends (lower activity)")
    print("4. Update config/day_classification.yaml with weekend day indices")
    print("\nHINT: Check the 'GROUPING BY DAY OF WEEK' table above.")
    print("Days with lower pings/unique_users are likely weekends.")

if __name__ == "__main__":
    main()
