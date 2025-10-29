"""
Analyze 7-Day Weekday Cycle

Creates cumulative and aggregate plots grouped by day (0-74) mod 7
to identify weekend days and Friday patterns.
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

def load_data():
    """Load cityD dataset"""
    print("Loading cityD-dataset.csv...")
    df = pd.read_csv(DATA_DIR / "cityD-dataset.csv")

    # Add day-of-week mod 7
    df['dow'] = df['d'] % 7

    return df

def analyze_cumulative_by_dow(df):
    """Calculate cumulative statistics for each day-of-week position"""
    print("\nCalculating cumulative statistics by day-of-week position...")

    # Group by day-of-week and calculate statistics
    daily_stats = df.groupby('d').agg({
        'uid': ['count', 'nunique'],
    }).reset_index()

    daily_stats.columns = ['d', 'total_pings', 'unique_users']
    daily_stats['dow'] = daily_stats['d'] % 7
    daily_stats['is_anomaly'] = daily_stats['d'] == 27

    # Calculate cumulative sums by day-of-week
    dow_cumulative = daily_stats.groupby('dow').agg({
        'total_pings': 'sum',
        'unique_users': 'sum',
    }).reset_index()

    dow_cumulative.columns = ['dow', 'cumulative_pings', 'cumulative_users']

    # Calculate average per occurrence for each day-of-week
    dow_count = daily_stats.groupby('dow').size().reset_index(name='num_occurrences')

    dow_stats = dow_cumulative.merge(dow_count, on='dow')
    dow_stats['avg_pings_per_day'] = dow_stats['cumulative_pings'] / dow_stats['num_occurrences']
    dow_stats['avg_users_per_day'] = dow_stats['cumulative_users'] / dow_stats['num_occurrences']

    return daily_stats, dow_stats

def print_detailed_analysis(daily_stats, dow_stats):
    """Print detailed analysis of day-of-week patterns"""
    print("\n" + "="*100)
    print("DAY-OF-WEEK ANALYSIS (Position in 7-day cycle: 0=first day, 1=second day, etc.)")
    print("="*100)

    day_names = ['Position 0', 'Position 1', 'Position 2', 'Position 3',
                 'Position 4', 'Position 5', 'Position 6']

    print("\nCUMULATIVE STATISTICS (sum of all occurrences):")
    print("-" * 100)
    print(f"{'Position':<12} | {'Occurrences':<12} | {'Total Pings':<15} | {'Avg Pings/Day':<15} | {'Total Users':<15} | {'Avg Users/Day':<15}")
    print("-" * 100)

    for idx, row in dow_stats.iterrows():
        dow = int(row['dow'])
        occurrences = int(row['num_occurrences'])
        total_pings = int(row['cumulative_pings'])
        avg_pings = int(row['avg_pings_per_day'])
        total_users = int(row['cumulative_users'])
        avg_users = int(row['avg_users_per_day'])

        print(f"{day_names[dow]:<12} | {occurrences:<12} | {total_pings:>14,} | {avg_pings:>14,} | {total_users:>14,} | {avg_users:>14,}")

    print("\n" + "="*100)
    print("IDENTIFYING PATTERN")
    print("="*100)

    # Find high and low days
    dow_stats_sorted = dow_stats.sort_values('avg_pings_per_day', ascending=False)

    print("\nRanking by average pings per day (highest to lowest):")
    print("-" * 100)
    for idx, (_, row) in enumerate(dow_stats_sorted.iterrows(), 1):
        dow = int(row['dow'])
        avg_pings = int(row['avg_pings_per_day'])
        pct_of_max = (row['avg_pings_per_day'] / dow_stats['avg_pings_per_day'].max()) * 100
        print(f"{idx}. {day_names[dow]:<12} - {avg_pings:>12,} pings ({pct_of_max:>5.1f}% of max)")

    # Calculate differences
    print("\n" + "-" * 100)
    print("PERCENTAGE DIFFERENCE FROM HIGHEST ACTIVITY DAY:")
    print("-" * 100)

    max_pings = dow_stats['avg_pings_per_day'].max()

    for idx in range(7):
        row = dow_stats[dow_stats['dow'] == idx].iloc[0]
        avg_pings = row['avg_pings_per_day']
        pct_diff = ((avg_pings - max_pings) / max_pings) * 100
        status = ""
        if pct_diff > -5:
            status = " ← WEEKDAY (similar to peak)"
        elif pct_diff < -20:
            status = " ← LIKELY WEEKEND (significantly lower)"
        elif pct_diff < -10:
            status = " ← FRIDAY? (moderately lower)"

        print(f"{day_names[idx]:<12} - {pct_diff:>6.1f}% lower than peak{status}")

    # Identify pattern
    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)

    sorted_days = dow_stats.sort_values('avg_pings_per_day', ascending=False)['dow'].values

    weekday_threshold = max_pings * 0.85  # Within 15% of max
    friday_threshold = max_pings * 0.75   # 25% lower
    weekend_threshold = max_pings * 0.60  # 40% lower

    weekdays = []
    fridays = []
    weekends = []

    for idx in range(7):
        row = dow_stats[dow_stats['dow'] == idx].iloc[0]
        avg_pings = row['avg_pings_per_day']

        if avg_pings >= weekday_threshold:
            weekdays.append(idx)
        elif avg_pings >= friday_threshold:
            fridays.append(idx)
        else:
            weekends.append(idx)

    print(f"\nHigh activity positions (weekdays): {weekdays}")
    print(f"Medium activity position (Friday?): {fridays}")
    print(f"Low activity positions (weekend): {weekends}")

    if len(weekends) >= 2:
        print(f"\nLikely weekend positions: {weekends}")
        if len(fridays) > 0:
            print(f"Possible Friday position: {fridays}")

    return weekdays, fridays, weekends

def create_cumulative_visualization(daily_stats, dow_stats):
    """Create cumulative and aggregate visualizations"""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Cumulative pings by day-of-week
    ax = axes[0, 0]
    # Position 0 = Sunday (weekend), Positions 1-5 = Weekdays, Position 6 = Saturday (weekend)
    colors = ['#d62728' if i in [0, 6] else '#1f77b4'
              for i in range(7)]

    dow_stats_sorted = dow_stats.sort_values('dow')
    bars1 = ax.bar(dow_stats_sorted['dow'], dow_stats_sorted['cumulative_pings'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_xlabel('Day-of-Week Position (0-6)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Pings (across all 75 days)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Pings by Day-of-Week Position', fontsize=14, fontweight='bold')
    ax.set_xticks(range(7))
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Average pings per day by day-of-week
    ax = axes[0, 1]
    bars2 = ax.bar(dow_stats_sorted['dow'], dow_stats_sorted['avg_pings_per_day'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_xlabel('Day-of-Week Position (0-6)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Pings per Day', fontsize=12, fontweight='bold')
    ax.set_title('Average Pings per Day by Day-of-Week Position', fontsize=14, fontweight='bold')
    ax.set_xticks(range(7))
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Time series with day-of-week coloring
    ax = axes[1, 0]

    daily_stats_clean = daily_stats[~daily_stats['is_anomaly']].copy()

    for dow in range(7):
        subset = daily_stats_clean[daily_stats_clean['dow'] == dow]
        ax.scatter(subset['d'], subset['total_pings'],
                  color=colors[dow], s=100, alpha=0.6, label=f'Position {dow}',
                  edgecolors='black', linewidth=0.5)

    ax.axvline(x=27, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Day 27 (anomaly)')
    ax.set_xlabel('Day (0-74)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Pings', fontsize=12, fontweight='bold')
    ax.set_title('Daily Pings Colored by Day-of-Week Position', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Box plot by day-of-week
    ax = axes[1, 1]

    data_by_dow = [daily_stats_clean[daily_stats_clean['dow'] == i]['total_pings'].values
                   for i in range(7)]

    bp = ax.boxplot(data_by_dow, labels=[f'Pos {i}' for i in range(7)], patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Day-of-Week Position (0-6)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Pings', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Daily Pings by Day-of-Week Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'weekday_cycle_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'weekday_cycle_analysis.png'}")
    plt.close()

def generate_weekend_days(weekends, fridays):
    """Generate the list of weekend days from identified positions"""
    print("\n" + "="*100)
    print("WEEKEND DAY CONFIGURATION")
    print("="*100)

    if len(weekends) < 2:
        print("\nCould not identify clear weekend pattern.")
        print("Please review the visualizations and manually identify weekend positions.")
        return None

    # Map day positions to actual days
    weekend_days = []

    for day in range(75):
        if day == 27:  # Skip anomaly
            continue
        dow = day % 7
        if dow in weekends:
            weekend_days.append(day)

    print(f"\nIdentified weekend positions: {weekends}")

    if fridays:
        friday_days = [day for day in range(75) if day % 7 in fridays and day != 27]
        print(f"Identified Friday positions: {fridays}")
        print(f"Friday days in dataset: {friday_days}")

    print(f"\nSuggested weekend_days configuration:")
    print("-" * 100)
    print(f"weekend_days: {weekend_days}")
    print(f"anomalous_days: [27]")

    print("\nUpdate config/day_classification.yaml with:")
    print(f"weekend_days: {weekend_days}")
    print(f"anomalous_days: [27]")

    return weekend_days

def main():
    """Run weekday cycle analysis"""
    print("="*100)
    print("7-DAY WEEKDAY CYCLE ANALYSIS")
    print("="*100)

    # Load data
    df = load_data()

    # Analyze
    daily_stats, dow_stats = analyze_cumulative_by_dow(df)

    # Print analysis
    weekdays, fridays, weekends = print_detailed_analysis(daily_stats, dow_stats)

    # Visualize
    create_cumulative_visualization(daily_stats, dow_stats)

    # Generate configuration
    weekend_days = generate_weekend_days(weekends, fridays)

    print("\n" + "="*100)
    print("NEXT STEPS")
    print("="*100)
    print("\n1. Review the visualization: analysis/weekday_cycle_analysis.png")
    print("2. Verify the identified pattern matches what you see")
    print("3. Check the suggested weekend_days and anomalous_days above")
    print("4. Update config/day_classification.yaml with the configuration")
    print("5. Run preprocessing: uv run scripts/run_pipeline.py")

if __name__ == "__main__":
    main()
