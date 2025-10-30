"""
Identify Weekend Days - Enhanced Analysis

Analyzes the 7-day cycle in temporal patterns to definitively identify weekends.
Shows consecutive low-activity days and helps determine Saturday vs Sunday.
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
    """Load cityA dataset"""
    print("Loading cityA-dataset.csv...")
    df = pd.read_csv(DATA_DIR / "cityA-dataset.csv")
    return df

def analyze_day_patterns(df):
    """Analyze patterns by day of week (assuming 7-day cycle)"""
    print("\nAnalyzing 7-day cycle patterns...")

    # Calculate daily statistics
    daily_stats = df.groupby('d').agg(
        total_pings=('uid', 'count'),
        unique_users=('uid', 'nunique'),
    ).reset_index()

    # Assume 7-day cycle and calculate day-of-week
    daily_stats['day_of_week'] = daily_stats['d'] % 7
    daily_stats['week_number'] = daily_stats['d'] // 7
    daily_stats['is_anomaly'] = daily_stats['d'] == 27

    # Group by day-of-week to see average pattern
    dow_stats = daily_stats.groupby('day_of_week').agg({
        'total_pings': ['mean', 'std', 'min', 'max'],
        'unique_users': ['mean', 'std', 'min', 'max'],
    }).round(0)

    print("\n" + "="*90)
    print("DAILY STATISTICS BY DAY-OF-WEEK (assuming 7-day cycle)")
    print("="*90)
    print("\nDay-of-week | Avg Pings | Std Dev | Min | Max | Avg Users | Std Dev | Min | Max")
    print("-" * 90)

    for dow in range(7):
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        name = day_names[dow]
        avg_pings = int(daily_stats[daily_stats['day_of_week'] == dow]['total_pings'].mean())
        std_pings = int(daily_stats[daily_stats['day_of_week'] == dow]['total_pings'].std())
        min_pings = int(daily_stats[daily_stats['day_of_week'] == dow]['total_pings'].min())
        max_pings = int(daily_stats[daily_stats['day_of_week'] == dow]['total_pings'].max())
        avg_users = int(daily_stats[daily_stats['day_of_week'] == dow]['unique_users'].mean())
        std_users = int(daily_stats[daily_stats['day_of_week'] == dow]['unique_users'].std())
        min_users = int(daily_stats[daily_stats['day_of_week'] == dow]['unique_users'].min())
        max_users = int(daily_stats[daily_stats['day_of_week'] == dow]['unique_users'].max())

        print(f"{name:>11} | {avg_pings:>9,} | {std_pings:>7,} | {min_pings:>7,} | {max_pings:>7,} | {avg_users:>9,} | {std_users:>7,} | {min_users:>7,} | {max_users:>7,}")

    return daily_stats, dow_stats

def identify_weekend_pattern(daily_stats):
    """Identify weekend days from the pattern"""
    print("\n" + "="*90)
    print("WEEKEND IDENTIFICATION")
    print("="*90)

    # Look for consecutive low-activity days
    daily_stats['is_anomaly'] = daily_stats['d'] == 27

    # Normalize pings to identify low days
    mean_pings = daily_stats[~daily_stats['is_anomaly']]['total_pings'].mean()
    std_pings = daily_stats[~daily_stats['is_anomaly']]['total_pings'].std()

    # Days with pings < (mean - 1.5*std) are likely weekends
    threshold = mean_pings - 1.5 * std_pings
    daily_stats['likely_weekend'] = (daily_stats['total_pings'] < threshold) & (~daily_stats['is_anomaly'])

    print(f"\nMean daily pings (excluding day 27): {mean_pings:,.0f}")
    print(f"Standard deviation: {std_pings:,.0f}")
    print(f"Weekend threshold (< {threshold:,.0f}): Days with significantly lower activity")

    # Find consecutive low-activity pairs
    print("\n" + "-"*90)
    print("CONSECUTIVE LOW-ACTIVITY DAY PAIRS (potential weekends):")
    print("-"*90)

    low_days = daily_stats[daily_stats['likely_weekend']]['d'].values
    consecutive_pairs = []

    for i in range(len(low_days) - 1):
        if low_days[i+1] - low_days[i] == 1:  # Consecutive days
            consecutive_pairs.append((low_days[i], low_days[i+1]))

    if consecutive_pairs:
        print(f"\nFound {len(consecutive_pairs)} potential weekend pairs:\n")
        for pair in consecutive_pairs:
            day1_pings = daily_stats[daily_stats['d'] == pair[0]]['total_pings'].values[0]
            day2_pings = daily_stats[daily_stats['d'] == pair[1]]['total_pings'].values[0]
            dow1 = pair[0] % 7
            dow2 = pair[1] % 7
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            print(f"  Days {pair[0]:2d}-{pair[1]:2d} ({day_names[dow1]}-{day_names[dow2]}): {day1_pings:>10,} | {day2_pings:>10,} pings")
    else:
        print("\nNo clear consecutive low-activity pairs found.")
        print("Weekends may not be adjacent, or pattern may be different.")

    return daily_stats

def visualize_weekly_pattern(daily_stats):
    """Create detailed visualization of weekly patterns"""
    print("\nGenerating weekly pattern visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Plot 1: Daily pings with DOW coloring
    ax = axes[0]
    colors = ['#1f77b4' if dow % 7 not in [5, 6] else '#ff7f0e'
              for dow in daily_stats['d'] % 7]
    ax.scatter(daily_stats['d'], daily_stats['total_pings'], c=colors, s=100, alpha=0.6)
    ax.plot(daily_stats['d'], daily_stats['total_pings'], alpha=0.3, color='gray')
    ax.axvline(x=27, color='red', linestyle='--', linewidth=2, label='Day 27 (anomaly)', alpha=0.7)

    # Highlight potential weekends
    weekend_days = daily_stats[daily_stats['likely_weekend']]['d'].values
    ax.scatter(weekend_days, daily_stats[daily_stats['likely_weekend']]['total_pings'],
              color='red', s=200, marker='s', label='Likely weekend', alpha=0.7, zorder=5)

    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Total Pings', fontsize=12)
    ax.set_title('Daily Ping Count with Weekend Highlighting (Orange DOW 5-6, Red squares = low activity)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Weekly pattern (stacked by week)
    ax = axes[1]
    for week in daily_stats['week_number'].unique():
        week_data = daily_stats[daily_stats['week_number'] == week].sort_values('day_of_week')
        if not week_data.empty:
            ax.plot(week_data['day_of_week'], week_data['total_pings'],
                   marker='o', label=f'Week {int(week)}', alpha=0.6, linewidth=2)

    ax.set_xlabel('Day of Week (0=Mon, 6=Sun)', fontsize=12)
    ax.set_ylabel('Total Pings', fontsize=12)
    ax.set_title('Weekly Patterns Overlaid (Each line = one week)', fontsize=14)
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Box plot by day of week
    ax = axes[2]
    data_by_dow = [daily_stats[daily_stats['day_of_week'] == dow]['total_pings'].values
                   for dow in range(7)]
    bp = ax.boxplot(data_by_dow, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    patch_artist=True)

    # Color weekends differently
    for patch, dow in zip(bp['boxes'], range(7)):
        if dow in [5, 6]:  # Saturday, Sunday
            patch.set_facecolor('#ff7f0e')
        else:
            patch.set_facecolor('#1f77b4')

    ax.set_ylabel('Total Pings', fontsize=12)
    ax.set_title('Distribution of Daily Pings by Day of Week', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'weekend_identification.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'weekend_identification.png'}")
    plt.close()

def generate_weekend_config(daily_stats):
    """Generate weekend day indices for config"""
    print("\n" + "="*90)
    print("WEEKEND CONFIGURATION")
    print("="*90)

    # Find days that are likely weekends
    daily_stats['is_anomaly'] = daily_stats['d'] == 27
    mean_pings = daily_stats[~daily_stats['is_anomaly']]['total_pings'].mean()
    std_pings = daily_stats[~daily_stats['is_anomaly']]['total_pings'].std()
    threshold = mean_pings - 1.5 * std_pings

    weekend_days = daily_stats[daily_stats['total_pings'] < threshold]['d'].tolist()
    weekend_days = [int(d) for d in weekend_days if d != 27]

    if weekend_days:
        print(f"\nIdentified {len(weekend_days)} potential weekend days:")
        print(f"  {weekend_days}")

        # Verify pattern
        weekend_dows = [d % 7 for d in weekend_days]
        print(f"\nDay-of-week distribution: {weekend_dows}")

        print("\nSuggested config/day_classification.yaml:")
        print("-" * 90)
        print(f"weekend_days: {weekend_days}")
        print(f"anomalous_days: [27]")

        return weekend_days
    else:
        print("\nCould not automatically identify weekend days.")
        print("Review the visualizations and manually inspect the patterns.")
        return []

def main():
    """Run weekend identification analysis"""
    print("="*90)
    print("WEEKEND DAY IDENTIFICATION ANALYSIS")
    print("="*90)

    # Load data
    df = load_data()

    # Analyze patterns
    daily_stats, dow_stats = analyze_day_patterns(df)

    # Identify weekends
    daily_stats = identify_weekend_pattern(daily_stats)

    # Visualize
    visualize_weekly_pattern(daily_stats)

    # Generate config
    weekend_days = generate_weekend_config(daily_stats)

    print("\n" + "="*90)
    print("NEXT STEPS")
    print("="*90)
    print("\n1. Review the visualization: analysis/weekend_identification.png")
    print("2. Check the 'CONSECUTIVE LOW-ACTIVITY DAY PAIRS' section above")
    print("3. Review the 'DAILY STATISTICS BY DAY-OF-WEEK' table")
    print("\n4. Update config/day_classification.yaml with the identified weekend days")
    print("   OR manually specify based on your analysis of the patterns")
    print("\n5. Run the preprocessing: uv run scripts/run_pipeline.py")

if __name__ == "__main__":
    main()
