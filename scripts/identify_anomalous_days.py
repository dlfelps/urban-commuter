"""
Identify Anomalous Days

Finds weekdays that have activity levels at or near weekend levels,
indicating they should be excluded from analysis as anomalous.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"

def load_data():
    """Load cityA dataset"""
    print("Loading cityA-dataset.csv...")
    df = pd.read_csv(DATA_DIR / "cityA-dataset.csv")
    df['dow'] = df['d'] % 7
    return df

def analyze_anomalies(df):
    """Identify anomalous days"""
    print("\nAnalyzing daily activity levels...")

    # Calculate daily statistics
    daily_stats = df.groupby('d').agg({
        'uid': ['count', 'nunique'],
    }).reset_index()

    daily_stats.columns = ['d', 'total_pings', 'unique_users']
    daily_stats['dow'] = daily_stats['d'] % 7

    # Calculate position statistics (excluding day 27)
    daily_stats_clean = daily_stats[daily_stats['d'] != 27].copy()

    # Get average activity by day-of-week position
    dow_stats = daily_stats_clean.groupby('dow').agg({
        'total_pings': ['mean', 'std'],
        'unique_users': ['mean', 'std'],
    }).reset_index()

    dow_stats.columns = ['dow', 'mean_pings', 'std_pings', 'mean_users', 'std_users']

    # Identify weekend positions (0 and 6)
    weekend_positions = [0, 6]
    weekday_positions = [1, 2, 3, 4, 5]

    weekend_avg_pings = dow_stats[dow_stats['dow'].isin(weekend_positions)]['mean_pings'].mean()
    weekday_avg_pings = dow_stats[dow_stats['dow'].isin(weekday_positions)]['mean_pings'].mean()

    print("\n" + "="*100)
    print("ACTIVITY LEVEL ANALYSIS")
    print("="*100)
    print(f"\nWeekend average pings: {weekend_avg_pings:,.0f}")
    print(f"Weekday average pings: {weekday_avg_pings:,.0f}")
    print(f"Ratio (weekend/weekday): {(weekend_avg_pings/weekday_avg_pings)*100:.1f}%")

    # Identify anomalous weekdays
    # Anomalous = weekday with less than 110,000 pings
    anomaly_threshold = 110000

    print(f"\nAnomaly threshold (less than 110K pings): {anomaly_threshold:,.0f}")

    anomalous_days = []
    suspicious_days = []

    # Check only weekday positions (1-5) for anomalies
    for dow in weekday_positions:
        dow_data = daily_stats_clean[daily_stats_clean['dow'] == dow]
        for _, row in dow_data.iterrows():
            day = int(row['d'])
            pings = int(row['total_pings'])
            pct_of_weekend = (pings / weekend_avg_pings) * 100
            pct_of_weekday = (pings / weekday_avg_pings) * 100

            if pings <= anomaly_threshold:
                anomalous_days.append((day, dow, pings, pct_of_weekend, pct_of_weekday))

    # Also check day 27 (position 0, but anomalous)
    day_27_data = daily_stats[daily_stats['d'] == 27]
    if len(day_27_data) > 0:
        row = day_27_data.iloc[0]
        day = int(row['d'])
        pings = int(row['total_pings'])
        pct_of_weekend = (pings / weekend_avg_pings) * 100
        pct_of_weekday = (pings / weekday_avg_pings) * 100
        anomalous_days.append((day, 0, pings, pct_of_weekend, pct_of_weekday))

    # Sort by ping count
    anomalous_days.sort(key=lambda x: x[2])

    print("\n" + "="*100)
    print("IDENTIFIED ANOMALOUS DAYS")
    print("="*100)
    print(f"\n{'Day':<6} {'DoW':<6} {'Pings':<15} {'% of Weekend':<15} {'% of Weekday':<15} Notes")
    print("-"*100)

    for day, dow, pings, pct_weekend, pct_weekday in anomalous_days:
        dow_name = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][dow]
        note = ""
        if day == 27:
            note = "← Known anomaly (natural disaster event)"
        else:
            note = "← Anomalous weekday (activity at weekend level)"

        print(f"{day:<6} {dow_name:<16} {pings:>13,} {pct_weekend:>13.1f}% {pct_weekday:>13.1f}% {note}")

    # All identified anomalies fall below the 110K threshold
    all_identified_anomalies = sorted(list(set([day for day, _, _, _, _ in anomalous_days])))

    print("\n" + "="*100)
    print("IDENTIFIED ANOMALIES")
    print("="*100)
    print(f"\nAll days with <110,000 pings: {sorted(all_identified_anomalies)}")

    all_anomalies = sorted(all_identified_anomalies)

    print("\n" + "="*100)
    print("RECOMMENDED CONFIGURATION")
    print("="*100)
    print(f"\nAdd to config/day_classification.yaml:")
    print(f"anomalous_days: {all_anomalies}")

    # Show impact on counts
    print("\n" + "="*100)
    print("IMPACT ON DAY COUNTS")
    print("="*100)

    total_days = 75
    weekend_days_count = len([d for d in range(75) if d % 7 in [0, 6] and d != 27])
    weekday_days_count = 75 - weekend_days_count - len(all_anomalies) - 1

    print(f"\nCurrent configuration (day 27 only):")
    print(f"  Weekdays: {75 - 11} days")
    print(f"  Weekends: {11} days")
    print(f"  Anomalous: 1 day (27)")

    print(f"\nWith identified anomalies:")
    print(f"  Weekdays: {75 - 11 - len(all_anomalies)} days")
    print(f"  Weekends: 11 days")
    print(f"  Anomalous: {len(all_anomalies) + 1} days ({all_anomalies + [27]})")

    return all_anomalies

def main():
    """Run anomaly detection"""
    print("="*100)
    print("ANOMALOUS DAY IDENTIFICATION")
    print("="*100)

    df = load_data()
    anomalies = analyze_anomalies(df)

    print("\n" + "="*100)
    print("NEXT STEPS")
    print("="*100)
    print("\n1. Review the anomalous days listed above")
    print("2. Update config/day_classification.yaml with the recommended anomalous_days list")
    print("3. Optionally, run scripts/analyze_weekday_cycle.py to visualize the impact")
    print("4. Run preprocessing: uv run scripts/run_pipeline.py")

if __name__ == "__main__":
    main()
