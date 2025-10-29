"""
Verify Weekday Mapping

Shows the mapping of days 0-74 to weekdays/weekends based on:
- Day 0 (position 0 mod 7) = Sunday (weekend)
- Day 6 (position 6 mod 7) = Saturday (weekend)
- Days 1-5 (positions 1-5 mod 7) = Weekdays (Mon-Fri)
"""

import pandas as pd

def generate_weekday_mapping():
    """Generate complete weekday mapping for days 0-74"""

    # Calendar mapping based on analysis results
    # Position 0 mod 7 = Sunday (weekend)
    # Position 1 mod 7 = Monday (weekday)
    # Position 2 mod 7 = Tuesday (weekday)
    # Position 3 mod 7 = Wednesday (weekday)
    # Position 4 mod 7 = Thursday (weekday)
    # Position 5 mod 7 = Friday (weekday)
    # Position 6 mod 7 = Saturday (weekend)

    day_names = {
        0: 'Sunday',
        1: 'Monday',
        2: 'Tuesday',
        3: 'Wednesday',
        4: 'Thursday',
        5: 'Friday',
        6: 'Saturday'
    }

    weekend_positions = {0, 6}

    # Generate mapping for all 75 days
    mapping = []
    for day in range(75):
        pos = day % 7
        day_name = day_names[pos]
        is_weekend = pos in weekend_positions
        day_type = 'WEEKEND' if is_weekend else 'WEEKDAY'

        is_anomaly = day == 27
        status = 'ANOMALY' if is_anomaly else day_type

        mapping.append({
            'Day': day,
            'Position (mod 7)': pos,
            'Day Name': day_name,
            'Type': status
        })

    return pd.DataFrame(mapping)

def main():
    """Print the weekday mapping"""

    print("="*80)
    print("WEEKDAY MAPPING FOR DAYS 0-74")
    print("="*80)
    print("\nBased on analysis results:")
    print("  • Position 0 (0 mod 7) = Sunday (WEEKEND)")
    print("  • Position 1 (1 mod 7) = Monday (WEEKDAY)")
    print("  • Position 2 (2 mod 7) = Tuesday (WEEKDAY)")
    print("  • Position 3 (3 mod 7) = Wednesday (WEEKDAY)")
    print("  • Position 4 (4 mod 7) = Thursday (WEEKDAY)")
    print("  • Position 5 (5 mod 7) = Friday (WEEKDAY)")
    print("  • Position 6 (6 mod 7) = Saturday (WEEKEND)")
    print("\nNote: Friday shows NO reduced activity - classified as regular weekday")
    print("="*80)

    mapping = generate_weekday_mapping()

    # Print in groups of 7 (one week at a time)
    print("\nDay Mapping (grouped by weeks):\n")

    for week in range(11):
        start_day = week * 7
        end_day = min((week + 1) * 7, 75)

        print(f"Week {week}:")
        week_data = mapping[start_day:end_day]

        for _, row in week_data.iterrows():
            day = int(row['Day'])
            pos = int(row['Position (mod 7)'])
            name = row['Day Name']
            status = row['Type']

            # Format status with color hints
            if status == 'WEEKEND':
                status_str = f"{status:9s}"
            elif status == 'ANOMALY':
                status_str = f"{status:9s} (excluded)"
            else:
                status_str = f"{status:9s}"

            print(f"  Day {day:2d} (pos {pos}) - {name:9s} {status_str}")

        print()

    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)

    weekday_count = (mapping['Type'] == 'WEEKDAY').sum()
    weekend_count = (mapping['Type'] == 'WEEKEND').sum()
    anomaly_count = (mapping['Type'] == 'ANOMALY').sum()

    print(f"\nWeekdays (Mon-Fri): {weekday_count} days")
    print(f"Weekends (Sat-Sun): {weekend_count} days")
    print(f"Anomalous days: {anomaly_count} day")
    print(f"Total: {weekday_count + weekend_count + anomaly_count} days")

    # Print weekend day indices
    weekend_days = sorted(mapping[mapping['Type'] == 'WEEKEND']['Day'].tolist())
    print(f"\nWeekend day indices (for config):")
    print(f"  {weekend_days}")

    # Print weekday day indices
    weekday_days = sorted(mapping[mapping['Type'] == 'WEEKDAY']['Day'].tolist())
    print(f"\nWeekday day indices (for reference):")
    print(f"  {weekday_days}")

    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print("\nAdd to config/day_classification.yaml:")
    print(f"\nweekend_days: {weekend_days}")
    print(f"anomalous_days: [27]")

if __name__ == "__main__":
    main()
