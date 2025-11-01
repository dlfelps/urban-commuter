"""
Plot Raw Observation Density - Log Scale

Loads the original unprocessed cityA dataset and counts observations at each
x,y grid location, then plots as a log-scale heatmap for comparison with
shortest path distances.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

GRID_SIZE = 200


def main():
    """Load raw data and create log-scale observation density plot"""
    print("=" * 80)
    print("RAW OBSERVATION DENSITY - LOG SCALE")
    print("=" * 80)

    # Load original cityA dataset
    data_path = DATA_DIR / "cityA-dataset.csv"

    if not data_path.exists():
        print(f"\n[ERROR] Dataset not found: {data_path}")
        return

    print(f"\nLoading {data_path.name}...")
    df = pd.read_csv(data_path)

    print(f"Raw data loaded: {len(df):,} observations")
    print(f"Columns: {list(df.columns)}")

    # Display first few rows
    print(f"\nFirst few rows:")
    print(df.head())

    # Get location columns (typically x, y or similar)
    # Check for common column names
    x_col = None
    y_col = None

    possible_x = ['x', 'X', 'lon', 'longitude', 'cell_x']
    possible_y = ['y', 'Y', 'lat', 'latitude', 'cell_y']

    for col in possible_x:
        if col in df.columns:
            x_col = col
            break

    for col in possible_y:
        if col in df.columns:
            y_col = col
            break

    if x_col is None or y_col is None:
        print(f"\n[ERROR] Could not find x/y coordinate columns")
        print(f"Available columns: {list(df.columns)}")
        return

    print(f"\nUsing coordinates: {x_col}, {y_col}")

    # Extract coordinates
    locations = df[[x_col, y_col]].copy()
    locations.columns = ['x', 'y']

    # Remove any invalid coordinates
    locations = locations.dropna()
    print(f"Valid coordinates: {len(locations):,}")

    # Count observations at each location
    print("\nCounting observations at each grid location...")
    location_counts = locations.groupby(['x', 'y']).size().reset_index(name='count')

    print(f"Unique locations: {len(location_counts):,}")
    print(f"\nObservation statistics:")
    print(f"  Min: {location_counts['count'].min():,}")
    print(f"  Max: {location_counts['count'].max():,}")
    print(f"  Mean: {location_counts['count'].mean():,.1f}")
    print(f"  Median: {location_counts['count'].median():,.1f}")
    print(f"  Std Dev: {location_counts['count'].std():,.1f}")

    # Create grid for heatmap
    print("\nCreating heatmap grid...")
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    for _, row in location_counts.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        count = int(row['count'])

        # Adjust for 1-indexed grid (or 0-indexed, depending on data)
        if 1 <= x <= GRID_SIZE and 1 <= y <= GRID_SIZE:
            grid[y - 1, x - 1] = count
        elif 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            grid[y, x] = count

    # Count zeros
    zero_cells = np.sum(grid == 0)
    nonzero_cells = np.sum(grid > 0)

    print(f"Grid cells with observations: {nonzero_cells:,}")
    print(f"Empty grid cells: {zero_cells:,}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Figure 1: Linear scale heatmap
    fig, ax = plt.subplots(figsize=(16, 14))

    # Mask zeros for better visualization
    masked_grid = np.ma.masked_where(grid == 0, grid)

    im = ax.imshow(masked_grid, cmap='YlOrRd', origin='lower',
                    extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                    interpolation='bilinear')

    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title('Raw Observation Density - Linear Scale\n(Original cityA Dataset)',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.2)

    cbar = plt.colorbar(im, ax=ax, label='Observation Count', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path1 = OUTPUT_DIR / "raw_observations_linear.png"
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path1}")
    plt.close()

    # Figure 2: Log scale heatmap
    fig, ax = plt.subplots(figsize=(16, 14))

    # Apply log transformation (add 1 to avoid log(0))
    log_grid = np.log10(grid + 1)

    # Mask the zeros in log space (those that were 0 in original)
    masked_log_grid = np.ma.masked_where(grid == 0, log_grid)

    im = ax.imshow(masked_log_grid, cmap='plasma', origin='lower',
                    extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                    interpolation='bilinear')

    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title('Raw Observation Density - Log Scale\n(Original cityA Dataset, Log10(Count + 1))',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.2)

    cbar = plt.colorbar(im, ax=ax, label='Log10(Observation Count + 1)', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path2 = OUTPUT_DIR / "raw_observations_logscale.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()

    # Figure 3: Multi-panel comparison
    fig = plt.figure(figsize=(20, 10))

    # Linear scale
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(masked_grid, cmap='YlOrRd', origin='lower',
                      extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                      interpolation='bilinear')
    ax1.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Scale', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Count')

    # Log scale
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(masked_log_grid, cmap='plasma', origin='lower',
                      extent=[1, GRID_SIZE, 1, GRID_SIZE], aspect='equal',
                      interpolation='bilinear')
    ax2.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax2.set_title('Log Scale (Log10)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Log10(Count + 1)')

    plt.suptitle('Raw Observation Density Comparison - Original cityA Dataset',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path3 = OUTPUT_DIR / "raw_observations_comparison.png"
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path3}")
    plt.close()

    # Figure 4: Distribution of observation counts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram of counts (linear)
    ax = axes[0, 0]
    ax.hist(location_counts['count'], bins=100, color='steelblue',
            edgecolor='black', alpha=0.7)
    ax.axvline(location_counts['count'].mean(), color='red',
               linestyle='--', linewidth=2,
               label=f'Mean: {location_counts["count"].mean():,.0f}')
    ax.axvline(location_counts['count'].median(), color='green',
               linestyle='--', linewidth=2,
               label=f'Median: {location_counts["count"].median():,.0f}')
    ax.set_xlabel('Observations per Location', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Observations per Location (Linear)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Histogram of counts (log scale y-axis)
    ax = axes[0, 1]
    ax.hist(location_counts['count'], bins=100, color='coral',
            edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Observations per Location', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency (log)', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Observations per Location (Log Y)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Cumulative distribution
    ax = axes[1, 0]
    sorted_counts = np.sort(location_counts['count'].values)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax.plot(sorted_counts, cumulative, linewidth=2, color='darkgreen')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax.set_xlabel('Observations per Location', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Percentile breakdown
    ax = axes[1, 1]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(location_counts['count'], p) for p in percentiles]
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    bars = ax.barh([f'{p}th' for p in percentiles], percentile_values, color=colors_bar,
                    edgecolor='black')
    ax.set_xlabel('Observations per Location', fontsize=11, fontweight='bold')
    ax.set_title('Percentile Breakdown', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, percentile_values)):
        ax.text(val, i, f' {val:.0f}', va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Raw Observation Count Statistics - Original cityA Dataset',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path4 = OUTPUT_DIR / "raw_observations_statistics.png"
    plt.savefig(output_path4, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path4}")
    plt.close()

    # Save the grid data for comparison
    print("\nSaving grid data...")
    grid_df = pd.DataFrame({
        'x': np.tile(np.arange(1, GRID_SIZE + 1), GRID_SIZE),
        'y': np.repeat(np.arange(1, GRID_SIZE + 1), GRID_SIZE),
        'count': grid.flatten()
    })

    grid_path = OUTPUT_DIR / "raw_observations_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"Saved: {grid_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total observations: {len(df):,}")
    print(f"Unique locations: {len(location_counts):,}")
    print(f"Grid coverage: {nonzero_cells:,}/{GRID_SIZE * GRID_SIZE:,} cells")
    print(f"Coverage percentage: {(nonzero_cells / (GRID_SIZE * GRID_SIZE)) * 100:.1f}%")
    print(f"\nObservations per location:")
    print(f"  Min: {location_counts['count'].min():,}")
    print(f"  Max: {location_counts['count'].max():,}")
    print(f"  Mean: {location_counts['count'].mean():,.1f}")
    print(f"  Median: {location_counts['count'].median():,.1f}")


if __name__ == "__main__":
    main()
