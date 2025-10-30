"""
Data Preprocessing Module

Transforms raw mobility data into intermediate format for network analysis:
- cell_flows.parquet: Cell-to-cell movement flows (self-loops excluded)
- cell_attributes.parquet: Cell properties and POI features
- temporal_metadata.parquet: Day classification and metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from .data_loader import DataLoader

class DataPreprocessor:
    """Transform raw data into intermediate format"""

    def __init__(self, data_dir: Path = None, config_dir: Path = None, output_dir: Path = None):
        """
        Initialize preprocessor

        Args:
            data_dir: Path to raw data directory
            config_dir: Path to config directory
            output_dir: Path to save intermediate data (default: ./intermediate)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "intermediate"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.loader = DataLoader(data_dir, config_dir)
        self.mobility_df = None
        self.poi_df = None
        self.day_mapping = None

    def load_all_data(self) -> None:
        """Load all raw data"""
        print("\n" + "="*80)
        print("LOADING RAW DATA")
        print("="*80)

        self.mobility_df = self.loader.load_mobility_data()
        self.poi_df = self.loader.load_poi_data()

        day_class = self.loader.load_day_classification()
        num_days = self.mobility_df['d'].max() + 1
        self.day_mapping = self.loader.create_day_type_mapping(day_class, num_days=num_days)

    def create_cell_flows(self) -> pd.DataFrame:
        """
        Create cell-to-cell flow data (excluding self-loops)

        Returns:
            DataFrame with columns:
            - source_cell_x, source_cell_y
            - target_cell_x, target_cell_y
            - timeslot (0-47)
            - day_type (weekday/weekend/anomalous)
            - flow_count (frequency)
        """
        print("\n" + "="*80)
        print("CREATING CELL FLOWS")
        print("="*80)

        df = self.mobility_df.copy()

        # Merge day type information
        df = df.merge(self.day_mapping, left_on='d', right_on='day', how='left')

        # Filter out anomalous days
        print(f"Original records: {len(df):,}")
        df = df[df['available']].copy()
        print(f"After removing anomalous days: {len(df):,}")

        # Sort by user and day/timeslot to identify consecutive observations
        df = df.sort_values(['uid', 'd', 't']).reset_index(drop=True)

        # Create lagged columns to identify flows
        # Each user's current position and next position form a flow
        df['next_x'] = df.groupby('uid')['x'].shift(-1)
        df['next_y'] = df.groupby('uid')['y'].shift(-1)

        # Remove rows where next observation is on a different day
        # (flows should be within same day)
        df['next_d'] = df.groupby('uid')['d'].shift(-1)
        df = df[df['d'] == df['next_d']].copy()

        # Remove self-loops (same cell to same cell)
        df = df[~((df['x'] == df['next_x']) & (df['y'] == df['next_y']))].copy()

        print(f"After filtering same-day flows only: {len(df):,}")
        print(f"After removing self-loops: {len(df):,}")

        # Aggregate flows by (source, target, timeslot, day_type)
        flows = df.groupby(
            ['x', 'y', 'next_x', 'next_y', 't', 'day_type']
        ).size().reset_index(name='flow_count')

        flows = flows.rename(columns={
            'x': 'source_cell_x',
            'y': 'source_cell_y',
            'next_x': 'target_cell_x',
            'next_y': 'target_cell_y',
            't': 'timeslot'
        })

        print(f"[OK] Created flow data")
        print(f"  Total flows: {len(flows):,}")
        print(f"  Unique (source, target) pairs: {flows[['source_cell_x', 'source_cell_y', 'target_cell_x', 'target_cell_y']].drop_duplicates().shape[0]:,}")
        print(f"  Timeslots represented: {flows['timeslot'].nunique()}")
        print(f"  Day types: {flows['day_type'].unique().tolist()}")

        return flows

    def create_cell_attributes(self) -> pd.DataFrame:
        """
        Create cell attribute data with POI enrichment

        Returns:
            DataFrame with columns:
            - cell_x, cell_y
            - total_pings (activity intensity)
            - unique_users
            - unique_cells_visited (destination diversity)
            - poi_* (POI counts for each category, aggregated)
            - cell_type (urban/rural based on activity)
        """
        print("\n" + "="*80)
        print("CREATING CELL ATTRIBUTES")
        print("="*80)

        df = self.mobility_df.copy()

        # Filter to available days only
        df = df.merge(self.day_mapping[['day', 'available']], left_on='d', right_on='day', how='left')
        df = df[df['available']].copy()

        # Aggregate cell statistics
        cells = df.groupby(['x', 'y']).agg(
            total_pings=('uid', 'count'),
            unique_users=('uid', 'nunique'),
        ).reset_index()

        cells = cells.rename(columns={'x': 'cell_x', 'y': 'cell_y'})

        print(f"[OK] Created cell statistics for {len(cells):,} cells")

        # Enrich with POI features
        # Pivot POI data so each category becomes a column
        poi_pivot = self.poi_df.pivot_table(
            index=['x', 'y'],
            columns='category',
            values='POI_count',
            fill_value=0
        )

        # Flatten column names
        poi_pivot.columns = [f'poi_{cat}' for cat in poi_pivot.columns]
        poi_pivot = poi_pivot.reset_index()
        poi_pivot = poi_pivot.rename(columns={'x': 'cell_x', 'y': 'cell_y'})

        print(f"[OK] Loaded POI features for {len(poi_pivot):,} cells")

        # Merge POI data (left join to keep all cells even without POI)
        cells = cells.merge(
            poi_pivot,
            on=['cell_x', 'cell_y'],
            how='left'
        )

        # Fill missing POI values with 0
        poi_cols = [col for col in cells.columns if col.startswith('poi_')]
        cells[poi_cols] = cells[poi_cols].fillna(0).astype(int)

        # Classify cells as urban/rural based on activity
        # Urban: high ping count, rural: low ping count
        median_pings = cells['total_pings'].median()
        cells['cell_type'] = cells['total_pings'].apply(
            lambda x: 'urban' if x >= median_pings else 'rural'
        )

        print(f"[OK] Classified cells")
        print(f"  Urban cells: {(cells['cell_type'] == 'urban').sum()}")
        print(f"  Rural cells: {(cells['cell_type'] == 'rural').sum()}")

        return cells

    def create_temporal_metadata(self) -> pd.DataFrame:
        """
        Create temporal metadata for reference

        Returns:
            DataFrame with columns:
            - day (0-74)
            - day_type (weekday/weekend/anomalous)
            - available (bool)
        """
        print("\n" + "="*80)
        print("CREATING TEMPORAL METADATA")
        print("="*80)

        metadata = self.day_mapping.copy()

        print(f"[OK] Created temporal metadata")
        print(f"  Weekdays: {(metadata['day_type'] == 'weekday').sum()}")
        print(f"  Weekends: {(metadata['day_type'] == 'weekend').sum()}")
        print(f"  Anomalous: {(metadata['day_type'] == 'anomalous').sum()}")

        return metadata

    def save_intermediate_data(
        self,
        flows: pd.DataFrame,
        cells: pd.DataFrame,
        metadata: pd.DataFrame
    ) -> None:
        """Save intermediate data to parquet files"""
        print("\n" + "="*80)
        print("SAVING INTERMEDIATE DATA")
        print("="*80)

        # Save flows
        flows_path = self.output_dir / "cell_flows.parquet"
        flows.to_parquet(flows_path, index=False)
        print(f"[OK] Saved cell_flows.parquet ({len(flows):,} rows)")

        # Save cell attributes
        cells_path = self.output_dir / "cell_attributes.parquet"
        cells.to_parquet(cells_path, index=False)
        print(f"[OK] Saved cell_attributes.parquet ({len(cells):,} rows)")

        # Save temporal metadata
        metadata_path = self.output_dir / "temporal_metadata.parquet"
        metadata.to_parquet(metadata_path, index=False)
        print(f"[OK] Saved temporal_metadata.parquet ({len(metadata):,} rows)")

        print(f"\nIntermediate data saved to: {self.output_dir}")

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run complete preprocessing pipeline

        Returns:
            Tuple of (flows, cells, metadata) DataFrames
        """
        print("\n" + "="*80)
        print("DATA PREPROCESSING PIPELINE")
        print("="*80)

        self.load_all_data()

        flows = self.create_cell_flows()
        cells = self.create_cell_attributes()
        metadata = self.create_temporal_metadata()

        self.save_intermediate_data(flows, cells, metadata)

        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)

        return flows, cells, metadata


def main():
    """Run preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    flows, cells, metadata = preprocessor.run()


if __name__ == "__main__":
    main()
