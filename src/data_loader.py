"""
Data Loading Module

Handles loading and validating the YJMob100K dataset and POI data.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Dict

class DataLoader:
    """Load and validate mobility and POI datasets"""

    def __init__(self, data_dir: Path = None, config_dir: Path = None):
        """
        Initialize data loader

        Args:
            data_dir: Path to data directory (default: ./data)
            config_dir: Path to config directory (default: ./config)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"

        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_mobility_data(self) -> pd.DataFrame:
        """
        Load cityD mobility dataset

        Returns:
            DataFrame with columns: uid, d, t, x, y
            - uid: user ID
            - d: day (0-74)
            - t: timeslot (0-47, 30-min bins)
            - x, y: grid coordinates (1-200)
        """
        filepath = self.data_dir / "cityD-dataset.csv"

        if not filepath.exists():
            raise FileNotFoundError(f"Mobility dataset not found: {filepath}")

        print(f"Loading mobility data from {filepath}...")
        df = pd.read_csv(filepath)

        # Validate required columns
        required_cols = ['uid', 'd', 't', 'x', 'y']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate data ranges
        assert df['d'].min() >= 0 and df['d'].max() <= 74, "Day out of range [0, 74]"
        assert df['t'].min() >= 0 and df['t'].max() <= 47, "Timeslot out of range [0, 47]"
        assert df['x'].min() >= 1 and df['x'].max() <= 200, "X coordinate out of range [1, 200]"
        assert df['y'].min() >= 1 and df['y'].max() <= 200, "Y coordinate out of range [1, 200]"

        print(f"✓ Loaded {len(df):,} mobility records")
        print(f"  Users: {df['uid'].nunique():,}")
        print(f"  Days: {df['d'].min()}-{df['d'].max()}")
        print(f"  Timeslots: {df['t'].min()}-{df['t'].max()}")
        print(f"  Grid cells: {df[['x', 'y']].drop_duplicates().shape[0]:,}")

        return df

    def load_poi_data(self) -> pd.DataFrame:
        """
        Load POI (Points of Interest) dataset

        Returns:
            DataFrame with columns: x, y, category, POI_count
            - x, y: grid coordinates (1-200)
            - category: POI category dimension (1-85)
            - POI_count: number of POIs in this category at this cell
        """
        filepath = self.data_dir / "POIdata_cityD.csv"

        if not filepath.exists():
            raise FileNotFoundError(f"POI dataset not found: {filepath}")

        print(f"Loading POI data from {filepath}...")
        df = pd.read_csv(filepath)

        # Validate required columns
        required_cols = ['x', 'y', 'category', 'POI_count']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"✓ Loaded {len(df):,} POI records")
        print(f"  Grid cells with POI data: {df[['x', 'y']].drop_duplicates().shape[0]:,}")
        print(f"  POI categories: {df['category'].min()}-{df['category'].max()}")

        return df

    def load_day_classification(self) -> Dict[str, list]:
        """
        Load day classification (weekday vs weekend) from config

        Returns:
            Dict with keys 'weekend_days' (list of day indices) and 'anomalous_days'

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        filepath = self.config_dir / "day_classification.yaml"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Day classification config not found: {filepath}\n"
                "Please run: python scripts/inspect_temporal_patterns.py\n"
                "Then create config/day_classification.yaml"
            )

        print(f"Loading day classification from {filepath}...")
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        weekend_days = config.get('weekend_days', [])
        anomalous_days = config.get('anomalous_days', [27])  # Day 27 is always anomalous

        if not isinstance(weekend_days, list):
            raise ValueError("'weekend_days' must be a list")
        if not isinstance(anomalous_days, list):
            raise ValueError("'anomalous_days' must be a list")

        print(f"✓ Loaded day classification")
        print(f"  Weekend days: {weekend_days}")
        print(f"  Anomalous days: {anomalous_days}")

        return {
            'weekend_days': weekend_days,
            'anomalous_days': anomalous_days
        }

    def create_day_type_mapping(self, day_classification: Dict[str, list]) -> pd.DataFrame:
        """
        Create mapping of days to day_type (weekday/weekend/anomalous)

        Args:
            day_classification: Dict from load_day_classification()

        Returns:
            DataFrame with columns: day, day_type, available
        """
        weekend_days = day_classification['weekend_days']
        anomalous_days = day_classification['anomalous_days']

        days = list(range(75))
        day_types = []
        available = []

        for day in days:
            if day in anomalous_days:
                day_types.append('anomalous')
                available.append(False)
            elif day in weekend_days:
                day_types.append('weekend')
                available.append(True)
            else:
                day_types.append('weekday')
                available.append(True)

        return pd.DataFrame({
            'day': days,
            'day_type': day_types,
            'available': available
        })


def main():
    """Test data loader"""
    loader = DataLoader()

    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    # Load mobility data
    mobility_df = loader.load_mobility_data()

    # Load POI data
    poi_df = loader.load_poi_data()

    # Load day classification
    day_class = loader.load_day_classification()

    # Create day type mapping
    day_mapping = loader.create_day_type_mapping(day_class)

    print("\n" + "="*80)
    print("DATA VALIDATION COMPLETE")
    print("="*80)
    print(f"All datasets loaded successfully!")


if __name__ == "__main__":
    main()
