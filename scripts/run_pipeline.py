#!/usr/bin/env python
"""
Complete Data Preprocessing Pipeline Runner

This script runs the full preprocessing pipeline in sequence:
1. Loads all raw data and validates it
2. Processes mobility data into cell flows
3. Aggregates cell attributes with POI data
4. Creates temporal metadata
5. Saves all intermediate data to parquet

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir.parent))

from src.preprocessing import DataPreprocessor


def main():
    """Run complete preprocessing pipeline"""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()

        # Run full pipeline
        flows, cells, metadata = preprocessor.run()

        print("\n" + "="*80)
        print("SUCCESS: Data preprocessing complete!")
        print("="*80)
        print("\nIntermediate data files ready in: ./intermediate/")
        print("\nYou can now:")
        print("  1. Load flows: pd.read_parquet('intermediate/cell_flows.parquet')")
        print("  2. Load cells: pd.read_parquet('intermediate/cell_attributes.parquet')")
        print("  3. Load metadata: pd.read_parquet('intermediate/temporal_metadata.parquet')")
        print("\nNext steps: Build NetworkX graphs and run analysis algorithms")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure:")
        print("  1. config/day_classification.yaml exists")
        print("  2. Run 'python scripts/inspect_temporal_patterns.py' first")
        print("  3. Update config/day_classification.yaml with weekend days")
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
