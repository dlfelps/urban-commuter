# Urban Mobility Hub Detection & Resilience Analysis

Identify critical mobility hubs in cities using graph algorithms and network analysis. This project applies centrality algorithms, community detection, and resilience testing to urban mobility networks to uncover infrastructure bottlenecks, activity centers, and optimal transit planning opportunities.

## Overview

This project analyzes urban mobility patterns using the LyMob dataset, which contains 100,000+ users' movement data across a 200×200 grid representing a city. By constructing directed networks from mobility flows, we apply classical graph algorithms to answer critical urban planning questions:

- **Critical Infrastructure**: Which grid cells act as bottlenecks (high betweenness centrality)?
- **Activity Centers**: Where are the true mobility attractors vs. pass-through areas (PageRank)?
- **Natural Neighborhoods**: What mobility patterns define neighborhood boundaries (community detection)?
- **Commute Dynamics**: Where are residential vs. commercial zones (flow imbalance analysis)?
- **Network Resilience**: Which cells are most critical for maintaining city connectivity?

## Key Features

- **Graph Analysis**: NetworkX-powered network construction and centrality computation
- **Temporal Analysis**: Weekday/weekend breakdown, 30-minute timeslot granularity
- **Scalable Pipeline**: From raw CSV to analysis-ready parquet format
- **Comprehensive Metrics**: Betweenness centrality, PageRank, community detection, flow imbalance
- **Resilience Testing**: Simulated cell closure analysis for network robustness evaluation

## Quick Start

### Prerequisites

- Python 3.13+ (see `.python-version`)
- `uv` package manager ([install here](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/urban-mobility.git
cd urban-mobility
```

2. The project uses `uv` for dependency management. No virtual environment setup needed! All commands automatically use the correct environment:

```bash
uv run python -m src.data_loader      # Validate data loads
uv run scripts/inspect_temporal_patterns.py  # Identify weekends
uv run scripts/run_pipeline.py        # Run full preprocessing
uv run pytest tests/                  # Run tests
```

## Project Structure

```
urban-mobility/
├── data/                              # Raw datasets
│   ├── cityD-dataset.csv             # Main mobility data (~150MB)
│   ├── POIdata_cityD.csv             # Points of interest by grid cell
│   ├── POI_datacategories.csv        # POI category reference
│   └── lymob-4cities.pdf             # Dataset documentation
├── src/
│   ├── data_loader.py                # Raw data loading & validation
│   └── preprocessing.py              # CSV → Parquet transformation
├── scripts/
│   ├── inspect_temporal_patterns.py  # Visualize to identify weekends
│   └── run_pipeline.py               # Execute full preprocessing
├── config/
│   └── day_classification.yaml       # Weekend day indices (user-configured)
├── intermediate/                      # Generated intermediate data (gitignored)
│   ├── cell_flows.parquet            # Aggregated flows between cells
│   ├── cell_attributes.parquet       # Cell properties with POI features
│   └── temporal_metadata.parquet     # Day/timeslot classification
├── tests/                             # Unit tests
├── CLAUDE.md                          # Development guide for Claude Code
└── pyproject.toml                     # Python project configuration
```

## Data Pipeline

### Stage 1: Raw Data
- **cityD-dataset.csv**: 111M observations of individual movements (user, day, timeslot, grid_x, grid_y)
- **POIdata_cityD.csv**: Points of interest enumerated by grid cell
- **Coverage**: 75 days, 48 timeslots/day (30-min bins), 200×200 grid cells

### Stage 2: Intermediate Format (Parquet)
Optimized for graph analysis:
- **cell_flows.parquet**: Directed edge list with flow counts by timeslot
- **cell_attributes.parquet**: Grid cell features (activity, POI categories, classification)
- **temporal_metadata.parquet**: Day type labels (weekday/weekend/anomalous)

### Stage 3: Analysis
- Construct directed graphs from flows
- Compute centrality metrics (betweenness, PageRank, degree)
- Community detection and flow imbalance analysis
- Network resilience via simulated failures

## Usage Examples

### 1. Validate Data
```bash
uv run python -m src.data_loader
```

### 2. Identify Weekend Days
Review temporal patterns to determine which days are weekends, then update `config/day_classification.yaml`:
```bash
uv run scripts/inspect_temporal_patterns.py
# Review output, then edit config/day_classification.yaml with weekend_days indices
```

### 3. Run Full Preprocessing
```bash
uv run scripts/run_pipeline.py
```
Generates intermediate parquet files ready for analysis.

### 4. Run Tests
```bash
uv run pytest tests/
uv run pytest tests/ --cov=src  # With coverage report
```

## Dataset

**Source**: LyMob Dataset
**DOI**: https://doi.org/10.5281/zenodo.14219563
**Documentation**: See `data/lymob-4cities.pdf`
**Coverage**: 100,000 users across 75 days with 48 30-minute timeslots per day

## Design Highlights

- **Self-loops excluded**: Reduces data size ~30-40% without losing analytical power
- **Timeslot-level granularity**: Enables flexible temporal filtering post-preprocessing
- **Automated day classification**: Inferred from data patterns, user-confirmed
- **Parquet optimization**: Columnar storage for efficient I/O and pandas integration
- **Configuration-driven**: Day classification stored separately for easy adjustment

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting pull requests:
```bash
uv run pytest tests/
```

## License

[Add your license here - e.g., MIT, Apache 2.0, etc.]

## Citation

If you use this project or the LyMob dataset in research, please cite:

```bibtex
@article{lymob2024,
  doi = {10.5281/zenodo.14219563},
  title = {LyMob: Large-scale Mobility Dataset},
  year = {2024}
}
```

## Acknowledgments

- LyMob dataset team for the comprehensive urban mobility data
- NetworkX for graph analysis capabilities