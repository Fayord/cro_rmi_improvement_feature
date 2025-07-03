# Risk Network Visualization System

An interactive web-based application that transforms complex risk assessment data into intuitive network visualizations using Dash and Cytoscape.js.

## Project Structure

```
network_analyzer/
├── app/                    # Dash application
│   ├── main.py            # App entry point
│   ├── layout.py          # Main layout
│   ├── callbacks.py       # Dash callbacks
│   └── assets/            # Static files (CSS, JS)
├── core/                  # Business logic
│   ├── models.py          # Data models
│   ├── services.py        # Business services
│   └── utils.py           # Utility functions
├── data/                  # Raw data storage
│   ├── raw/               # Raw Excel files
│   └── processed/         # Processed data files
├── data_processor/        # Data processing and management
│   ├── processing.py      # Data processing
│   └── storage.py         # Data storage
├── ai/                    # AI components (Phase 2)
│   ├── classifier.py      # Risk relationship classification
│   └── cache.py           # AI response caching
├── visualization/         # Graph visualization
│   ├── layouts.py         # Graph layout algorithms
│   └── styling.py         # Graph styling
├── config/                # Configuration
│   └── settings.py        # App settings
├── tests/                 # Tests
└── tasks/                 # Task management
```

## Development Phases

### Phase 1: Data Preparation - Current Focus
- Postprocess risk data
- Create risk network data structures
- Implement AI-powered risk relationship classification

### Phase 2: MVP Visualization - Next
- Create main graph visualization with modular architecture
- Create subgraph visualization capabilities
- Implement visualization toggles and controls

### Phase 3: Polish and Enhancement - Future
- Enhance main graph styling and interactions
- Enhance subgraph analysis capabilities
- Implement click-to-popup data displays and navigation

### Phase 4: Graph Condensation - Future
- Implement LLM-based graph condensation
- Add algorithmic graph condensation (Leiden, etc.)
- Create hybrid condensation approaches

## Getting Started

1. Install PDM (if not already installed):
```bash
pip install pdm
```

2. Install dependencies:
```bash
pdm install
```

3. Run the application:
```bash
pdm run python main.py
```

4. Open browser to `http://localhost:8050`

## Development Workflow

### Install development dependencies:
```bash
pdm install -G dev
```

### Run tests:
```bash
pdm run pytest
```

### Format code:
```bash
pdm run black .
pdm run isort .
```

### Lint code:
```bash
pdm run flake8 .
```

### Install production dependencies:
```bash
pdm install -G prod
```

## Growth Strategy

This structure is designed to grow organically:
- Start with single files for each component
- Split files when they exceed 500 lines
- Create subdirectories when you have 3+ related files
- Add new top-level folders for major features

See `.cursor/rules/project_structure.mdc` for detailed guidelines.
