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
├── data/                  # Data management
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

### Phase 1 (MVP) - Current Focus
- Basic network visualization
- Company selection
- Simple filtering
- Static data

### Phase 2 (AI Integration) - Future
- AI-powered relationship classification
- Dynamic edge generation
- Subgraph analysis

### Phase 3 (Enhancement) - Future
- Advanced layouts
- Export functionality
- Performance optimization

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

3. Open browser to `http://localhost:8050`

## Growth Strategy

This structure is designed to grow organically:
- Start with single files for each component
- Split files when they exceed 500 lines
- Create subdirectories when you have 3+ related files
- Add new top-level folders for major features

See `.cursor/rules/project_structure.mdc` for detailed guidelines.
