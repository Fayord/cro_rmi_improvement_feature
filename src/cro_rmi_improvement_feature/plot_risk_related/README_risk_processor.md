# Risk Data Processor

A modular Python script for processing risk assessment data from Excel files and generating structured risk data for analysis and visualization.

## Overview

This module was extracted from the Jupyter notebook `etl_data_250528_no_risk_desc.ipynb` and provides a clean, reusable interface for processing risk assessment data. The `RiskDataProcessor` class encapsulates all the functionality into modular methods that can be used independently or as part of a complete workflow.

## Features

- **Data Loading**: Load risk assessment data from Excel files
- **Risk Scoring**: Calculate risk scores and levels based on likelihood and impact
- **Data Aggregation**: Aggregate risk data using different methods (MAX or RMI)
- **Data Cleaning**: Process and clean complex data structures
- **Catalog Integration**: Merge with risk catalog data
- **Export**: Save processed data to JSON format
- **Flexible Configuration**: Customize risk categories and processing parameters

## Installation

Ensure you have the required dependencies:

```bash
pip install pandas openpyxl
```

## Quick Start

### Basic Usage

```python
from risk_data_processor import RiskDataProcessor

# Initialize the processor
processor = RiskDataProcessor()

# Process data with default settings
result = processor.process_complete_workflow(
    assessment_file="data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx",
    catalog_file="data/RMI-V2-Translate_20250508.xlsx",
    output_file="result/company_risk_data.json",
    companies=["PCG"],
    aggregation_method="RMI"
)

print(f"Processed {len(result)} risk records")
```

### Custom Configuration

```python
# Define custom risk categories
custom_categories = ["Operational Risk", "Strategic Risk", "Credit Risk"]

# Initialize with custom categories
processor = RiskDataProcessor(selected_risk_categories=custom_categories)

# Process with MAX aggregation method
result = processor.process_complete_workflow(
    assessment_file="data/assessment.xlsx",
    catalog_file="data/catalog.xlsx",
    output_file="result/output.json",
    aggregation_method="MAX"
)
```

## API Reference

### RiskDataProcessor Class

#### Constructor

```python
RiskDataProcessor(selected_risk_categories: Optional[List[str]] = None)
```

**Parameters:**
- `selected_risk_categories`: List of risk categories to include. Defaults to:
  - Operational Risk
  - Strategic Risk
  - Credit Risk
  - Market Risk
  - Liquidity Risk

#### Methods

##### `load_assessment_data(file_path: str, company_name: str = "PCG") -> pd.DataFrame`

Load risk assessment data from Excel file.

**Parameters:**
- `file_path`: Path to the Excel file
- `company_name`: Name of the company (default: "PCG")

**Returns:** DataFrame with loaded data

##### `calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame`

Calculate risk scores and levels for the dataset.

**Parameters:**
- `df`: Input DataFrame with likelihood and impact columns

**Returns:** DataFrame with calculated risk scores and levels

##### `aggregate_risk_data(df: pd.DataFrame, method: str = "RMI") -> pd.DataFrame`

Aggregate risk data by company and risk name.

**Parameters:**
- `df`: Input DataFrame
- `method`: Aggregation method ("MAX" or "RMI")

**Returns:** Aggregated DataFrame

##### `process_complete_workflow(...) -> List[Dict]`

Execute the complete risk data processing workflow.

**Parameters:**
- `assessment_file`: Path to risk assessment Excel file
- `catalog_file`: Path to risk catalog Excel file
- `output_file`: Path to save output JSON file
- `companies`: List of companies to include (optional)
- `aggregation_method`: Method for aggregating risk data ("MAX" or "RMI")

**Returns:** List of dictionaries containing processed risk data

## Data Processing Workflow

The complete workflow consists of the following steps:

1. **Load Assessment Data**: Read Excel file and add metadata
2. **Calculate Risk Scores**: Compute risk scores and levels
3. **Rename Columns**: Standardize column names
4. **Aggregate Data**: Group by company and risk name
5. **Filter Companies**: Select specific companies (optional)
6. **Select Columns**: Choose output columns
7. **Process Structures**: Clean complex data structures
8. **Merge Columns**: Combine rootcause and process data
9. **Load Catalog**: Read risk catalog data
10. **Add Catalog Risks**: Include catalog risks in output
11. **Save Results**: Export to JSON format

## Aggregation Methods

### RMI Method (Default)
- Sorts data by risk score and impact
- Takes the maximum risk score for each risk
- Recalculates likelihood based on max score and impact

### MAX Method
- Takes maximum values for likelihood, impact, and risk level
- Recalculates risk score from max likelihood and impact

## Input Data Format

### Assessment Data Excel File
Should contain columns:
- `Risk Category`: Risk category (e.g., "Operational Risk")
- `Risk Item`: Risk name
- `Risk Description`: Risk description
- `Root Cause`: Root cause information
- `Process`: Process information
- `likelihood_combined`: Likelihood score
- `impact_combined`: Impact score

### Risk Catalog Excel File
Should contain sheets:
- `Risks`: Risk definitions with columns:
  - `Risk-EN`: Risk name in English
  - `Description-EN`: Risk description
  - `Risk-category`: Risk category
- `Risk_Cause_mapping`: Risk-cause mappings with columns:
  - `RiskName`: Risk name
  - `RiskCause`: Root cause

## Output Format

The processed data is saved as a JSON file containing a list of dictionaries with the following structure:

```json
[
  {
    "company": "PCG",
    "risk_cat": "Operational Risk",
    "risk": "Business interruption from fire hazards",
    "risk_desc": "เกิดเพลิงไหม้อาคารคลังสินค้า...",
    "rootcause": "rootcause :Negligence or Human Error...",
    "process": "process :Distribution Center (DC): -...",
    "risk_level": 2
  }
]
```

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic usage with default settings
- Custom risk categories
- Step-by-step processing
- Data analysis
- Error handling

## Error Handling

The module includes comprehensive error handling for:

- Missing input files
- Invalid aggregation methods
- Data processing errors
- File I/O errors

## Dependencies

- `pandas`: Data manipulation and analysis
- `openpyxl`: Excel file reading
- `json`: JSON file operations
- `os`: File system operations
- `typing`: Type hints
- `warnings`: Warning suppression

## Notes

- The module suppresses pandas FutureWarning messages about deprecated methods
- All text processing preserves Unicode characters (Thai, Chinese, etc.)
- The module automatically creates output directories if they don't exist
- Risk levels are calculated on a scale of 0-4 based on risk scores

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure input Excel files exist and paths are correct
2. **Missing Columns**: Verify that input files contain all required columns
3. **Memory Issues**: For large datasets, consider processing in chunks
4. **Encoding Issues**: Ensure Excel files are saved with proper encoding

### Debug Mode

To enable debug output, you can modify the print statements in the code or add logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When modifying the code:

1. Maintain backward compatibility
2. Add type hints for new functions
3. Update documentation for new features
4. Add tests for new functionality
5. Follow the existing code style 