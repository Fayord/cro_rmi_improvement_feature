# Risk Data Processor

A Python module for processing risk assessment data from Excel files and generating standardized JSON output.

## Overview

This module extracts the data processing logic from the Jupyter notebook `etl_data_250528.ipynb` and provides a functional interface for processing risk assessment data. It supports:

- Loading risk assessment data from Excel files
- Processing risk catalog data
- Company-specific preprocessing (currently supports PCG)
- Risk score calculation and level determination
- Data aggregation and transformation
- JSON output generation

## Features

- **Company-specific processing**: Different preprocessing logic for different companies
- **Flexible input**: Works with various Excel file formats
- **Risk scoring**: Calculates risk scores and determines risk levels
- **Data aggregation**: Aggregates risk data using RMI or MAX methods
- **Catalog integration**: Merges with risk catalog data
- **JSON output**: Generates standardized JSON output

## Installation

Ensure you have the required dependencies:

```bash
pip install pandas openpyxl
```

## Usage

### Basic Usage

```python
from risk_data_processor import process_risk_data

# Process risk data
result = process_risk_data(
    risk_data_path="data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx",
    catalog_data_path="data/RMI-V2-Translate_20250508.xlsx",
    company_name="PCG",
    output_path="result/risk_data.json"
)

print(f"Processed {len(result)} risk records")
```

### Function Parameters

#### `process_risk_data()`

- **risk_data_path** (str): Path to the risk assessment Excel file
- **catalog_data_path** (str): Path to the risk catalog Excel file
- **company_name** (str): Name of the company for preprocessing
- **output_path** (str, optional): Path to save JSON output

#### Returns

- **List[Dict]**: List of processed risk data dictionaries

### Output Format

Each risk record contains:

```json
{
  "company": "PCG",
  "risk_cat": "Operational Risk",
  "risk": "Business interruption from fire hazards",
  "risk_desc": "The business interruption due to a fire incident...",
  "risk_level": 2,
  "rootcause": "rootcause :Negligence or Human Error in Handling Flammable Materials...",
  "process": "process :Maintenance: -, Engineering: -, Production: -..."
}
```

## Company-Specific Processing

### PCG Company

The module includes specific preprocessing for PCG company data:

- Adds company column
- Calculates risk scores from likelihood and impact
- Renames columns to standardized format
- Determines risk levels

### Generic Companies

For other companies, the module provides generic preprocessing:

- Adds company column
- Calculates risk scores if required columns exist
- Basic data structure handling

## Risk Categories

The module processes the following risk categories:

- Operational Risk
- Strategic Risk
- Credit Risk
- Market Risk
- Liquidity Risk

## Risk Level Calculation

Risk levels are determined based on risk scores:

- **Level 4**: Risk score ≥ 20
- **Level 3**: Risk score 10-16
- **Level 2**: Risk score 4-9
- **Level 1**: Risk score 1-3 or score 4 with impact 2
- **Level 0**: Risk score 0

## Data Aggregation Methods

### RMI Method (Default)

- Sorts by company, risk, risk score, and impact
- Takes maximum risk score for each risk
- Recalculates likelihood from risk score and impact

### MAX Method

- Groups by company and risk
- Takes maximum values for risk level, likelihood, and impact
- Recalculates risk score from maximum likelihood and impact

## Example Usage

See `example_usage.py` for complete examples.

### Processing PCG Data

```python
from risk_data_processor import process_risk_data

result = process_risk_data(
    risk_data_path="data/250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx",
    catalog_data_path="data/RMI-V2-Translate_20250508.xlsx",
    company_name="PCG",
    output_path="result/pcg_risk_data.json"
)
```

### Processing Generic Company Data

```python
result = process_risk_data(
    risk_data_path="data/generic_risk_assessment.xlsx",
    catalog_data_path="data/RMI-V2-Translate_20250508.xlsx",
    company_name="GENERIC_COMPANY",
    output_path="result/generic_risk_data.json"
)
```

## File Structure

```
plot_risk_related/
├── risk_data_processor.py      # Main processing module
├── example_usage.py            # Usage examples
├── README_risk_processor.md    # This documentation
└── etl_data_250528.ipynb      # Original notebook
```

## Input File Requirements

### Risk Assessment File

Should contain columns:
- Risk Category
- Risk Item
- Risk Description
- Root Cause
- Process
- likelihood_combined
- impact_combined

### Risk Catalog File

Should contain sheets:
- **Risks**: Risk definitions and descriptions
- **Risk_Cause_mapping**: Mapping between risks and root causes

## Error Handling

The module includes comprehensive error handling:

- File existence checks
- Data validation
- Exception handling with detailed error messages
- Graceful fallbacks for missing data

## Dependencies

- pandas: Data manipulation
- openpyxl: Excel file reading
- json: JSON output generation
- os: File path handling

## Contributing

To add support for new companies:

1. Create a new preprocessing function following the pattern of `preprocess_pcg_data()`
2. Add the company name condition in `process_risk_data()`
3. Update documentation

## License

This module is part of the CRO RMI Improvement Feature project. 