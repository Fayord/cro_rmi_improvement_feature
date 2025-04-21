# cro_rmi_improvement_feature

A tool for validating and improving risk management descriptions.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Fayord/cro_rmi_improvement_feature.git
cd cro_rmi_improvement_feature
```

2. Install PDM (if not already installed):
```bash
pip install pdm
```

3. Install project dependencies:
```bash
pdm install
```

4. Activate the virtual environment:
```bash
pdm venv activate
```

## Usage

You can run the project in two ways:

### Using the Jupyter Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `src/cro_rmi_improvement_feature/rmi_risk_validate_feedback/notebook.ipynb`

### Using the Python Script

Run the main script directly:
```bash
python src/cro_rmi_improvement_feature/rmi_risk_validate_feedback/main.py
```

## For clearing cache
```from langchain.globals import clear_llm_cache
clear_llm_cache() 
```
## Note

Make sure to set up your environment variables (like API keys) as shown in the notebook examples before running the code.