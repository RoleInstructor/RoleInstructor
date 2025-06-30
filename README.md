## Getting Started

### Using the MEF Framework
To launch the MEF framework application:
```bash
cd app/
python app.py
```

### Generating Evaluation-Interaction Data Pairs
```bash
cd src/
python <metric_name>_eval.py  # Replace <metric_name> with your target metric
```

### Generating Updated Interactions with RoleInstructor
```bash
cd src/
python run_roleinstructor.py
```

### Directory Structure
```
├── app/               # MEF framework application
│   └── app.py         # Main application entry point
├── src/               # Data processing scripts
│   ├── *_eval.py      # Evaluation pair generators (e.g., accuracy_eval.py)
│   ├── run_roleinstructor.py  # Interaction data updater
│   └── ...                # Other project files
├── data/               # Datasource
├── scripts/               # All game scripts used in this work
└── models/                # RoleInstructor models 
```
## Note
1. Ensure all dependencies are installed before running scripts
2. Replace <metric_name> in evaluation scripts with your specific metric name
3. Input/output configurations may require adjustment in script parameters
