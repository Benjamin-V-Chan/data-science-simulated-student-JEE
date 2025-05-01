# data-science-simulated-student-JEE

## Project Overview

This repository implements a complete data science pipeline to predict whether a student will drop out after Class 12 using a simulated JEE student dataset. The workflow covers data inspection, preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and feature importance analysis.

## Folder Structure

```
project-root/
├── data/
│   └── JEE_Dropout_After_Class_12.csv
├── scripts/
│   ├── 01_data_inspection.py
│   ├── 02_data_preprocessing.py
│   ├── 03_eda_visualization.py
│   ├── 04_feature_engineering.py
│   ├── 05_model_training.py
│   ├── 06_evaluation.py
│   └── 07_feature_importance.py
└── outputs/
    ├── data_summary.csv
    ├── data_splits.pkl
    ├── data_fe_splits.pkl
    ├── preprocessor.pkl
    ├── figures/
    └── models/
```

## Usage

1. Setup the Project:

   - Clone the repository.
   - Ensure you have Python installed.
   - Install required dependencies using the requirements.txt file.
     ```bash
     pip install -r requirements.txt
     ```

2. Inspect the raw data
   ```bash
   python scripts/01_data_inspection.py
   ```

3. Preprocess and split
   ```bash
   python scripts/02_data_preprocessing.py
   ```

4. Generate exploratory visualizations
   ```bash
   python scripts/03_eda_visualization.py
   ```

5. Create additional features
   ```bash
   python scripts/04_feature_engineering.py
   ```

6. Train and tune models
   ```bash
   python scripts/05_model_training.py
   ```

7. Evaluate model performance
   ```bash
   python scripts/06_evaluation.py
   ```

8. Visualize top feature importances
   ```bash
   python scripts/07_feature_importance.py
   ```

Outputs will be saved under `outputs/` (e.g., figures in `outputs/figures` and models in `outputs/models`).

## Requirements

- pandas
- scikit-learn
- matplotlib
- joblib
- numpy

## Acknowledgments

**Dataset name:** Simulated Dataset: JEE Dropout After Class 12  
**Dataset author:** Jayanta Nath  
**Dataset source:** https://www.kaggle.com/datasets/jayaantanaath/simulated-dataset-jee-dropout-after-class-12