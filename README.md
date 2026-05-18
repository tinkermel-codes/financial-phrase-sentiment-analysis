# Financial Phrase Sentiment Analysis

A machine learning project for classifying financial news phrases into positive, negative and neutral sentiment categories.

## Project Overview
This project analyzes financial text snippets and evaluates multiple machine learning models on the Financial PhraseBank dataset. The goal is to compare models by the following metrics:
- Accuracy
- Confusion matrices
- Misclassification patterns
- Model size
- Efficiency (performance per kilobyte)

This repository demonstrates end-to-end ML engineering skills:
- Exploratory Data Analysis
- Data Preprocessing
- Model Training
- Model Evaluation
- Result Visualization


## Dataset
This project uses the Financial PhraseBank dataset from Hugging Face:
**Dataset:** https://huggingface.co/datasets/takala/financial_phrasebank

The dataset contains short financial news phrases labeled by human annotators with either positive, negative or neutral sentiment.

The dataset it available in four possible configurations depending on the percentage of agreement of annotators. For this project, the dataset with an agreement rate of 100% between the annotators is used. It consists of 2264 samples. 


## Models
The following classical machine learning models were trained and evaluated:
- Logistic Regression
- Linear Support Vectore Machine (SVM)
- Naive Bayes

Each model is evaluated using:
- Accuracy
- Macro-F1
- Confusion Matrix
- Misclassification analysis
- Model size (KB)
- Efficiency metric: Macro-F1 / Model size (KB)


## Results & Visualizations
### Confusion Matrices
Show how well each model distinguisches between positive, neutral and negative sentiment.

### Learning Curves
Reveal underfitting/overfitting behavior and data efficiency.

### Misclassifications
Highlight difficult sentiments and model weaknesses.

### Model Size & Efficiency
Compare performance relative to model size.


## Project Structure
```
financial-phrase-sentiment-analysis/
│
├── configs/
│   ├── config.yaml
│
├── data/
│   ├── processed/
│   ├── raw/
│
├── models/
│   ├── linear_svm/
|   │   ├── config.yaml
|   │   ├── confusion_matrix.csv
|   │   ├── feature_importances.csv
|   │   ├── learning_curve.csv
|   │   ├── metrics.json
|   │   ├── misclassifications.csv
|   │   ├── model.pkl
|   |
│   ├── logistic_regression/
|   │   ├── config.yaml
|   │   ├── confusion_matrix.csv
|   │   ├── feature_importances.csv
|   │   ├── learning_curve.csv
|   │   ├── metrics.json
|   │   ├── misclassifications.csv
|   │   ├── model.pkl
|   |
│   ├── naive_bayes/
|   │   ├── config.yaml
|   │   ├── confusion_matrix.csv
|   │   ├── learning_curve.csv
|   │   ├── metrics.json
|   │   ├── misclassifications.csv
|   │   ├── model.pkl
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_results_visualization.ipynb
│
├── src/
│   ├── data/
│   │   ├── clean_data.py
│   │   ├── make_dataset.py
│   │
│   ├── features/
│   │   ├── features.py
│   │
│   ├── models/
│   │   ├── evaluate_model.py
│   │   ├── feature_importance.json
│   │   ├── train_model.py
│   │
│   ├── utils/
│   │   ├── loaders.py
│   │   ├── model_plotter.py
│   │   ├── model_repository.py
│   │   ├── utils.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## How to Run the Project
This project is designed to be run interactively through Jupyter notebooks. Follow the steps below to install dependencies, configure the pipeline and run the experiments.

### 1. Install Dependencies
Create a virtual environment (recommended) and install all required packages:
```
pip install -r requirements.txt
```

### 2. Configure the Pipeline (optional)
All preprocessing and feature-extraction settings are stored in:
```
config/config.yaml
```

You can adjust:
#### Data settings:
```
data:
  raw_path: "../data/raw/Sentences_AllAgree.txt"
  preprocessed_path: "../data/processed/clean_data.csv"
  text_column: "clean_text"
  label_column: "label"
```

#### Training settings:
```
training:
  test_size: 0.2
  random_state: 42
```

#### TF-IDF Vectorizer:
```
tfidf:
  max_features: 3000
  ngram_range: [1, 2]
```

These parameters control data loading, preprocessing, train/test split and feature extraction.

### 3: Run the Notebooks
To reproduce all results, simply open the notebooks in the notebooks/ directory and execute them in order.
Each notebook performs ine stage of the workflow:
- 01_exploratory_data_analysis.ipynb - explorative data analysis
- 02_preprocessing.ipynb - data cleaning, preprocessing
- 03_modeling.ipynb - model training, storing model performances
- 04_model_comparison.ipynb - comparison of model performances
- 05_results_visualization.ipynb - visualization of the results

All generated outputs are automatically saved inside the corresponding model folder under:
```
models/<model_name>/
```

## License
This project is licensed under the MIT License.  
See the **LICENSE.txt** for details.
