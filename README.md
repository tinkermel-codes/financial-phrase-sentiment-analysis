# Financial Phrase Sentiment Analysis

A machine learning project for classifying financial news phrases into positive, negative and neutral sentiment categories.

## Project Overview
This project analyzes financial text snippets and evaluates multiple machine learning models on the Financial PhraseBank dataset. The goal is to compare models by the following metrics:
- Accuracy
- Macro Metrics (Precision, Recall, F1)
- F1-Scores on Class-Level
- Confusion matrices
- Misclassification patterns
- Class-wise Error Rates
- Learning Curves
- Feature Importances for Linear Models
- Model size
- Efficiency (performance per kilobyte)

This repository demonstrates end-to-end ML engineering skills:
- Exploratory Data Analysis
- Data Preprocessing
- Model Training
- Model Evaluation
- Result Visualization


## Business Motivation
Financial markets react instantly to news, reports and public sentiment. Analysts, traders and automated systems rely in fast, accurate interpretation of textual information to assess risk and identify opportunities. Manual sentiment evaluation is slow, inconsistent and impossible to scale across thousands of daily financial statements. 

This project addresses the gap by building models that can automatically classifiy financial phrases into positive, negative or neutral sentiment. Such systems support:
- Faster decision-making in trading and portfolio management
- Risk detection through early identification of negative sentiment
- Market monitoring at scale across news streams and reports


## Engineering Decisions
This project evaluates three classic machine learning models for financial sentiment classification: Logistic Regression, Linear SVM, and Naive Bayes. The design choices focus on interpretability, reproducibility, and fair comparison.

### Model Selection
The selected models are Logistic Regression, Linear SVM and Naive Bayes because they are strong baselines for TF-IDF-based text classification. Each model represents a different approach:
- Logistic Regression: probabilistic linear classifier
- Linear SVM: margin-maxing linear separator
- Naive Bayes: generative model with strong independece assumptions

### Preprocessing Strategy
A minimal preprocessing pipeline was used to preserve financial terminology:
- Lowercasing
- Tokenization
- Stopword removal
- TF-IDF vectorization

This ensures that differences in performance come from the models, not from heavy text normalization.

### Training and Experiment Setup
All models share the same:
- Train/Test split
- Preprocessing pipeline
- Random seed
This ensures a fair comparison.


## Dataset
This project uses the Financial PhraseBank dataset from Hugging Face:
**Dataset:** https://huggingface.co/datasets/takala/financial_phrasebank

The dataset contains short financial news phrases labeled by human annotators with either positive, negative or neutral sentiment.

The dataset it available in four possible configurations depending on the percentage of agreement of annotators. For this project, the dataset with an agreement rate of 100% between the annotators is used. It consists of 2264 samples. 


## Results & Visualizations
### Accuracy
Measures the overall proportion of correctly predicted sentiment labels across the dataset.

### Macro Metrics
Treat all sentiment classes equally and reveal how well the model performs under class imbalance.

### Class-Level F1 Scores
Show the balance between Precision and Recall for each sentiment class individually

### Confusion Matrices
Show how well each model distinguisches between positive, neutral and negative sentiment.

### Misclassifications
Highlight difficult sentiments and model weaknesses.

### Class-wise Error Rates
Indicate how often each sentiment class is predicted incorrectly.

### Learning Curves
Reveal underfitting or overfitting behavior and data efficiency.

### Feature Importances (Logistic Regression, Linear SVM)
Identify which n-grams most strongly influence sentiment predictions in linear models.

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
├── LICENSE.txt
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
