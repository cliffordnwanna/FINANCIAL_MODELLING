# PREDICTIVE_MODELLING
## Systemic Crisis, Banking Crisis, Inflation Crisis in Africa

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Objective](#objective)
- [Project Steps](#project-steps)
- [Data Preprocessing](#data-preprocessing)
  - [Handling Missing Values](#handling-missing-values)
  - [Handling Duplicates](#handling-duplicates)
  - [Outlier Detection and Treatment](#outlier-detection-and-treatment)
  - [Encoding Categorical Features](#encoding-categorical-features)
- [Modeling](#modeling)
  - [Train-Test Split](#train-test-split)
  - [Model Selection](#model-selection)
  - [Model Performance Evaluation](#model-performance-evaluation)
  - [Model Improvement Techniques](#model-improvement-techniques)
    - [Feature Selection](#feature-selection)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Cross-Validation](#cross-validation)
    - [Handling Class Imbalance](#handling-class-imbalance)
    - [Trying Different Algorithms](#trying-different-algorithms)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [How to Run the Project](#how-to-run-the-project)
- [Real-World Applications](#real-world-applications)
- [Visualizations and Model Comparison](#visualizations-and-model-comparison)
- [Screenshots](#screenshots)
- [Additional Code for Model Improvements](#additional-code-for-model-improvements)

---

## Project Overview
This project focuses on analyzing and predicting systemic crises across 13 African countries between 1860 and 2014. The aim is to build a machine learning model that can predict the emergence of a systemic crisis based on various economic indicators such as inflation rates, exchange rates, debt defaults, and more.

## Dataset Description
The dataset includes information on banking, financial, inflation, and systemic crises from 1860 to 2014 in the following African countries: Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia, and Zimbabwe.

### Dataset Columns:
- **country_number**: Numeric country identifier
- **country_code**: ISO code of the country
- **country**: Name of the country
- **year**: Year of observation
- **systemic_crisis**: Indicates whether a systemic crisis occurred (1: Yes, 0: No)
- **exch_usd**: Exchange rate against USD
- **domestic_debt_in_default**: Domestic debt in default
- **sovereign_external_debt_default**: External debt default
- **gdp_weighted_default**: GDP-weighted default rate
- **inflation_annual_cpi**: Annual inflation rate
- **independence**: Whether the country was independent in that year
- **currency_crises**: Occurrence of a currency crisis (1: Yes, 0: No)
- **inflation_crises**: Occurrence of an inflation crisis (1: Yes, 0: No)
- **banking_crisis**: Whether a banking crisis occurred (1: Yes, 0: No)

### Dataset Source: 
Kaggle - African Economic Crises Data Analysis (https://www.kaggle.com/code/ezzaldin6/african-economic-crises-data-analysis)
### Dataset description : 
This dataset focuses on the Banking, Debt, Financial, Inflation and Systemic Crises that occurred, from 1860 to 2014, in 13 African countries, including: Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia and Zimbabwe
### Dataset link : 
https://drive.google.com/file/d/1fTQ9R29kgAhInFO0HMqvkcAfSZWg6fCx/view

## Objective
The main goal of this project is to predict the likelihood of a systemic crisis in any of the 13 African countries using various economic features from the dataset.

## Project Steps
1. **Data Importation & Exploration**: Load the data and understand its structure using Pandas Profiling.
2. **Data Preprocessing**: Handle missing values, duplicates, outliers, and encode categorical variables.
3. **Model Selection & Training**: Train machine learning models using the preprocessed data.
4. **Model Evaluation**: Evaluate models based on accuracy, precision, recall, and F1-score.
5. **Model Improvement Techniques**: Use feature selection, hyperparameter tuning, and cross-validation to improve model performance.

---

## Data Preprocessing

### Handling Missing Values
No missing values were detected in the dataset.

### Handling Duplicates
Duplicates were checked and removed using `df.drop_duplicates()`.

### Outlier Detection and Treatment
Outliers in **exch_usd** (exchange rate) and **gdp_weighted_default** were detected using box plots and scatter plots. Winsorization and Z-score techniques were used to handle extreme values while preserving data integrity.

### Encoding Categorical Features
Label Encoding was applied to categorical columns like **country_code**, **country**, and **banking_crisis** using `LabelEncoder`.

---

## Modeling

### Train-Test Split
The data was split into training and testing sets:
- **Features**: All columns except `systemic_crisis`.
- **Target**: `systemic_crisis`.

The split was 80% for training and 20% for testing.

### Model Selection
We selected **RandomForestClassifier** as the initial model due to its robustness in handling various feature types and its ability to manage class imbalance.

### Model Performance Evaluation
- **Accuracy**: 96.34%
- Precision, Recall, and F1-scores were evaluated for both positive and negative classes, with good performance noted across the board, especially in detecting systemic crises.

---

## Model Improvement Techniques

### Feature Selection
RandomForest-based feature importance was used to select the most relevant features, improving model interpretability.

### Hyperparameter Tuning
We used **GridSearchCV** to fine-tune hyperparameters of the Random Forest model to find the best parameters.

### Cross-Validation
**StratifiedKFold** cross-validation was used to maintain class balance during training and validation.

### Handling Class Imbalance
**SMOTE** (Synthetic Minority Oversampling Technique) was applied to address class imbalance, especially for under-represented classes.

### Trying Different Algorithms
We also experimented with other models:
- **Logistic Regression**
- **Support Vector Machines (SVM)**

RandomForest outperformed both models in terms of accuracy and handling class imbalance.

---

## Conclusion
- The RandomForest model achieved a high accuracy of 96.34% and performed well across precision, recall, and F1-score metrics.
- Class imbalance remains a challenge, but techniques like SMOTE helped improve the model’s ability to detect minority-class instances.
- The predictive model offers valuable insights for early warning systems and economic policy decisions in African countries.

---

## Future Work
- Further optimize hyperparameters to improve model performance.
- Experiment with advanced techniques for handling class imbalance, such as undersampling or custom loss functions.
- Explore neural network models for potentially better predictions.

---

## How to Run the Project
1. **Clone the repository** and install the required dependencies using `requirements.txt`.
2. **Download the dataset** from Kaggle.
3. **Run the main notebook or script** to execute the data analysis and model training.
4. **Review the output** for performance metrics, visualizations, and predictions.

---

## Real-World Applications
- **Early Warning Systems**: Governments can anticipate crises using these models, adjusting policies proactively.
- **Policy Decision-Making**: Central banks may adjust inflation targeting or currency controls based on model insights.
- **Investment Risk Assessment**: Investors can make informed decisions based on the predicted likelihood of financial crises.
- **Credit Rating Agencies**: Credit assessments can be improved by factoring in these predictions.
- **International Aid Allocation**: Organizations like the IMF can allocate resources proactively to countries at risk.

---

## Visualizations and Model Comparison

### Feature Importance Visualization
This plot helps visualize which features are most important in predicting systemic crises.

```python
import matplotlib.pyplot as plt
import seaborn as sns

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(y=features.columns[indices], x=importances[indices])
plt.title("Feature Importance")
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()



This bar chart compares the accuracy of different models.

python
Copy code
import matplotlib.pyplot as plt

model_names = ['Random Forest', 'Logistic Regression', 'SVM']
accuracies = [0.9634, 0.9487, 0.9345]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Comparison: Accuracy Scores')
plt.ylabel('Accuracy')
plt.show()
Confusion Matrix Visualization
To visualize model performance, confusion matrices can be plotted for each model.

python
Copy code
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix - Random Forest')
plt.show()


### Screenshots
Pandas Profiling Report:
Screenshot of summary statistics and missing values.

Feature Importance Plot:
Screenshot showing the most relevant features for prediction.

Confusion Matrix:
Visualize classification performance using confusion matrices.

Model Comparison Plot: Summary of model accuracy for RandomForest, Logistic Regression, and SVM.

Additional Code for Model Improvements
Hyperparameter Tuning
python
Copy code
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
Cross-Validation
python
Copy code
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores)}")
Conclusion
The RandomForest model demonstrates strong predictive performance with a 96.34% accuracy. Future enhancements could include more advanced techniques for handling class imbalance and exploring neural network models to improve predictions. This project has real-world implications for policy-making, risk assessment, and economic stability in African countries.

css
Copy code

This version is organized, formatted, and includes the necessary code snippets and instructions for anyone to understand and run your project.








