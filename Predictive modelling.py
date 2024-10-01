# -*- coding: utf-8 -*-
"""GOMYCODE CHECKPOINT 22 [supervised learning classification].ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13j3anQUA1h4ECeYuUGvwOVFUErwZ8PQZ

# Project Title: Systemic Crisis, Banking Crisis, Inflation Crisis in Africa - Predictive Modeling
I am going to work on the 'Systemic Crisis, Banking Crisis, inflation Crisis In Africa' dataset that was provided by Kaggle.The ML model objective is to predict the likelihood of a Systemic crisis emergence given a set of indicators like the annual inflation rates.

Dataset description : This dataset focuses on the Banking, Debt, Financial, Inflation and Systemic Crises that occurred, from 1860 to 2014, in 13 African countries, including: Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia and Zimbabwe. The dataset contains the following columns: country_number, country_code, country	year,	systemic_crisis, exch_usd, domestic_debt_in_default	, sovereign_external_debt_default	, gdp_weighted_default, inflation_annual_cpi,  independence, currency_crises, inflation_crises & banking_crisis.

Dataset link : https://drive.google.com/file/d/1fTQ9R29kgAhInFO0HMqvkcAfSZWg6fCx/view

STEPS:

Import the data and perform basic data exploration phase.

Display general information about the dataset.

Create a pandas profiling reports to gain insights into the dataset.

Handle Missing and corrupted values.

Remove duplicates, if they exist.

Handle outliers, if they exist.

Encode categorical features.

Select your target variable and the features.

Split your dataset to training and test sets.

Based on the data exploration phase, select a ML classification algorithm and train it on the training set.

Assess the model's performance on the test set using relevant evaluation metrics.

Discuss alternative ways to improve the model's performance.
"""

!pip install ydata_profiling

!pip uninstall tensorflow

#import libraries
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#import dataset
from google.colab import drive
drive.mount('/content/drive')

#load dataset
df = pd.read_csv("/content/drive/MyDrive/Untitled folder/DATASETS/African_crises_dataset.csv")

# Make a copy of the original dataframe
df1 = df.copy()

df.head()

df.info

df.describe()

#Create a pandas profiling reports to gain insights into the dataset


# Create a Pandas Profiling report
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)


# Save the report as an HTML file
profile.to_file("ProfileReport.html")

# Display and explore the report
profile

"""#DATA PREPROCESSING"""

# Handle Missing and corrupted values
# Check for missing values
df.isnull().sum()
# There are no missing values in the dataframe

# Step 5: Remove duplicates, if they exist
#df.duplicated().count
df = df.drop_duplicates()

"""Handle outliers, if they exist."""

#Visualization Methods: Box plots, scatter plots, or histograms can visually highlight potential outliers.

# Box plot for outlier visualization of the 5 numerical columns to identify outliers
import seaborn as sns
#sns.boxplot(x=df['systemic_crisis']) # it has few outliers that can be ignored/handled
#sns.boxplot(x=df['domestic_debt_in_default']) # it has few outliers that can be ignored/handled
#sns.boxplot(x=df['inflation_annual_cpi']) # it has few outliers that can be ignored/handled
#sns.boxplot(x=df['sovereign_external_debt_default']) # it has few outliers that can be ignored/handled
#sns.boxplot(x=df['currency_crises']) # it has few outliers that can be handled
#sns.boxplot(x=df['gdp_weighted_default']) # contains outliers that must be handled
sns.boxplot(x=df['exch_usd']) # contains outliers that must be handled

import matplotlib.pyplot as plt

# Create a scatter plot for 'exch_usd'
plt.figure(figsize=(8, 6))
plt.scatter(range(len(df['exch_usd'])), df['exch_usd'], alpha=0.7, color='blue')
plt.title('Scatter Plot of exch_usd')
plt.xlabel('Index')
plt.ylabel('exch_usd values')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Create a scatter plot for 'gdp_weighted_default'
plt.figure(figsize=(8, 6))
plt.scatter(range(len(df['gdp_weighted_default'])), df['gdp_weighted_default'], alpha=0.7, color='blue')
plt.title('Scatter Plot of gdp_weighted_default')
plt.xlabel('Index')
plt.ylabel('gdp_weighted_default')
plt.grid(True)
plt.show()

from scipy.stats import mstats

# Calculate the 95th percentile value
percentile_95_gdp = df['gdp_weighted_default'].quantile(0.95)
percentile_95_exch = df['exch_usd'].quantile(0.95)

print(percentile_95_gdp)
print(percentile_95_exch)

# Apply Winsorization to cap outliers at the 95th percentile
df['gdp_weighted_default'] = mstats.winsorize(df['gdp_weighted_default'], limits=[None, 0.05])
df['gdp_weighted_default'] = mstats.winsorize(df['gdp_weighted_default'], limits=[None, 0.05])

# Define a function to handle outliers using the Z-score method
def handle_outliers_zscore(df, columns, z_threshold=2):
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[(z_scores > z_threshold)]
        df = df[(z_scores <= z_threshold)]
        print(f"Outliers removed in column '{col}': {len(outliers)}")
    return df

# Define the columns you want to handle outliers for
columns_to_handle_outliers = ['gdp_weighted_default']

# Handle outliers in the specified columns
df_cleaned = handle_outliers_zscore(df, columns_to_handle_outliers)

# df_cleaned now contains the DataFrame with outliers removed

# i will proceed with further analysis and modeling using df_cleaned

# Define a function to handle outliers using the Z-score method
def handle_outliers_zscore(df, columns, z_threshold=1):
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[(z_scores > z_threshold)]
        df = df[(z_scores <= z_threshold)]
        print(f"Outliers removed in column '{col}': {len(outliers)}")
    return df

# Define the columns you want to handle outliers for
columns_to_handle_outliers = ['exch_usd' ]

# Handle outliers in the specified columns
df_cleaned = handle_outliers_zscore(df, columns_to_handle_outliers)

# df_cleaned now contains the DataFrame with outliers removed

# i will proceed with further analysis and modeling using df_cleaned

#Scatter chart after outliers have been handled

# Create a scatter plot for 'gdp_weighted_default'
plt.figure(figsize=(8, 6))
plt.scatter(range(len(df_cleaned['gdp_weighted_default'])), df_cleaned['gdp_weighted_default'], alpha=0.7, color='blue')
plt.title('Scatter Plot of gdp_weighted_default')
plt.xlabel('Index')
plt.ylabel('gdp_weighted_default')
plt.grid(True)
plt.show()

# Create a scatter plot for 'exch_usd'
plt.figure(figsize=(8, 6))
plt.scatter(range(len(df_cleaned['exch_usd'])), df_cleaned['exch_usd'], alpha=0.7, color='blue')
plt.title('Scatter Plot of exch_usd')
plt.xlabel('Index')
plt.ylabel('exch_usd values')
plt.grid(True)
plt.show()

df_cleaned.describe()

# Encode categorical features

# df_cleaned is the DataFrame and categorical_columns are the columns to be encoded
#I intend to replace the original categorical columns with their encoded values by using ".loc"

label_encoder = LabelEncoder()
categorical_columns = ['country_code', 'country', 'banking_crisis']

for col in categorical_columns:
    df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])

df_cleaned.info()

# Select your target variable and the features
target_variable = 'systemic_crisis'
features = df_cleaned.drop(target_variable, axis=1)
target = df_cleaned[target_variable]

# Split your dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Select a ML classification algorithm and train it on the training set
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Assess your model performance on the test set using relevant evaluation metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print("Classification Report:")
print(classification_rep)

"""The evaluation metrics from the classification report provide insights into your model's performance:

Accuracy Score: The accuracy score indicates the proportion of correctly classified samples. In your case, the model achieved an accuracy of approximately 96.34%, which suggests that it correctly predicted the class labels for around 96.34% of the test set samples.

Precision and Recall: Precision represents the proportion of correctly predicted positive instances out of all instances predicted as positive, while recall indicates the proportion of correctly predicted positive instances out of all actual positive instances.

Class 0 (Negative class): The precision and recall for class 0 are both high (around 98% and 98%, respectively). This suggests that the model performs well in correctly identifying negative instances (class 0).

Class 1 (Positive class): The precision for class 1 is 80%, indicating that among instances predicted as positive, 80% were actually positive. The recall for class 1 is 75%, showing that the model correctly identified 75% of the actual positive instances.

F1-Score: The F1-score is the harmonic mean of precision and recall and provides a balance between the two metrics. The weighted average F1-score is around 96% for your model.

Support: Indicates the number of samples in each class.

Macro and Weighted Averages: Macro-average calculates metrics independently for each class and then takes the average, giving each class equal weight. Weighted average calculates metrics for each class and weights them by the number of true instances for each class.

#Interpretation:
Overall, your model performs well with high accuracy and good precision and recall for both classes.

Class 0 (the majority class) has excellent precision and recall.

Class 1 (the minority class) has a slightly lower precision and recall but still shows reasonable performance, considering the class imbalance.

"""

#Below is a Python code that elaborates on some techniques to improve model performance,
#including feature selection, hyperparameter tuning, cross-validation, handling class imbalance,
#and trying different algorithms:

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Feature Selection
# Using RandomForestClassifier for feature selection
feature_selector = SelectFromModel(RandomForestClassifier(random_state=42))
feature_selector.fit(X_train, y_train)
selected_features = feature_selector.transform(X_train)
selected_features_test = feature_selector.transform(X_test)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(selected_features, y_train)
best_params = grid_search.best_params_

# Cross-validation using StratifiedKFold
model = RandomForestClassifier(**best_params)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kfold.split(selected_features, y_train):
    X_train_cv, X_val = selected_features[train_idx], selected_features[val_idx]
    y_train_cv, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model.fit(X_train_cv, y_train_cv)
    y_pred_cv = model.predict(X_val)
    cv_accuracy = accuracy_score(y_val, y_pred_cv)
    cv_scores.append(cv_accuracy)

# Dealing with Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(selected_features, y_train)

# Trying Different Algorithms
models_to_try = {
    'Random Forest': RandomForestClassifier(**best_params),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC()
}

for name, clf in models_to_try.items():
    clf.fit(X_train_resampled, y_train_resampled)
    y_pred_test = clf.predict(selected_features_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Model: {name}, Test Accuracy: {test_accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test))

"""#INTERPRETATIONS

Random Forest nterpretation:
The Random Forest model shows similar performance to the Logistic Regression model in predicting both the majority class (Class 0) and the minority class (Class 1).
For Class 0 (Negative class), the model demonstrates high precision, recall, and F1-score, indicating its effectiveness in identifying negative instances.
For Class 1 (Positive class), the precision is moderate (62%), suggesting that among instances predicted as positive, 62% were actually positive. The recall is high (94%), implying that the model correctly identified 94% of the actual positive instances.
The overall accuracy is high due to good performance in predicting both classes, especially considering the higher precision and recall values for the minority class compared to other models.
The Random Forest model also exhibits promising results similar to the Logistic Regression model in handling the imbalance and performing well in identifying both positive and negative instances. However, further analysis and considerations should be made based on the specific requirements of the problem domain.



Logistic Regression Interpretation:
The model performs well in predicting both the majority class (Class 0) and the minority class (Class 1).
For Class 0 (Negative class), the model shows high precision, recall, and F1-score, indicating its effectiveness in identifying negative instances.
For Class 1 (Positive class), the precision is moderate (62%), indicating that among instances predicted as positive, 62% were actually positive. The recall is high (94%), suggesting that the model correctly identified 94% of the actual positive instances.
The overall accuracy is high due to good performance in predicting both classes, especially considering the higher precision and recall values for the minority class compared to other models.
This model shows promising results in handling the imbalance and performing well in identifying both positive and negative instances. However, further evaluation and considerations should be made based on the specific context and requirements of the problem domain.



SVM Interpretation:
The model performs very well in predicting the majority class (Class 0) with high precision, recall, and F1-score, indicating that it effectively identifies negative instances.
However, the model performs extremely poorly in predicting the minority class (Class 1), showing 0% precision, recall, and F1-score. This means it fails to identify any actual positive instances and misclassifies all positive instances as negative.
The overall test accuracy is considerably high due to the imbalance in the dataset (the majority class dominates the predictions), but this high accuracy is deceptive because the model fails to capture the minority class.
In cases of severe class imbalance, like in this scenario, where Class 1 has very few samples, it's crucial to address class imbalance by using techniques such as oversampling, undersampling, or employing different evaluation metrics like AUC-ROC, Precision-Recall curve, or focusing on sensitivity/recall for the minority class to better assess model performance.

### Feature Importance Visualization
This plot helps visualize which features are most important in predicting systemic crises.
"""

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

"""### Model Comparison Visualization
This bar chart compares the accuracy of different models.
"""

import matplotlib.pyplot as plt

model_names = ['Random Forest', 'Logistic Regression', 'SVM']
accuracies = [0.9634, 0.9487, 0.9345]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Comparison: Accuracy Scores')
plt.ylabel('Accuracy')
plt.show()

"""### Confusion Matrix Visualization
To visualize model performance, confusion matrices can be plotted for each model.
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix - Random Forest')
plt.show()

"""### Conclusion
The RandomForest model demonstrates strong predictive performance with a 96.34% accuracy. Future enhancements could include more advanced techniques for handling class imbalance and exploring neural network models to improve predictions. This project has real-world implications for policy-making, risk assessment, and economic stability in African countries.
"""