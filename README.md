# 🌟 Welcome to My GitHub Profile! 🌟

![Header](https://img.shields.io/badge/-Hello,%20I'm%20Arif%20Mia-blueviolet?style=for-the-badge)

Hi there! 👋 I'm **Arif Mia**, a passionate **Machine Learning Engineer** who loves to solve real-world problems using data and algorithms. I also enjoy coding, analyzing datasets, and creating amazing projects.

---

## 💻 **About Me**

- 🎓 Computer Science and Engineering Student
- 🌟 Expert in Machine Learning, Deep Learning, and Computer Vision
- 🔬 Passionate about Data Science and AI Research
- 🌎 Working remotely for an international company
- 💼 Kaggle Notebooks Expert | LinkedIn Services: Machine Learning Engineering
- 🛠️ Tools: Python, Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, and more!

---

## 🛠 **Skills & Tools**
- 🚀 **Programming Languages**: Python, SQL, R
- 📊 **Data Analysis & Visualization**: Matplotlib, Seaborn, Power BI
- 🤖 **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- 🧠 **Deep Learning**: TensorFlow, Keras, PyTorch
- 🌐 **Web Development**: HTML, CSS, JavaScript

---

## 📈 **Stats & Contributions**

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=Arif-miad&show_icons=true&theme=radical)

---

## 🌐 **Find Me Online**

- **📧 Email**: [arifmiahcse@gmail.com](mailto:arifmiahcse@gmail.com)
- **📂 Kaggle**: [Kaggle Profile](https://www.kaggle.com/code/arifmia/comprehensive-machine-learning-workflow-for-predic)
- **💼 LinkedIn**: [GitHub Profile](https://github.com/Arif-miad)

Let's connect and collaborate! 🚀

---


# Room Occupancy Prediction using Environmental Sensor Data

This repository contains a comprehensive machine learning workflow for predicting room occupancy based on environmental sensor data, including features such as temperature, humidity, light intensity, CO2 levels, and humidity ratio. The dataset includes timestamped readings that allow for an accurate classification of room occupancy (Occupied/Not Occupied).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
- [Performance Evaluation](#performance-evaluation)
- [Results](#results)
- [License](#license)

## Overview

This project aims to predict room occupancy using sensor data. The dataset includes environmental readings collected at regular intervals with a ground-truth label indicating whether the room is occupied or not.

- **Features**: Temperature, Humidity, Light, CO2, Humidity Ratio
- **Target**: Occupancy (0: Not Occupied, 1: Occupied)

The project involves the following steps:

1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Model Training with multiple classification algorithms
4. Performance Evaluation (Confusion Matrix, Accuracy, F1-Score)

## Dataset

The dataset contains the following columns:

- **id**: Unique identifier for each entry
- **date**: Timestamp of the reading
- **Temperature**: Temperature value in the room
- **Humidity**: Humidity percentage in the room
- **Light**: Light intensity in the room
- **CO2**: CO2 concentration in the room
- **HumidityRatio**: Ratio of humidity to temperature
- **Occupancy**: Ground truth label indicating whether the room is occupied (1) or not (0)

You can download the dataset from [here](#).

## Prerequisites

To run the project, you need to have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `tensorflow` or `keras`

Install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm tensorflow
```

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/room-occupancy-prediction.git
   ```

2. Change directory to the project folder:

   ```bash
   cd room-occupancy-prediction
   ```

3. Install the necessary libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook or script to start exploring and analyzing the data.

## Data Preprocessing

- **Handling Missing Values**: We identify and handle missing data points (if any).
- **Feature Engineering**: Extracted features such as year, month, day, and hour from the `date` column.
- **Scaling**: Applied Min-Max scaling or StandardScaler to normalize continuous features (e.g., Temperature, Humidity, CO2).

## Exploratory Data Analysis (EDA)

During the EDA phase, we performed various visualizations to understand the data distribution and relationships between features:

- **Histograms** for distribution of individual features
- **Boxplots** to identify outliers
- **Pairplots** to visualize correlations between features
- **Heatmap** to inspect correlation between variables
- **KDE plots** for distribution estimation
- **Pie chart** for occupancy distribution

## Machine Learning Models

We implemented and compared the performance of multiple classification algorithms to predict room occupancy:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Classifier (SVC)**
4. **K-Nearest Neighbors (KNN)**
5. **Gradient Boosting Classifier**
6. **XGBoost**
7. **LightGBM**
8. **Decision Tree Classifier**
9. **Naive Bayes Classifier**
10. **Neural Network (MLP)**

Each model is evaluated using cross-validation, and hyperparameters are tuned to maximize accuracy.

## Performance Evaluation

After training the models, we evaluated their performance using the following metrics:

- **Confusion Matrix**: To visualize true positives, false positives, true negatives, and false negatives
- **Accuracy**: The ratio of correct predictions
- **Precision, Recall, and F1-Score**: For a better understanding of model performance in imbalanced datasets
- **ROC Curve & AUC**: To assess classification performance across all thresholds

Example of a confusion matrix plot:

```python
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Step 3: Handle missing values (if any)
df.fillna(df.median(), inplace=True)

# Step 4: Encode categorical features (if any)
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Define feature matrix (X) and target variable (y)
X = df.drop('Occupancy', axis=1)
y = df['Occupancy']

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 7: Standardize the feature matrix
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Step 9: Train and evaluate each model
results = []
for model_name, model in models.items():
    # Step 10: Fit the model
    model.fit(X_train, y_train)
    
    # Step 11: Make predictions
    y_pred = model.predict(X_test)
    
    # Step 12: Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results.append({"Model": model_name, "Accuracy": accuracy})
    
    # Step 13: Print classification report
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Step 14: Create a DataFrame of results
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

# Step 15: Plot model accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.show()

# Step 16: Hyperparameter tuning for the best model (e.g., Random Forest)
best_model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Step 17: Display best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Score:", grid_search.best_score_)

# Step 18: Retrain the best model with optimal parameters
optimized_model = RandomForestClassifier(**grid_search.best_params_)
optimized_model.fit(X_train, y_train)

# Step 19: Evaluate the optimized model
y_pred_optimized = optimized_model.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print("Optimized Model Accuracy:", accuracy_optimized)

# Step 20: Visualize feature importance (for Random Forest)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': optimized_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="viridis")
plt.title("Feature Importance")
plt.show()

# Step 21: Cross-validation scores for all models
cv_results = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_results[model_name] = scores.mean()

# Step 22: Display cross-validation results
cv_results_df = pd.DataFrame(list(cv_results.items()), columns=['Model', 'CV Accuracy']).sort_values(by='CV Accuracy', ascending=False)

# Step 23: Plot cross-validation results
plt.figure(figsize=(10, 6))
sns.barplot(x="CV Accuracy", y="Model", data=cv_results_df, palette="viridis")
plt.title("Cross-Validation Accuracy Comparison")
plt.show()

# Step 24: Save the best-performing model
import joblib
joblib.dump(optimized_model, "optimized_model.pkl")

# Step 25: Load and test the saved model
loaded_model = joblib.load("optimized_model.pkl")
test_prediction = loaded_model.predict(X_test[:5])
print("Test Predictions from Loaded Model:", test_prediction)

# Step 26: Precision-Recall Curve for Logistic Regression
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, models["Logistic Regression"].predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve (Logistic Regression)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Step 27: ROC Curve and AUC for Random Forest
from sklearn.metrics import roc_curve, roc_auc_score
rf_prob = models["Random Forest"].predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, rf_prob)
auc_score = roc_auc_score(y_test, rf_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.title("ROC Curve (Random Forest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Step 28: Check for overfitting (train vs test accuracy for Random Forest)
train_accuracy = optimized_model.score(X_train, y_train)
test_accuracy = optimized_model.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

# Step 29: Test other metrics for Random Forest
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_optimized))
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_optimized))

# Step 30: Summary of model performances
print("Summary of Model Accuracies:\n", results_df)
print("Summary of Cross-Validation Scores:\n", cv_results_df)

```

## Results

The model with the best performance is chosen based on the evaluation metrics. For instance, **Random Forest Classifier** might perform best with a high accuracy and F1-score, ensuring a good balance between precision and recall.



