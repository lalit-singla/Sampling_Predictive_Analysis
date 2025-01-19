# Sampling_Predictive_Analysis
This give a brief idea about sampling with different type of sampling.

# Steps

# Step 1: Import Libraries

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
```
# Step 2: Load the Dataset
```
data = pd.read_csv(r'C:\Users\91991\Downloads\Creditcard_data.csv')
```
# Step 3: Initial Exploration of the Dataset
```
data.head()
data.info()
data.describe()
```
data.head(): To display the first 5 rows of the dataset.

data.info(): Provides metadata like column names, data types, and non-null counts.

data.describe(): Provides descriptive analysis of the columns like mean,median etc .

# Step 4: Class Distribution Analysis
```
print("Dataset preview:")
print(data.head())
print("\nClass distribution before balancing:")
print(data['Class'].value_counts())
```
Count the number of instances having target variable (class) as class == 1 or class == 0.

# Step 5: Missing Values Check
```
missing_values = data.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)
```
To check whether there is any missing value or not in any of the columns.

# Step 6: Balance the Dataset using SMOTE
```
X = data.drop('Class', axis=1)
y = data['Class']
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("\nClass distribution after balancing:")
print(pd.Series(y_balanced).value_counts())
```
SMOTE (Synthetic Minority Oversampling Technique): Balances the dataset by creating synthetic samples for the minority class.

x: Independent features.

y: Target variable.

fit_resample: Applies SMOTE to create x_smote and y_smote, balanced datasets

# Step 7: Define sample size formula (example: 10% of the dataset)
```
SAMPLE_SIZE = int(0.1 * len(X_balanced))
```
# Step 8:  Create samples using different sampling techniques
```
samples = {}
np.random.seed(42)  # For reproducibility
```

# Step 9: Sampling Techniques
```
# Sampling Technique 1: Random Sampling
samples['Random'] = (
    X_balanced.sample(SAMPLE_SIZE, random_state=42),  # Ensure reproducibility
    y_balanced.loc[X_balanced.sample(SAMPLE_SIZE, random_state=42).index]
)

# Sampling Technique 2: Stratified Sampling
stratified_indices = (
    y_balanced
    .groupby(y_balanced)  # Group by unique values of y_balanced
    .apply(lambda x: x.sample(n=int(SAMPLE_SIZE / y_balanced.nunique()), random_state=42))
    .index.get_level_values(1)  # Extract the original indices after grouping
)
samples['Stratified'] = (
    X_balanced.loc[stratified_indices], 
    y_balanced.loc[stratified_indices]
)

# Sampling Technique 3: Systematic Sampling
step = max(1, len(X_balanced) // SAMPLE_SIZE)  # Avoid step = 0 if SAMPLE_SIZE > len(X_balanced)
systematic_indices = list(range(0, len(X_balanced), step))[:SAMPLE_SIZE]
samples['Systematic'] = (
    X_balanced.iloc[systematic_indices], 
    y_balanced.iloc[systematic_indices]
)

# Sampling Technique 4: Cluster Sampling
clusters = pd.qcut(X_balanced['Amount'], q=5, labels=False, duplicates='drop')  # Handle duplicate bins
cluster_0_indices = X_balanced.index[clusters == 0]  # Indices of cluster 0
clustered_indices = cluster_0_indices[:min(SAMPLE_SIZE, len(cluster_0_indices))]  # Adjust if cluster size is smaller
samples['Cluster'] = (
    X_balanced.loc[clustered_indices], 
    y_balanced.loc[clustered_indices]
)

# Sampling Technique 5: Oversampling a specific class
oversample_class_indices = y_balanced[y_balanced == 1].sample(
    n=SAMPLE_SIZE, replace=True, random_state=42
).index  # Oversample with replacement
samples['Oversampled'] = (
    X_balanced.loc[oversample_class_indices], 
    y_balanced.loc[oversample_class_indices]
)
```
Multiple sampling techniques are applied to the balanced dataset to extract samples:
1. Simple Random Sampling: Selects a random subset of the data without replacement.
2. Stratified Sampling: Ensures that the class distribution is maintained in the sample.
3. Systematic Sampling: Selects samples based on a fixed interval k.
4. Cluster Sampling: Divides the dataset into clusters and selects one cluster.
5. Bootstrapping: Samples data points with replacement.

# Step 10: Train models and evaluate accuracy

 ```
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}
```

# Step 11: Evaluate Accuracy
```
results = pd.DataFrame(columns=['Model', 'Sampling Technique', 'Accuracy'])

for technique, (X_sample, y_sample) in samples.items():
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample)
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results = pd.concat([results, pd.DataFrame([[model_name, technique, acc]], columns=['Model', 'Sampling Technique', 'Accuracy'])], ignore_index=True)
            except ValueError as e:
                print(f"Model {model_name} failed for sampling technique {technique} due to: {e}")
            except Exception as e:
                print(f"Unexpected error with {model_name} and {technique}: {e}")
    except ValueError as e:
        print(f"Train-test split failed for sampling technique {technique} due to: {e}")
    except Exception as e:
        print(f"Unexpected error during sampling {technique}: {e}")

print("Accuracy Results:")
print(results)
results.to_csv('sampling_results.csv', index=False)
```
# Step 12: Results 
```
print("\nAccuracy Results:")
print(results)

results.to_csv('sampling_results.csv', index=False)

```
# Step 13: Identify best sampling technique for each model
```
best_results = results.loc[results.groupby('Model')['Accuracy'].idxmax()]
print("\nBest Sampling Technique for Each Model:")
print(best_results)
```

# Best Model for each sample 
# Sample 1: Gradient Boosting 
# Sample 2: Gradient Boosting 
# Sample 3: Logistic Regression
# Sample 4: Decision Tree / Gradient Boosting 
# Sample 5: Gradient Boosting (overfits) / both logistic Regression and Decision Tree
