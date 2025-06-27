"""
*Dataset Link:- https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data
Dataset Info:
This is a dataset consisting of several features of patient,s cardiovascular system.
*|--------------------------------------------------------------------------------------------------------------------------------|
*|1) Age: age of the patient [years]                                                                                              |
*|2) Sex: sex of the patient [M: Male, F: Female]                                                                                 |
*|3) ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]          |
*|4) RestingBP: resting blood pressure [mm Hg]                                                                                    |
*|5)Cholesterol: serum cholesterol [mm/dl]                                                                                        |
*|6) FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]                                                   |
*|7) RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and\or    |   STelevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]   |
*|8) MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]                                                        |
*|9) ExerciseAngina: exercise-induced angina [Y: Yes, N: No]                                                                      |
*|10) Oldpeak: oldpeak = ST [Numeric value measured in depression]                                                                |
*|11) ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]                          |
*|12HeartDisease: output class [1: heart disease, 0: Normal]                                                                      |
*|--------------------------------------------------------------------------------------------------------------------------------|
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Style Configuration ---
plt.style.use('dark_background') # Apply dark background style
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size
plt.rcParams['axes.titlesize'] = 16 # Title font size
plt.rcParams['axes.labelsize'] = 12 # Axis label font size
plt.rcParams['xtick.labelsize'] = 10 # X tick label size
plt.rcParams['ytick.labelsize'] = 10 # Y tick label size
plt.rcParams['legend.fontsize'] = 10 # Legend font size
plt.rcParams['font.family'] = 'sans-serif' # A commonly available font

# Define some color palettes that work well on dark background
palette_binary = 'coolwarm' # Good for binary distinctions (like target)
palette_hist = 'viridis'   # Good for histograms with hue
palette_count = 'magma'    # Good for countplots with hue
palette_sequential = 'plasma' # Good for feature importance
cmap_heatmap = 'Blues'     # Good for confusion matrix

# --- 1. Data Loading ---
print("--- 1. Loading Data ---")
try:
    file_path = 'archive/heart.csv'
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from '{file_path}'.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please download the dataset from 'https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data' and place it in the correct directory.")
    exit()

# --- 2. Data Analysis (Exploratory Data Analysis - EDA) ---
print("\n--- 2. Data Analysis (EDA) ---")

# Display basic information (No visualization change needed here)
print("\nBasic Info:")
df.info()
print("\nFirst 5 Rows:")
print(df.head())
print("\nDescriptive Statistics (Numerical Features):")
print(df.describe())
print("\nDescriptive Statistics (Categorical Features):")
print(df.describe(include='object'))
print("\nMissing Values:")
print(df.isnull().sum())

# Analyze target variable distribution
print("\nTarget Variable Distribution (HeartDisease):")
plt.figure(figsize=(8, 5))
sns.countplot(x='HeartDisease', data=df, palette=palette_binary)
plt.title('Distribution of Heart Disease (0: No, 1: Yes)', fontsize=14)
plt.xlabel('Heart Disease Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['No Heart Disease (0)', 'Heart Disease (1)']) # Clearer labels
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
print(df['HeartDisease'].value_counts(normalize=True)) # Show percentages

numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
print(f"\nAnalyzing distributions of: {', '.join(numerical_features)}")
for col in numerical_features:
    plt.figure(figsize=(10, 6)) # Use default figure size
    sns.histplot(data=df, x=col, hue='HeartDisease', kde=True, palette=palette_hist, alpha=0.7)
    plt.title(f'Distribution of {col} by Heart Disease Status', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Heart Disease', labels=['Yes (1)', 'No (0)']) # Clarify legend
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Analyze distributions of categorical features by target variable
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
print(f"\nAnalyzing distributions of: {', '.join(categorical_features)}")
for col in categorical_features:
    plt.figure(figsize=(10, 6)) # Use default figure size
    sns.countplot(data=df, x=col, hue='HeartDisease', palette=palette_count, alpha=0.8)
    plt.title(f'Distribution of {col} by Heart Disease Status', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=30, ha='right') # Rotate labels slightly if needed
    plt.legend(title='Heart Disease', labels=['No (0)', 'Yes (1)']) # Clarify legend
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- 3. Data Pre-processing ---
print("\n--- 3. Data Pre-processing ---")
categorical_cols = df.select_dtypes(include='object').columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).drop('HeartDisease', axis=1).columns

print(f"\nIdentified Categorical Columns: {list(categorical_cols)}")
print(f"Identified Numerical Columns: {list(numerical_cols)}")

df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nData after One-Hot Encoding (first 5 rows):")
print(df_processed.head())
print(f"Shape after encoding: {df_processed.shape}")

X = df_processed.drop('HeartDisease', axis=1)
y = df_processed['HeartDisease']

print("\nFeatures (X) columns:")
print(X.columns)
print(f"Shape of X: {X.shape}")
print("\nTarget (y) values sample:")
print(y.head())
print(f"Shape of y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nData Splitting:")
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
print(f"Training set target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Testing set target distribution:\n{y_test.value_counts(normalize=True)}")

# Optional Scaling (remains commented out)
# scaler = StandardScaler()
# X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# --- 4. Model Training (Decision Tree Classifier) ---
print("\n--- 4. Model Training ---")
dt_classifier = DecisionTreeClassifier(random_state=42)
print("Training the Decision Tree model...")
dt_classifier.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Model Evaluation ---
print("\n--- 5. Model Evaluation ---")
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=['No Heart Disease (0)', 'Heart Disease (1)'])
print(report)

# Generate and Display  Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_heatmap, # Use direct seaborn heatmap
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease'],
            annot_kws={"size": 14, "color": 'black' if cmap_heatmap in ['Blues', 'viridis'] else 'white'} # Adjust text color based on cmap
           )
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

# --- 6. Feature Importance ---
print("\n--- 6. Feature Importance ---")
importances = dt_classifier.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance's according to the Decision Tree:")
print(feature_importance_df)

# Visualize Feature Importance's (Top 15)
plt.figure(figsize=(12, 8)) # Larger figure for better label spacing
top_n = 15
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette=palette_sequential)
plt.title(f'Top {top_n} Feature Importances', fontsize=15)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6) # Add vertical grid lines
plt.tight_layout()
plt.show()

# --- 7. Data Visualization (Decision Tree Structure) ---
print("\n--- 7. Data Visualization ---")
print("Visualizing the Decision Tree structure (limited depth for clarity)...")

# Increase figure size significantly for the tree plot
plt.figure(figsize=(28, 16))

plot_tree(
    dt_classifier,
    filled=True,
    rounded=True,
    feature_names=X.columns.tolist(),
    class_names=['No HD', 'HD'],
    max_depth=3,
    fontsize=12
)
# Add a title using plt.title after plot_tree
plt.title("Decision Tree Structure (Max Depth = 3)", fontsize=20, y=1.02) # Adjust y position if needed
plt.show()

print("\n--- Analysis Complete ---")