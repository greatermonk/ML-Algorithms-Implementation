"""
*Dataset Link:- https://www.kaggle.com/datasets/shahriarkabir/procurement-strategy-dataset-for-kraljic-matrix
*Description:
This dataset simulates procurement and supply chain data for strategic decision-making using the Kraljic Matrix,
a framework to classify products/services based on supply risk and profit impact.
It includes realistic biases and patterns to mimic real-world procurement scenarios,
such as supplier dependencies, geopolitical risks, and sustainability challenges.

*Dataset Details
i) Instances: 1,000 synthetic procurement items.
ii) Features: 11 columns (mix of categorical, numerical, and ordinal data).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.family'] = 'serif'

# Define color palettes
palette_heatmap = 'Blues' # For confusion matrix
# Custom colormap for decision boundary plots (adjust colors as needed)
cmap_decision_boundary = ListedColormap(['#FF5733', '#33FF57', '#3357FF', '#F1C40F']) # Example: Orange, Green, Blue, Yellow
cmap_scatter = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12'] # Matching colors for scatter points


# --- Configuration ---
# Define the target column based on dataset inspection
target_column = 'Kraljic_Category'
# Define columns to drop (e.g., identifiers or high cardinality categorical not suitable for OHE)
# Review dataset: 'Product_Name', 'Supplier_Region' likely identifiers. Let's drop them.
columns_to_drop = ['Product_Name', 'Supplier_Region']
# Define key numerical features for 2D visualization (based on Kraljic concept)
# Check dataset: 'Financial Impact' and 'Supply Risk' seem appropriate.
viz_feature_1 = 'Profit_Impact_Score'
viz_feature_2 = 'Supply_Risk_Score'

file_path = "archive/realistic_kraljic_dataset.csv"
# --- Load Data ---
print("--- Loading Data ---")
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from '{file_path}'.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please download the dataset from 'https://www.kaggle.com/datasets/shahriarkabir/procurement-strategy-dataset-for-kraljic-matrix' and place it in the correct directory.")
    exit()

# Display basic info
print("\nBasic Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum()) # Check for missing values


# --- i) Data Pre-Processing ---
print("\n--- i) Data Pre-Processing ---")

# Drop specified columns
print(f"\nDropping columns: {columns_to_drop}")
df_processed = df.drop(columns=columns_to_drop)

# Identify feature types
categorical_cols = df_processed.select_dtypes(include='object').drop(target_column, axis=1).columns
numerical_cols = df_processed.select_dtypes(include=np.number).columns

print(f"\nIdentified Categorical Features (excluding target): {list(categorical_cols)}")
print(f"Identified Numerical Features: {list(numerical_cols)}")

# Apply One-Hot Encoding to categorical features
if not categorical_cols.empty:
    print("\nApplying One-Hot Encoding...")
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    print("One-Hot Encoding complete.")
    print(f"Shape after encoding: {df_processed.shape}")
    print("Columns after encoding:", df_processed.columns)
else:
    print("\nNo categorical features found for One-Hot Encoding.")

# Encode the target variable
print(f"\nEncoding target variable: '{target_column}'")
le = LabelEncoder()
# Ensure the target column exists before encoding
if target_column not in df_processed.columns:
     print(f"Error: Target column '{target_column}' not found after potential preprocessing steps.")
     exit()

df_processed[target_column] = le.fit_transform(df_processed[target_column])
print("Target variable encoding complete.")
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
target_classes = le.classes_ # Store class names for later use

# Define Features (X) and Target (y)
X = df_processed.drop(target_column, axis=1)
y = df_processed[target_column] # Already encoded target

# Split data into Training and Testing sets
print("\nSplitting data into Training (80%) and Testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data Splitting complete.")
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Note: Feature Scaling (e.g., StandardScaler) is generally not required for Random Forests.
# --- ii) Fitting the Random Forest Algorithm to the training set ---
print("\n--- ii) Fitting Random Forest ---")

# Initialize the Random Forest Classifier
# n_estimators=100 is a common default, adjust as needed
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Added class_weight='balanced' for potentially imbalanced classes

# Train the model
print("Training the Random Forest model on the full feature set...")
rf_classifier.fit(X_train, y_train)
print("Model training complete.")


# --- iii) Predicting testing set results ---
print("\n--- iii) Predicting Test Set Results ---")

y_pred = rf_classifier.predict(X_test)
print("Predictions generated for the test set.")


# --- iv) Creating & Visualizing Confusion matrix ---
print("\n--- iv) Confusion Matrix and Metrics ---")

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")

# Generate Classification Report
print("\nClassification Report:")
# Use target names from the LabelEncoder
report = classification_report(y_test, y_pred, target_names=target_classes)
print(report)

# Generate and Display Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8)) # Adjusted size for potentially 4x4 matrix
sns.heatmap(cm, annot=True, fmt='d', cmap=palette_heatmap,
            xticklabels=target_classes, yticklabels=target_classes,
            annot_kws={"size": 14, "color": 'black'} # Assuming 'Blues' cmap makes black readable
           )
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# --- v) Visualizing training set and testing set results (2D Approximation) ---
print(f"\n--- v) Visualizing Decision Boundaries (2D Approximation using '{viz_feature_1}' and '{viz_feature_2}') ---")
print("NOTE: This visualization uses only two features and may not fully represent the model's behavior with all features.")

# Check if the chosen visualization features exist
if viz_feature_1 not in X.columns or viz_feature_2 not in X.columns:
    print(f"Error: One or both visualization features ('{viz_feature_1}', '{viz_feature_2}') not found in the processed data.")
    print("Skipping 2D visualization.")
else:
    # Select only the two features for visualization
    X_train_viz = X_train[[viz_feature_1, viz_feature_2]]
    X_test_viz = X_test[[viz_feature_1, viz_feature_2]]

    # Train a *new* RF model using only these two features
    print(f"Training a separate Random Forest model using only '{viz_feature_1}' and '{viz_feature_2}' for visualization...")
    rf_viz = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_viz.fit(X_train_viz, y_train) # y_train is already encoded
    print("Visualization model training complete.")

    # Function to plot decision boundaries
    def plot_decision_boundary(model, X_set, y_set, title, feature1_name, feature2_name):
        plt.figure(figsize=(10, 6))
        X1, X2 = X_set.iloc[:, 0], X_set.iloc[:, 1] # Use iloc for position-based selection
        y_true = y_set

        # Create mesh grid
        x_min, x_max = X1.min() - 1, X1.max() + 1
        y_min, y_max = X2.min() - 1, X2.max() + 1
        # Increase density for smoother contours if needed (e.g., step=0.01)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        # Predict on mesh grid
        mesh_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[feature1_name, feature2_name])
        Z = model.predict(mesh_data)
        Z = Z.reshape(xx.shape)

        # Plot decision regions
        plt.contourf(xx, yy, Z, alpha=0.6, cmap=cmap_decision_boundary)

        # Scatter plot actual points
        for i, j in enumerate(np.unique(y_true)):
             plt.scatter(X1[y_true == j], X2[y_true == j],
                         color=cmap_scatter[i % len(cmap_scatter)], label=target_classes[j], # Use modulo for safety
                         edgecolor='w', s=40, alpha=0.9) # Add white edge color

        plt.title(title, fontsize=16)
        plt.xlabel(feature1_name, fontsize=14)
        plt.ylabel(feature2_name, fontsize=14)
        plt.legend(loc='best')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.grid(False) # Turn off grid for contour plot clarity
        plt.tight_layout()
        plt.show()

    # Plot for Training Set
    plot_decision_boundary(rf_viz, X_train_viz, y_train,
                           f'Random Forest Decision Boundary (Training Set - 2D Approx.)',
                           viz_feature_1, viz_feature_2)

    # Plot for Test Set
    plot_decision_boundary(rf_viz, X_test_viz, y_test,
                           f'Random Forest Decision Boundary (Test Set - 2D Approx.)',
                           viz_feature_1, viz_feature_2)

print("\n--- Analysis Complete ---")