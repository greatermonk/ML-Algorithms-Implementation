"""
*Dataset Link:- https://www.kaggle.com/datasets/deepu1109/star-dataset
Dataset Info:
This is a dataset consisting of several features of stars.
*|----------------------------------------------------------|
*| i)Absolute Temperature (in K)                            |
*| ii)Relative Luminosity (L/Lo)                            |
*| iii)Relative Radius (R/Ro)                               |
*| iv)Absolute Magnitude (Mv)                               |
*| v)Star Color (white,Red,Blue,Yellow,yellow-orange etc.)  |
*| vi)Spectral Class (O,B,A,F,G,K,M)                        |
*| vii)Star Type (0: Red Dwarf, 1: Brown Dwarf,             |
*|     2: White Dwarf, 3: Main Sequence, 4: SuperGiants     |
*|     5: HyperGiants)                                      |
*|----------------------------------------------------------|
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. Load Data ---
dataset_path = "archive/stars.csv"
try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: path:{dataset_path} not found.")
    print("Please download the dataset from https://www.kaggle.com/datasets/deepu1109/star-dataset and place it in the correct directory.")
    exit()

# --- 2. Preprocessing (Minimal for Viz) ---
# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[()]', '', regex=True)

# Star Type Mapping
star_type_mapping = {
    0: 'Brown Dwarf', 1: 'Red Dwarf', 2: 'White Dwarf',
    3: 'Main Sequence', 4: 'Supergiant', 5: 'Hypergiant'
}
df['Star_type_label'] = df['Star_type'].map(star_type_mapping)

# Clean Star Color
df['Star_color'] = df['Star_color'].str.strip().str.lower().str.replace('-', ' ')
# Consolidate similar colors (example)
color_replacements = {
    'blue white': 'blue-white',
    'yellow white': 'yellow-white',
    'whitish': 'white',
    'pale yellow orange': 'orange-yellow' # Or just 'orange'/'yellow'? Depends on desired granularity
}
df['Star_color'] = df['Star_color'].replace(color_replacements)


# Log Transforms for skewed data (useful for some plots)
df['Log_Luminosity'] = np.log10(df['LuminosityL/Lo'])
df['Log_Radius'] = np.log10(df['RadiusR/Ro'])

# --- 3. Aesthetic Visualizations Setup ---

sns.set_theme(style="darkgrid", context="talk") # "talk" increases font sizes slightly
custom_palette = sns.color_palette("viridis", n_colors=6) # Using viridis

# a) Distribution of Star Types
plt.figure(figsize=(12, 7), facecolor="white")
ax = sns.countplot(data=df, x='Star_type_label', order=star_type_mapping.values(), palette=custom_palette, saturation=0.8)
plt.title('Distribution of Star Types in the Dataset', fontsize=18, weight='bold', pad=20)
plt.xlabel('Star Type', fontsize=14, labelpad=15)
plt.ylabel('Number of Stars', fontsize=14, labelpad=15)
plt.xticks(rotation=30, ha='right', fontsize=12)
plt.yticks(fontsize=12)
# Add count labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fontsize=11.5, padding=3)
sns.despine() # Remove top and right spines
plt.tight_layout()
plt.show()

# b) Distribution of Numerical Features (using KDE plots for smooth shape)
numerical_features = ['Temperature_K', 'LuminosityL/Lo', 'RadiusR/Ro', 'Absolute_magnitudeMv']
log_numerical_features = ['Temperature_K', 'Log_Luminosity', 'Log_Radius', 'Absolute_magnitudeMv'] # Using log versions

print(f"\n--- Visualizing Numerical Feature Distributions ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()
plot_features = log_numerical_features # Choose original or log-transformed
feature_titles = ['Temperature (K)', 'Log10 Luminosity (L/Lo)', 'Log10 Radius (R/Ro)', 'Absolute Magnitude (Mv)']

for i, (col, title) in enumerate(zip(plot_features, feature_titles)):
    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette("magma", 4)[i], bins=20)
    axes[i].set_title(f'Distribution of {title}', fontsize=16, pad=15)
    axes[i].set_xlabel(title, fontsize=12, labelpad=10)
    axes[i].set_ylabel('Frequency', fontsize=12, labelpad=10)
    axes[i].tick_params(axis='both', which='major', labelsize=11)
plt.suptitle('Distributions of Key Numerical Features', y=1.03, fontsize=20, weight='bold')
plt.tight_layout()
plt.show()


# c) Distribution of Categorical Features (Horizontal for Star Color)
print("\n--- Visualizing Categorical Feature Distributions ---")
fig, axes = plt.subplots(2, 1, figsize=(12, 14)) # Adjusted figure size for horizontal plot

# Star Color (Horizontal)
color_order = df['Star_color'].value_counts().index
sns.countplot(data=df, y='Star_color', order=color_order, palette='Spectral', ax=axes[0], saturation=0.85)
axes[0].set_title('Distribution of Star Colors (Cleaned)', fontsize=16, pad=15)
axes[0].set_xlabel('Number of Stars', fontsize=12, labelpad=10)
axes[0].set_ylabel('Star Color', fontsize=12, labelpad=10)
axes[0].tick_params(axis='both', which='major', labelsize=11)

# Spectral Class (Vertical)
class_order = df['Spectral_Class'].value_counts().index
sns.countplot(data=df, x='Spectral_Class', order=class_order, palette='coolwarm', ax=axes[1], saturation=0.85)
axes[1].set_title('Distribution of Spectral Classes', fontsize=16, pad=15)
axes[1].set_xlabel('Spectral Class', fontsize=12, labelpad=10)
axes[1].set_ylabel('Number of Stars', fontsize=12, labelpad=10)
axes[1].tick_params(axis='both', which='major', labelsize=11)

plt.suptitle('Distributions of Categorical Features', y=1.01, fontsize=20, weight='bold')
plt.tight_layout(h_pad=4) # Add padding between subplots
plt.show()


# d) Numerical Features vs. Star Type (Boxplots or Violinplots)
print("\n--- Visualizing Numerical Features across Star Types ---")
fig2, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.ravel()
star_type_order = list(star_type_mapping.values()) # Ensure consistent order

for i, (col, title) in enumerate(zip(plot_features, feature_titles)):
    # Use boxplot for clear quantile info, could use violinplot for shape
    sns.boxplot(data=df, x='Star_type_label', y=col, order=star_type_order, palette=custom_palette, ax=axes[i])
    sns.violinplot(data=df, x='Star_type_label', y=col, order=star_type_order, palette=custom_palette, ax=axes[i], inner='quartile') # Alternative
    axes[i].set_title(f'{title} by Star Type', fontsize=16, pad=15)
    axes[i].set_xlabel('Star Type', fontsize=12, labelpad=10)
    axes[i].set_ylabel(title, fontsize=12, labelpad=10)
    axes[i].tick_params(axis='x', rotation=30, labelsize=11)
    axes[i].tick_params(axis='y', labelsize=11)

plt.suptitle('Numerical Feature Distributions Across Star Types', y=1.02, fontsize=20, weight='bold')
plt.tight_layout()
plt.show()


# e) Enhanced Pairplot (Focus on key features)
pairplot_features = ['Temperature_K', 'Log_Luminosity', 'Log_Radius', 'Absolute_magnitudeMv', 'Star_type_label']
print("\n--- Generating Enhanced Pairplot (takes a moment) ---")
g = sns.pairplot(df[pairplot_features], hue='Star_type_label',
                 palette=custom_palette, diag_kind='kde', # Use KDE for diagonal
                 plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k', 'linewidth': 0.5}, # Style scatter points
                 diag_kws={'fill': True}) # Fill KDE plots
g.fig.suptitle('Pairwise Relationships Between Key Features (Colored by Star Type)', y=1.03, fontsize=20, weight='bold')
# Improve legend location
# g._legend.set_bbox_to_anchor((1.05, 0.5)) # May need adjustment depending on window size
plt.show()


# f) Enhanced Correlation Heatmap
plt.figure(figsize=(10, 8), facecolor='white')
# Calculate correlation on original numerical features + target code
correlation_matrix = df[numerical_features + ['Star_type']].corr()
mask = np.triu(correlation_matrix) # Mask for upper triangle
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, linecolor='black', mask=mask, # Add lines and mask
            annot_kws={"size": 12}) # Adjust annotation font size
plt.title('Correlation Matrix of Numerical Features & Star Type', fontsize=18, weight='bold', pad=20)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# Define features (X) and target (y)
X = df.drop(['Star_type', 'Star_type_label', 'Log_Luminosity', 'Log_Radius'], axis=1)
y = df['Star_type']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Create preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build and Train Naive Bayes Model Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])
model_pipeline.fit(X_train, y_train)

# Make Predictions
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
target_names = [star_type_mapping[i] for i in sorted(y.unique())]
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

plt.figure(figsize=(10, 8), facecolor='white')
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', # Blues is a nice sequential palette for CMs
            linewidths=0.5, linecolor='gray', annot_kws={"size": 14}) # Larger annotations
plt.title('Confusion Matrix - Naive Bayes Classifier', fontsize=18, weight='bold', pad=20)
plt.xlabel('Predicted Star Type', fontsize=14, labelpad=15)
plt.ylabel('True Star Type', fontsize=14, labelpad=15)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0) # Keep y-axis labels horizontal
plt.tight_layout()
plt.show()
