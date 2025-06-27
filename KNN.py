"""
*Dataset Link:- https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023/data
*Description:
Salaries of Different Data Science Fields in the Data Science Domain

*Dataset Details
Dataset Columns:

*i) work_year: The year the salary was paid.
*ii) experience_level: The experience level in the job during the year
*iii) employment_type: The type of employment for the role
*iv) job_title: The role worked in during the year.
*v)salary: The total gross salary amount paid.
*6) salary_currency: The currency of the salary paid as an ISO 4217 currency code.
*7) salaryinusd: The salary in USD
*8) employee_residence: Employee's primary country of residence in during the work year as an ISO 3166 country code.
*9) remote_ratio: The overall amount of work done remotely
*10) company_location: The country of the employer's main office or contracting branch
*11) company_size: The median number of people that worked for the company during the year
"""

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore potential convergence warnings and future warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# %% Plotting Style Configuration
# Use a dark background style
plt.style.use('dark_background')
# Adjust default figure size and font size
plt.rcParams['figure.figsize'] = (11, 7) # Slightly larger default size
plt.rcParams['font.size'] = 13
# Define colors for consistency and contrast
text_color = 'white'
bar_color = 'cyan'       # Bright color for bars
grid_color = '#555555'    # Muted gray for grid lines
palette_sequential = 'viridis' # Good sequential palette (used for heatmap)
palette_categorical = 'plasma' # Good categorical palette (used for hue/boxplots)

# %% --- 1. Load Data ---
dataset_path = "archive/ds_salaries.csv"
try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
    # Keep a copy for EDA before dropping salary_in_usd
    df_original_eda = df.copy()
    # Drop columns
    columns_to_drop = ['salary', 'salary_currency', 'work_year']
    df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
    print(f"Dropped columns: {columns_to_drop}")
    print(f"Remaining columns for modeling: {df.columns.tolist()}")

except FileNotFoundError:
    print("Error: 'ds_salaries.csv' not found.")
    print(f"Please ensure the dataset file is in the correct directory ({dataset_path}) or provide the full path.")

# %% --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- Starting Exploratory Data Analysis (EDA) ---")

# --- Histogram of Salary (USD) ---
plt.figure()
sns.histplot(df_original_eda['salary_in_usd'], kde=True, color=bar_color, bins=30) # Added bins
plt.title('Distribution of Salaries (USD)', color=text_color, fontsize=15)
plt.xlabel('Salary in USD', color=text_color)
plt.ylabel('Frequency', color=text_color)
plt.xticks(color=text_color)
plt.yticks(color=text_color)
plt.grid(True, color=grid_color, linestyle='--', linewidth=0.5) # Add grid
plt.tight_layout()
plt.show()

# --- Countplots for Key Categorical Features ---
categorical_eda_cols = ['experience_level', 'employment_type', 'company_size', 'remote_ratio']
for col in categorical_eda_cols:
    plt.figure()
    order = df_original_eda[col].value_counts().index
    ax = sns.countplot(data=df_original_eda, x=col, order=order, color=bar_color)
    plt.title(f'Count of Records by {col.replace("_", " ").title()}', color=text_color, fontsize=15)
    plt.xlabel(col.replace("_", " ").title(), color=text_color)
    plt.ylabel('Count', color=text_color)
    plt.xticks(rotation=45 if df_original_eda[col].nunique() > 5 else 0, color=text_color)
    plt.yticks(color=text_color)
    ax.spines['top'].set_color(grid_color) # Color axes spines
    ax.spines['right'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    # No grid or legend typically needed for countplots
    plt.tight_layout()
    plt.show()

# --- Box Plot: Salary vs Experience Level ---
plt.figure()
ax = sns.boxplot(data=df_original_eda, x='experience_level', y='salary_in_usd',
                 order=['EN', 'MI', 'SE', 'EX'], palette=palette_categorical)
plt.title('Salary Distribution by Experience Level', color=text_color, fontsize=15)
plt.xlabel('Experience Level', color=text_color)
plt.ylabel('Salary in USD', color=text_color)
plt.xticks(color=text_color)
plt.yticks(color=text_color)
ax.spines['top'].set_color(grid_color)
ax.spines['right'].set_color(grid_color)
ax.spines['left'].set_color(grid_color)
ax.spines['bottom'].set_color(grid_color)
plt.grid(True, axis='y', color=grid_color, linestyle='--', linewidth=0.5) # Horizontal grid lines
# No legend typically needed for boxplots comparing categories on axis
plt.tight_layout()
plt.show()

# --- Countplot for Top Job Titles (Before Grouping for Model) ---
plt.figure(figsize=(12, 8)) # Wider figure
top_jobs = df_original_eda['job_title'].value_counts().nlargest(15).index
ax = sns.countplot(data=df_original_eda[df_original_eda['job_title'].isin(top_jobs)],
                   y='job_title', order=top_jobs, color=bar_color)
plt.title('Count of Top 15 Job Titles', color=text_color, fontsize=15)
plt.xlabel('Count', color=text_color)
plt.ylabel('Job Title', color=text_color)
plt.xticks(color=text_color)
plt.yticks(color=text_color)
ax.spines['top'].set_color(grid_color)
ax.spines['right'].set_color(grid_color)
ax.spines['left'].set_color(grid_color)
ax.spines['bottom'].set_color(grid_color)
plt.grid(True, axis='x', color=grid_color, linestyle='--', linewidth=0.5) # Vertical grid lines
plt.tight_layout()
plt.show()

# %% --- 3. Define Target Variable & Initial Features ---
try:
    df['salary_bracket'] = pd.qcut(df['salary_in_usd'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    print("\nTarget variable 'salary_bracket' created:")
    print(df['salary_bracket'].value_counts())

    # --- EDA Plot for Target Variable ---
    plt.figure()
    order = df['salary_bracket'].value_counts().index
    ax = sns.countplot(data=df, x='salary_bracket', order=order, color=bar_color)
    plt.title('Distribution of Target Salary Brackets', color=text_color, fontsize=15)
    plt.xlabel('Salary Bracket', color=text_color)
    plt.ylabel('Count', color=text_color)
    plt.xticks(color=text_color)
    plt.yticks(color=text_color)
    ax.spines['top'].set_color(grid_color)
    ax.spines['right'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    plt.tight_layout()
    plt.show()

    # --- Box Plot: Salary vs Salary Bracket ---
    plt.figure()
    ax = sns.boxplot(data=df, x='salary_bracket', y='salary_in_usd', palette=palette_categorical, order=['Low', 'Medium', 'High', 'Very High'])
    plt.title('Salary Distribution within Target Brackets', color=text_color, fontsize=15)
    plt.xlabel('Salary Bracket', color=text_color)
    plt.ylabel('Salary in USD', color=text_color)
    plt.xticks(color=text_color)
    plt.yticks(color=text_color)
    ax.spines['top'].set_color(grid_color)
    ax.spines['right'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    plt.grid(True, axis='y', color=grid_color, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # --- Define Features (X) and Target (y) ---
    X = df.drop(['salary_in_usd', 'salary_bracket'], axis=1)
    y = df['salary_bracket']
    print("\nInitial Features (X) columns:", X.columns.tolist())

except KeyError as e:
    print(f"\nError preparing features/target: Missing column - {e}. Adjust column names if needed.")
    X, y = None, None
except ValueError as e:
    print(f"\nError creating salary brackets (likely too few unique values): {e}")
    if len(df) <= 10: # Simplified handling for dummy data
         le = LabelEncoder()
         # Ensure y is categorical for stratification
         if df['salary_in_usd'].nunique() > 1:
             y = pd.qcut(df['salary_in_usd'], q=min(2, df['salary_in_usd'].nunique()), labels=False, duplicates='drop')
             y = y.astype('category') # Make sure it's treated as categorical
         else: # Handle case with only one salary value
             y = pd.Series([0]*len(df)).astype('category')
         y.name = 'salary_bracket'
         X = df.drop(['salary_in_usd', 'salary_bracket'], axis=1, errors='ignore')
         print("\nUsing simplified target for dummy data. Stratification might be affected.")
    else:
        X, y = None, None

# %% --- 4. Pre-processing (Cardinality Reduction before Split) ---
if X is not None and y is not None and y.nunique() > 1: # Check if target has more than 1 class
    print("\nApplying pre-processing (cardinality reduction)...")
    high_cardinality_cols = ['job_title', 'employee_residence', 'company_location']
    categorical_features_initial = X.select_dtypes(exclude=np.number).columns.tolist()

    for col in high_cardinality_cols:
        if col in X.columns:
            top_categories = X[col].value_counts().nlargest(10).index
            X[col] = X[col].apply(lambda x: x if x in top_categories else 'Other')
            print(f"Reduced cardinality for '{col}'.")

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Numerical features for model: {numerical_features}")
    print(f"Categorical features for model: {categorical_features}")

    # %% --- 5. Train-Test Split ---
    # Stratify requires at least 2 members per class, might fail on very small dummy data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        print(f"\nData split into Train ({X_train.shape[0]} samples) and Test ({X_test.shape[0]} samples).")
    except ValueError as e:
        print(f"\nCould not stratify split (likely due to small sample/class size): {e}")
        print("Splitting without stratify for demonstration.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


    # %% --- 6. Build Full Pipeline with Feature Selection and KNN ---
    print("\nBuilding Pipeline with Encoding -> Feature Selection -> Scaling -> KNN...")
    encoder_transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('encoder', encoder_transformer),
        ('selector', SelectKBest(score_func=f_classif)),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ])

    # %% --- 7. Hyperparameter Tuning ---
    print("\nPerforming GridSearchCV for K (features) and n_neighbors (KNN)...")
    # Estimate max features after encoding (rough estimate)
    n_cat_features_after_encoding = sum(X_train[cat].nunique() for cat in categorical_features)
    n_num_features = len(numerical_features)
    max_features_approx = n_cat_features_after_encoding + n_num_features
    print(f"(Approximate max features after encoding: {max_features_approx})")

    k_selector_options = [k for k in [10, 15, 20, 25, 30, 35] if k < max_features_approx]
    if not k_selector_options: k_selector_options = [min(10, max_features_approx -1)] # Ensure at least one option
    if max_features_approx <= 1 : k_selector_options = [1] # Handle edge case

    param_grid = {
        'selector__k': k_selector_options,
        'classifier__n_neighbors': np.arange(3, min(20, len(X_train)), 2) # Ensure n_neighbors <= n_samples
    }
    print(f"GridSearch - Selector K options: {param_grid['selector__k']}")
    print(f"GridSearch - Classifier N options: {param_grid['classifier__n_neighbors']}")


    grid_search = GridSearchCV(pipeline, param_grid, cv=min(5, y_train.nunique(), len(y_train)//y_train.nunique() if y_train.nunique()>0 else 3), # Adjust cv based on data/classes
                               scoring='accuracy', n_jobs=-1, error_score='raise')

    best_pipeline = None # Initialize
    try:
        grid_search.fit(X_train, y_train)
        best_k_features = grid_search.best_params_['selector__k']
        best_n_neighbors = grid_search.best_params_['classifier__n_neighbors']
        best_cv_score = grid_search.best_score_
        print(f"\nBest parameters found:")
        print(f"  - Optimal number of features (selector__k): {best_k_features}")
        print(f"  - Optimal number of neighbors (classifier__n_neighbors): {best_n_neighbors}")
        print(f"Best cross-validation accuracy: {best_cv_score:.4f}")
        best_pipeline = grid_search.best_estimator_

        # --- Optional: Get names of selected features ---
        try:
            temp_pipeline = Pipeline(best_pipeline.steps[:-2]) # Encoder -> Selector
            temp_pipeline.fit(X_train, y_train)
            encoder_step = temp_pipeline.named_steps['encoder']
            selector_step = temp_pipeline.named_steps['selector']
            encoded_feature_names = encoder_step.get_feature_names_out()
            feature_mask = selector_step.get_support()
            selected_feature_names = encoded_feature_names[feature_mask]
            print(f"\nNames of the {best_k_features} selected features:")
            print(selected_feature_names[:20].tolist(), "..." if len(selected_feature_names)>20 else "")
        except Exception as feat_ex:
             print(f"Could not retrieve selected feature names: {feat_ex}")


    except ValueError as e:
        print(f"\nError during GridSearchCV: {e}")
        print("This might happen if 'k' in selector__k exceeds the number of features after encoding,")
        print("or due to issues with cross-validation folds (e.g., too few samples per class).")


    # %% --- 8. Evaluate Final Model ---
    if best_pipeline:
        print("\nEvaluating the best model found on the Test Set...")
        y_pred = best_pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        # Ensure y_test and y_pred have the same categories if possible
        labels = sorted(y.unique())
        print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap=palette_sequential) # Use sequential palette

        # --- FIX for Colorbar Ticks ---
        # Access the colorbar IF it exists and set tick colors
        if hasattr(disp, 'im_') and hasattr(disp.im_, 'colorbar') and disp.im_.colorbar:
            disp.im_.colorbar.ax.tick_params(colors=text_color)
        # --- End Fix ---

        plt.title(f'Confusion Matrix (KNN with {best_k_features} features, k={best_n_neighbors})', color=text_color)
        plt.xticks(rotation=45, color=text_color)
        plt.yticks(color=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.spines['top'].set_color(grid_color)
        ax.spines['right'].set_color(grid_color)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        plt.tight_layout()
        plt.show()

        # --- PCA Scatter Plot of Selected Features ---
        print("\nGenerating PCA Scatter Plot of selected features...")
        try:
            # 1. Get the processed data
            processing_pipeline = Pipeline(best_pipeline.steps[:-1]) # Encoder -> Selector -> Scaler
            X_test_processed = processing_pipeline.transform(X_test)

            # Check if processed data is valid (not empty, enough features)
            if X_test_processed is None or X_test_processed.shape[0] == 0 or X_test_processed.shape[1] < 2:
                 print("Skipping PCA plot: Not enough features after processing.")
            else:
                # 2. Apply PCA
                pca = PCA(n_components=2, random_state=42)
                X_test_pca = pca.fit_transform(X_test_processed)
                print(f"Explained variance ratio by 2 PCA components: {pca.explained_variance_ratio_.sum():.3f}")

                # 3. Create DataFrame for plotting
                pca_df = pd.DataFrame(data=X_test_pca, columns=['PCA Component 1', 'PCA Component 2'])
                pca_df['Salary Bracket'] = y_test.values

                # 4. Plot
                plt.figure(figsize=(10, 8))
                ax_pca = sns.scatterplot(
                    x='PCA Component 1', y='PCA Component 2',
                    hue='Salary Bracket',
                    palette=palette_categorical, # Use categorical palette
                    data=pca_df,
                    alpha=0.8, # Slightly increased alpha
                    s=60
                )
                plt.title('Test Data Projected onto First 2 PCA Components (Selected Features)', color=text_color, fontsize=15)
                plt.xlabel('PCA Component 1', color=text_color)
                plt.ylabel('PCA Component 2', color=text_color)
                plt.xticks(color=text_color)
                plt.yticks(color=text_color)
                ax_pca.spines['top'].set_color(grid_color)
                ax_pca.spines['right'].set_color(grid_color)
                ax_pca.spines['left'].set_color(grid_color)
                ax_pca.spines['bottom'].set_color(grid_color)

                # --- Add Legend with proper colors ---
                legend = plt.legend(title='Salary Bracket', title_fontsize='14')
                legend.get_title().set_color(text_color) # Set legend title color
                for text in legend.get_texts(): # Set legend item text color
                    text.set_color(text_color)
                # --- End Legend ---

                plt.grid(True, color=grid_color, linestyle='--', linewidth=0.5) # Add light grid
                plt.tight_layout()
                plt.show()
        except Exception as pca_ex:
            print(f"Could not generate PCA plot: {pca_ex}")

    else:
        print("\nSkipping final evaluation and PCA plot as model tuning failed.")

# %% --- End of Script ---
elif y is not None and y.nunique() <= 1:
    print("\nSkipping model training: Target variable has only one class after processing.")
else:
    print("\nSkipping model training and evaluation due to issues loading or processing data.")