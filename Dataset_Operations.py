import csv
import json
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_selection import f_regression, SelectKBest, VarianceThreshold, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def read_txt(filepath):
    """Reads data from a text file."""
    try:
        with open(filepath, 'r') as file:
            return file.readlines()
    except FileNotFoundError:
        return f"Error: File not found at {filepath}"
    except Exception as e:
        return f"An error occurred: {e}"

def read_csv(filepath):
    """Reads data from a CSV file."""
    try:
        with open(filepath, 'r', newline='') as file:
            reader = csv.reader(file)
            return list(reader)
    except FileNotFoundError:
        return f"Error: File not found at {filepath}"
    except Exception as e:
        return f"An error occurred: {e}"

def read_json(filepath):
    """Reads data from a JSON file."""
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return f"Error: File not found at {filepath}"
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in {filepath}"
    except Exception as e:
        return f"An error occurred: {e}"

def read_xml(filepath):
    """Reads data from an XML file and extracts employee data."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        employees = []  # List to store employee data
        for employee_element in root:
            employee = {}
            for child in employee_element:
                employee[child.tag] = child.text
            employees.append(employee)
        return employees
    except FileNotFoundError:
        return f"Error: File not found at {filepath}"
    except ET.ParseError:
        return f"Error: Invalid XML format in {filepath}"
    except Exception as e:
        return f"An error occurred: {e}"


# txt_data = read_txt("archive/Students.txt")
# print("TXT Data:\n", txt_data)
# print()
# csv_data = read_csv("archive/Students.csv")
# print("CSV Data:\n", csv_data)
# print()
# json_data = read_json("archive/Rolex_retail_original.json")[:5]
# print("JSON Data:\n", json_data)
# print()
# xml_data = read_xml("archive/employees.xml")[:5]
# print("XML Data:\n", xml_data)


def categorize_json_attributes(json_data):
    """
    TODO: Categorize attributes in JSON data as categorical or numerical.
    *Args:
        json_data (dict or list): The JSON data (loaded as a Python dictionary or list).
    *Returns:
        dict: A dictionary containing lists of categorical and numerical attributes.
    """
    categorical_attributes, numerical_attributes = set(), set()
    # If the data is a single dict, wrap it in a list.
    if isinstance(json_data, dict):
        json_data = [json_data]
    # If the data is in file, read it first.
    if json_data.endswith(".json"):
        json_data = read_json(json_data)

    for item in json_data:
        if isinstance(item, dict):
            for key, value in item.items():
                # Check booleans first, since bool is a subclass of int.
                if isinstance(value, bool):
                    categorical_attributes.add(key)
                elif isinstance(value, (int, float)):
                    numerical_attributes.add(key)
                else:
                    categorical_attributes.add(key)
    return {
        "categorical": list(categorical_attributes),
        "numerical": list(numerical_attributes)
    }
print()
# print(categorize_json_attributes("archive/Rolex_retail_original.json"))

def process_product_data(data, method='drop'):
    """
    Clean product data by converting numeric fields, stripping whitespace,
    and handling missing values.

    *Args:
        data (list of dict or pd.DataFrame): Records with fields:
            'Size', 'Reference', 'Collection', 'Description', 'RRP', 'Complication'
        method (str): How to handle missing values:
                      'drop' removes rows with missing data,
                      'fill' replaces missing values with defaults.

    *Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
    # Convert numeric fields
    for col in ['Size', 'RRP']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Strip whitespace from string fields
    for col in ['Reference', 'Collection', 'Description', 'Complication']:
        if col in df:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Handle missing values
    if method == 'drop':
        return df.dropna()
    elif method == 'fill':
        defaults = {
            'Reference': 'Unknown',
            'Collection': 'Unknown',
            'Description': 'No Description',
            'Complication': 'None',
            'Size': df['Size'].mean() if 'Size' in df else None,
            'RRP': df['RRP'].mean() if 'RRP' in df else None,
        }
        return df.fillna(defaults)
    else:
        raise ValueError("Method must be either 'drop' or 'fill'.")

Rolex_Retail_Watch_Data = read_json("archive/Rolex_retail_original.json")
# print(process_product_data(Rolex_Retail_Watch_Data, method='drop'))


def rescale_data(data, columns=None, method='minmax'):
    """
    Rescales numeric columns in a DataFrame.

    Args:
        data (pd.DataFrame or list of dict): The data to process.
        columns (list, optional): Specific columns to scale. If None, all numeric columns are scaled.
        method (str): Scaling method - 'minmax' (default) or 'standard'.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns re-scaled.
    """
    # Convert data to DataFrame if it's not already one.
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df = data.copy()
    # If no columns specified, select all numeric columns.
    if columns is None:
        columns = df.select_dtypes(include=['number'], exclude=['object']).columns.tolist()

    # Choose scaler based on the method.
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Choose 'minmax' or 'standard'.")

    df[columns] = scaler.fit_transform(df[columns])
    return df


# print(rescale_data(Rolex_Retail_Watch_Data))


def encode_data(data, columns=None, drop_first=False):
    """
    Encodes categorical columns using one-hot encoding.

    Args:
        data (pd.DataFrame or list of dict): The data to process.
        columns (list): The list of categorical columns to encode.
        drop_first (bool): Whether to drop the first category (default False).

    Returns:
        pd.DataFrame: A new DataFrame with categorical columns encoded.
    """

    # Convert data to DataFrame if it's not already one.
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    return pd.get_dummies(data, columns=columns, drop_first=drop_first)

# print(encode_data(Rolex_Retail_Watch_Data, columns=["Reference", "Collection", "Description", "RRP"]))


def feature_selection(X, y=None, method='variance', k=2, variance_threshold=0.0, encode_non_numeric=True):
    """
    Performs feature selection on a DataFrame of features.

    Args:
        X (pd.DataFrame or list of dict): Feature data.
        y (pd.Series, optional): Target variable for supervised selection.
        method (str): 'variance' for unsupervised VarianceThreshold,
                      'kbest' for supervised SelectKBest.
        k (int): Number of top features to select (used with 'kbest').
        variance_threshold (float): Threshold for VarianceThreshold (used with 'variance').
        encode_non_numeric (bool): If True, automatically encode non-numeric columns.

    Returns:
        pd.DataFrame: DataFrame containing the selected features.
    """
    # Convert input to DataFrame if necessary.
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if encode_non_numeric:
        # Convert any list values in non-numeric columns to tuples.
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        # Now encode non-numeric features using one-hot encoding.
        X = pd.get_dummies(X)

    if method == 'variance':
        selector = VarianceThreshold(threshold=variance_threshold)
        X_selected = selector.fit_transform(X)
        features = X.columns[selector.get_support(indices=True)]
        return pd.DataFrame(X_selected, columns=features)

    elif method == 'kbest':
        if y is None:
            raise ValueError("A target variable y must be provided for 'kbest' selection.")
        score_func = f_regression if pd.api.types.is_numeric_dtype(y) else f_classif
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        features = X.columns[selector.get_support(indices=True)]
        return pd.DataFrame(X_selected, columns=features)
    else:
        raise ValueError("Invalid method. Use 'variance' or 'kbest'.")


dataframe = pd.DataFrame(Rolex_Retail_Watch_Data)
X_Unsupervised = dataframe.drop("RRP", axis=1)

selected_unsupervised = feature_selection(X_Unsupervised)
print(f"Un-Supervised Feature Selection (Variance Threshold)\n")
print(selected_unsupervised, selected_unsupervised.columns)
print()

X_Supervised = dataframe.drop("RRP", axis=1)
Y_Supervised = dataframe["RRP"]
selected_supervised = feature_selection(X_Supervised, Y_Supervised, method='kbest', k=2)
print(f"Supervised Feature selection (select k-best)\n")
print(selected_supervised, selected_supervised.columns)
