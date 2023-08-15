import ast
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ---------- #
# cleaner.py #
# ---------- #

# Custom transformer to drop columns with missing values
class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


# Custom transformer to remove rows with the same ID
class DropRowsWithSameIDTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_column):
        self.id_column = id_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        unique_indices = X[self.id_column].drop_duplicates().index
        X_cleaned = X.loc[unique_indices]
        return X_cleaned


def to_datetime_transformer(data, columns):
    """transformer function to convert columns to datetime format"""
    data[columns] = data[columns].apply(pd.to_datetime,
                                        origin='unix', unit='s')
    return data


def extract_category_info(category_str):
    """Extract "parent_name" and "name" from the "category" column"""
    category_dict = ast.literal_eval(category_str)
    words = category_dict['slug'].split('/')
    if len(words) < 2:
        words.append('no subcategory')
    return words


def category_transformation(data):
    """Define a separate function for the "category" column transformation"""
    return data['category'].apply(lambda y: pd.Series(extract_category_info(y),
                                                     index=['main_category',
                                                            'sub_category']))


def calculate_usd_goal(data):
    """transformer function to calculate 'usd_goal'"""
    usd_goal = data['goal'] * data['static_usd_rate']
    return pd.DataFrame(usd_goal, columns=['usd_goal'])


def calculate_usd_pledged(data):
    """transformer function to calculate 'usd_pledged'"""
    usd_pledged = data['pledged'] * data['static_usd_rate']
    return pd.DataFrame(usd_pledged, columns=['usd_pledged'])


# ----------------- #
# build_features.py #
# ----------------- #

def calculate_name_length(data):
    return data['name'].str.split().str.len().to_frame('name_length')


def calculate_description_length(data):
    description_length = data['blurb'].str.split().str.len()
    # Make sure to replace null values for length 0
    return description_length.fillna(0).to_frame('description_length')


# Time between creating and launching a project (hours)
def calculate_creation_to_launch_hours(data):
    creation_to_launch_hours = (data['launched_at'] - data['created_at']).dt.round('h') / np.timedelta64(1, 'h')
    return pd.DataFrame(creation_to_launch_hours, columns=['creation_to_launch_hours'])


# Campaign duration (hours)
def calculate_campaign_hours(data):
    campaign_hours = (data['deadline'] - data['launched_at']).dt.round('d').dt.round('h') / np.timedelta64(1, 'h')
    return pd.DataFrame(campaign_hours, columns=['campaign_hours'])


# To ensure consistency and prevent data leakage, we should calculate
# the medians on the training set and use the same median values for
# transforming the validation and test sets

def calculate_diff_main_category_goal(data, medians):
    main_categories = data['main_category'].values
    median_main = medians['main_category'][main_categories].values
    return abs(data['usd_goal'] - median_main).to_frame('diff_main_category_goal')


def calculate_diff_sub_category_goal(data, medians):
    sub_categories = data['sub_category'].values
    median_sub = medians['sub_category'][sub_categories].values
    return abs(data['usd_goal'] - median_sub).to_frame('diff_sub_category_goal')


class MedianDiffCalculatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = None

    def fit(self, X, y=None):
        self.medians = {
            'main_category': X.groupby("main_category")['usd_goal'].median(),
            'sub_category': X.groupby("sub_category")['usd_goal'].median()
        }
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed['diff_main_category_goal'] = calculate_diff_main_category_goal(X_transformed, self.medians)
        X_transformed['diff_sub_category_goal'] = calculate_diff_sub_category_goal(X_transformed, self.medians)
        return X_transformed


# adds 1 to each value before taking the logarithm, which handles
# cases where the data contains zeros or negative values.
def turn_to_log(X):
    return np.log1p(X)