import pandas as pd
import os
import numpy as np
import pickle
import plotly.express as px
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_csv(file_path, encoding='utf-8'):
    return pd.read_csv(file_path, encoding=encoding)

def load_excel(file_path, **kwargs):
    return pd.read_excel(file_path, **kwargs)


def prepare_data(df):
    X = df.drop(columns=['ad_price_cap_log'])
    y = df['ad_price_cap_log']
    return train_test_split(X, y, test_size=0.20, random_state=42)

def load_model(file_path):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    return model

def make_predictions(model, X_test):
    return model.predict(X_test)
