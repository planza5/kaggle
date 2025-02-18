from statsmodels.stats.correlation_tools import corr_thresholded

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import os
import gc

def to_32bit(df):
    for col in df.select_dtypes(include=['int64','float64']).columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype('int32')
        else:
            df[col] = df[col].astype('float32')
    return df

def clean_non_numeric_columns(df):
    for column in df.select_dtypes(exclude=['number']).columns:
        df[column] = df[column].apply(
            lambda x: re.sub(r'_+', '_', re.sub(r'[^\w]', '_', str(x))) if pd.notna(x) else x
        )
    return df

def lowercase_headers(df, exclude):
    df.columns = [col if col in exclude else col.lower() for col in df.columns]
    return df


def one_hot_encode_train_test(train, test):
    n_train_rows = train.shape[0]
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    df_encoded = []
    column_order = []
    for column in combined.columns:
        if column == 'race_group':
            df_encoded.append(combined[[column]])
            column_order.append(column)
            continue
        if combined[column].dtype in ['object', 'category']:
            temp_col = combined[column].fillna('missing')
            dummies = pd.get_dummies(temp_col, prefix=column, dtype=int)
            df_encoded.append(dummies)
            column_order.extend(dummies.columns)
        else:
            df_encoded.append(combined[[column]])
            column_order.append(column)
    combined_encoded = pd.concat(df_encoded, axis=1)
    combined_encoded = combined_encoded[column_order]
    train_encoded = combined_encoded.iloc[:n_train_rows].copy()
    test_encoded = combined_encoded.iloc[n_train_rows:].copy()
    missing_columns = set(train_encoded.columns) - set(test_encoded.columns)
    for col in missing_columns:
        test_encoded[col] = 0
    test_encoded = test_encoded[train_encoded.columns]
    return train_encoded, test_encoded

def autohot_encoding(df, exclude=None):
    if exclude is None:
        exclude = []
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            col_index = df.columns.get_loc(col)
            df.insert(col_index, col, df.pop(col))
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        unique_values = df[col].dropna().unique()
        new_columns = {}
        for value in unique_values:
            column_name = f"{col}_{value}"
            new_columns[column_name] = (df[col] == value).astype(int)
        nan_column_name = f"{col}_nan"
        new_columns[nan_column_name] = df[col].isna().astype(int)
        col_index = df.columns.get_loc(col)
        for i, (new_col_name, new_col_data) in enumerate(new_columns.items()):
            df.insert(col_index + i, new_col_name, new_col_data)
        df.drop(columns=[col], inplace=True)
    return df

def replace_special_characters(df):
    def replace_chars(value):
        if pd.isna(value):
            return value
        value = str(value)
        value = re.sub(r'\+', '_plus_', value)
        value = re.sub(r'-', '_minus_', value)
        value = re.sub(r'>', '_gt_', value)
        value = re.sub(r'<', '_lt_', value)
        return value
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column] = df[column].apply(replace_chars)
    return df

def replace_nan_median(df, exclude):
    columnas_numericas = [col for col in df.select_dtypes(include=['number']).columns if col not in exclude]

    for col in columnas_numericas:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    return df

def replace_nan_moda(df, exclude):
    columnas_numericas = [col for col in df.select_dtypes(include=['number']).columns if col not in exclude]

    for col in columnas_numericas:
        moda = df[col].mode()[0]
        df[col] = df[col].fillna(moda)

    return df


def clean(df):
    df = replace_special_characters(df)
    df = clean_non_numeric_columns(df)
    return df

def normalization(df, exclude_columns=None):
    numeric_columns = df.select_dtypes(exclude=['object']).columns
    columns_to_normalize = [
        col for col in numeric_columns
        if col not in exclude_columns and not (set(df[col].unique()) <= {0, 1})
    ]
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def check_all_nan_cols(df, df_name):
    all_nan_cols = [col for col in df.columns if df[col].isnull().all()]
    if all_nan_cols:
        print(f"En {df_name}, las siguientes columnas están 100% en NaN:\n{all_nan_cols}")
    else:
        print(f"En {df_name} no hay columnas 100% NaN")

def best_correlated_features(df, target_column, n=10, not_eligible=None):
    if not_eligible is None:
        not_eligible = []
    columns_a_excluir = ["race_group"] + not_eligible
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=columns_a_excluir, errors="ignore")
    if target_column not in numeric_df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no es numérica o no se encontró en el DataFrame.")
    corr_series = numeric_df.corr()[target_column].drop(labels=[target_column])
    positive_corr = corr_series[corr_series > 0].sort_values(ascending=False)
    negative_corr = corr_series[corr_series < 0].sort_values(ascending=True)
    half = n // 2
    top_positive = positive_corr.head(half).index.tolist()
    top_negative = negative_corr.head(half).index.tolist()
    return top_positive + top_negative

def identify_columns_to_drop(df: pd.DataFrame, corr_limit: float, to_exclude: list) -> list:
    to_drop = []
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    columns_to_consider = [col for col in numeric_columns if col not in to_exclude]
    corr_matrix = df[columns_to_consider].corr().abs()
    for column in corr_matrix.columns:
        for other_column in corr_matrix.columns:
            if column == other_column:
                continue
            if corr_matrix.loc[column, other_column] > corr_limit:
                if other_column not in to_drop:
                    to_drop.append(other_column)
    return to_drop

def get_worst_corrs(dataframe, col_target, cols_to_exclude=[], corr_limit=0.1):
    if col_target is None:
        raise ValueError("col_target no puede ser nulo.")
    if col_target in cols_to_exclude:
        raise ValueError("col_target no puede estar en cols_to_exclude.")
    correlations = {}
    for column in dataframe.columns:
        if column in cols_to_exclude or dataframe[column].dtype == 'object':
            continue
        if column != col_target:
            corr = dataframe[column].corr(dataframe[col_target])
            if abs(corr) < corr_limit:
                correlations[column] = corr
    return correlations

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            print(f'Submission column {col} must be a number')
            exit(1)
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        c_index_race = concordance_index(
            merged_df_race[interval_label],
            -merged_df_race[prediction_label],
            merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

def prepare_survival_data(df):
    y = np.array(
        [(bool(event), time) for event, time in zip(df["efs"], df["efs_time"])],
        dtype=[("event", "bool"), ("time", "float")]
    )
    return y

def predict2(model, xdata):
    print('predicting....')
    return  model.predict(xdata.drop(['ID','efs','efs_time','race_group'],axis=1))


def predict(model, xdata):
    print('predicting....')
    chf =  model.predict_cumulative_hazard_function(xdata.drop(['ID','efs','efs_time','race_group'],axis=1))
    risk_scores = np.array([fn(fn.x[-1]) for fn in chf])
    return risk_scores

def train(model, xtrain ,weights=None):
    print('training....')
    ytrain = np.array([(bool(e), t) for e, t in zip(xtrain["efs"], xtrain["efs_time"])],
                       dtype=[("event", "bool"), ("time", "float")])

    model.fit(xtrain.drop(['ID','efs','efs_time','race_group'],axis=1),ytrain,sample_weight=weights)
    return model

def get_score(x, prediction):
    prediction = pd.DataFrame({
        'ID': x['ID'],
        'prediction': prediction
    })

    return score(x.copy(),prediction,'ID')


def calcular_cindex_por_grupo(X_valid, predict, id_cols=['ID', 'race_group', 'efs', 'efs_time']):
    result = X_valid[id_cols].copy()
    result['risk_score'] = predict
    groups = result.groupby('race_group')
    cindex_per_group = {}
    for group_name, group_data in groups:
        c_index_group = concordance_index(
            group_data['efs_time'],
            -group_data['risk_score'],
            group_data['efs']
        )
        cindex_per_group[group_name] = c_index_group
        print(f"Grupo: {group_name}, C-index: {c_index_group:.3f}")
    return cindex_per_group


exceptions=['dri_score']

path_data_train='/kaggle/input/equity-post-HCT-survival-predictions/train.csv'
path_data_test='/kaggle/input/equity-post-HCT-survival-predictions/test.csv'

if 'KAGGLE_URL_BASE' in os.environ:
    print("Ejecutándose en un Notebook de Kaggle")
else:
    print("Ejecutándose de forma local")
    path_data_train = 'train.csv'
    path_data_test = 'test.csv'

train_data = pd.read_csv(path_data_train)
test_data = pd.read_csv(path_data_test)

#train_data = to_32bit(train_data)
#test_data = to_32bit(test_data)

train_data = clean(train_data)
test_data = clean(test_data)

train_data, test_data = one_hot_encode_train_test(train_data, test_data)

train_data = lowercase_headers(train_data,exclude=['ID'])
test_data = lowercase_headers(test_data,exclude=['ID'])

train_data = replace_nan_median(train_data,exclude=['ID','efs','efs_time'])
test_data = replace_nan_median(test_data,exclude=['ID','efs','efs_time'])

features = best_correlated_features(train_data,"efs_time",60,['ID','efs'])
features = ['ID','race_group'] + features + ['efs','efs_time']

train_data = train_data[features].copy()
test_data = test_data[features].copy()
print('features=',len(features))


train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=42)

X_train = train_df[features]
X_test = valid_df[features]

rsf = RandomSurvivalForest(n_estimators=30, n_jobs=2, min_samples_split=15, min_samples_leaf=10, random_state=42,
                           low_memory=False, verbose=4)

rsf = train(rsf, X_train)
prediction = predict2(rsf, X_test)

print('risk score ',get_score(X_test,prediction))

group_cindexes = calcular_cindex_por_grupo(X_test, prediction)
epsilon = 1e-6

weights = {group: 1 / (cindex + epsilon) for group, cindex in group_cindexes.items()}
total_weight = sum(weights.values())
weights = {group: weight / total_weight for group, weight in weights.items()}

mapping_weights = X_train['race_group'].map(weights).values

if mapping_weights is None or np.any(np.isnan(mapping_weights)):
    raise ValueError("Algunos valores de race_group no tienen pesos asociados.")

rsf = train(rsf,X_train,weights=mapping_weights)
prediction = predict(rsf, X_test)
print(get_score(X_test, prediction))
