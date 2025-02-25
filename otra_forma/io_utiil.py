import pandas as pd
import numpy as np
from lifelines.utils import concordance_index

def clean_column_headers(df):
   replacements = {
       '(': '_', ')': '_', '+': 'plus', ',': '_',
       '-': '_', '.': '_', '/': '_',
       '<': 'less_than', '=': 'equals', '>': 'greater_than'
   }
   df.columns = df.columns.str.translate(str.maketrans(replacements))
   return df

def load_dataframes(train_path='train.csv', test_path='test.csv'):
   train_df = pd.read_csv(train_path)
   test_df = pd.read_csv(test_path)

   train_df = clean_column_headers(train_df)
   test_df = clean_column_headers(test_df)

   special_columns = ['ID', 'race_group', 'efs', 'efs_time']
   procesable_columns = [col for col in train_df.columns if col not in special_columns]

   train_df['origin'] = 'train'
   test_df['origin'] = 'test'

   combined_df = pd.concat([train_df, test_df], ignore_index=True)

   for col in procesable_columns:
       if combined_df[col].dtype == 'object':
           one_hot = pd.get_dummies(combined_df[col], prefix=col, dummy_na=True, drop_first=False)
           combined_df = pd.concat([combined_df, one_hot], axis=1)
           combined_df.drop(columns=[col], inplace=True)
       elif pd.api.types.is_numeric_dtype(combined_df[col]):
           combined_df[col] = combined_df[col].fillna(combined_df[col].median())

   final_train = combined_df[combined_df['origin'] == 'train']
   final_test = combined_df[combined_df['origin'] == 'test']

   processed_train_df = final_train[special_columns + list(final_train.columns[~final_train.columns.isin(special_columns + ['origin'])])]
   processed_test_df = final_test[special_columns + list(final_test.columns[~final_test.columns.isin(special_columns + ['origin'])])]

   #processed_train_df = processed_train_df.drop(['origin'], axis=1)
   #processed_test_df = processed_test_df.drop(['origin'], axis=1)

   return processed_train_df, processed_test_df

def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            print('error!!!!!!!!!!!!')
            exit(1)
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
            merged_df_race[interval_label],
            -merged_df_race[prediction_label],
            merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))

def get_score(x, prediction):
    prediction = pd.DataFrame({
        'prediction': prediction
    })

    return score(x.copy(),prediction,'ID')

def drop_top_n_corr(df, n=10, col='efs_time', exclude=None):
    c = df.corr()[col].abs().drop(col).sort_values(ascending=False)
    return df.drop(columns=c.head(n).index.tolist())

def train(model, xtrain ):
    print('training '+ str(xtrain.shape[0]) + ' lines')
    y_train = np.array([(bool(e), t) for e, t in zip(xtrain["efs"], xtrain["efs_time"])],
                       dtype=[("event", "bool"), ("time", "float")])

    model.fit(xtrain,y_train)
